import csv
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from evogym import hashable, sample_robot  # type: ignore

from alg.coea.structure import Structure, mutate
from utils import AgentID


class Population:
    """進化アルゴリズムのためのロボット構造の集団を管理するクラス。

    このクラスは以下を含む集団の進化サイクルを扱う:
    - ランダムなロボット構造の初期化
    - 世代を通じた適応度スコアの追跡
    - 個体の選択と再生産
    - 集団の状態の永続化と読み込み

    Attributes:
        agent_id: エージェントの識別子（例: "robot_1" or "robot_2"）
        save_path: 集団データが保存されるディレクトリパス
        csv_path: 適応度履歴を含むCSVファイルのパス
        structures: 集団を表すStructureオブジェクトのリスト
        population_structure_hashes: 重複する形態を防ぐためのハッシュセット
        generation: 現在の世代番号
    """

    def __init__(
        self,
        agent_id: AgentID,
        save_path: str,
        pop_size: int,
        robot_shape: Tuple[int, int],
        is_continuing: bool = False,
    ):
        """ロボット構造の集団を初期化する。

        Args:
            agent_id: この集団が表すエージェントの識別子
            save_path: 集団データが保存されるディレクトリパス
            pop_size: 集団内の個体数
            robot_shape: ロボットのグリッドサイズを定義するタプル (高さ, 幅)
            is_continuing: Trueの場合、save_pathから既存の集団を読み込む
                          Falseの場合、新しいランダムな集団を作成する
        """

        self.agent_id = agent_id
        self.save_path = save_path
        self.csv_path = os.path.join(self.save_path, "fitnesses.csv")
        self.structures: List[Structure] = []
        self.population_structure_hashes: Dict[str, bool] = {}
        self.generation = 0

        if not is_continuing:

            # create log files
            os.mkdir(self.save_path)
            with open(self.csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["generation"] + [f"id{i:02}" for i in range(pop_size)])
            generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
            os.mkdir(generation_path)

            # generate a population
            for robot_id in range(pop_size):

                body, connections = sample_robot(robot_shape)
                while hashable(body) in self.population_structure_hashes:
                    body, connections = sample_robot(robot_shape)

                self.structures.append(Structure(os.path.join(generation_path, f"id{robot_id:02}"), body, connections))
                self.population_structure_hashes[hashable(body)] = True

        else:
            assert os.path.exists(self.save_path)
            assert os.path.exists(self.csv_path)
            generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")

            while os.path.exists(generation_path):

                for robot_id in range(pop_size):
                    structure_path = os.path.join(generation_path, f"id{robot_id:02}")

                    if self.generation == 0:
                        assert os.path.exists(structure_path)
                        structure = Structure.from_save_path(structure_path)
                        self.structures.append(structure)
                    else:
                        if not os.path.exists(structure_path):
                            continue
                        structure = Structure.from_save_path(structure_path)
                        self.structures[robot_id] = structure

                    self.population_structure_hashes[hashable(structure.body)] = True

                self.generation += 1
                generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")

            self.generation -= 1

    def update(self, num_survivors: int, num_reproductions: int) -> List[int]:
        """選択と再生産を行い、次世代を作成する。

        このメソッドは以下を実行する:
        1. 適応度に基づいて上位の個体を選択
        2. 生存しなかった個体を死亡としてマーク
        3. 新しい世代のディレクトリを作成
        4. 生存者を突然変異させて子孫を生成

        Args:
            num_survivors: 保持する上位個体の数
            num_reproductions: 生存者から作成する子孫の数

        Returns:
            生存しなかった個体のインデックスのリスト

        Raises:
            ValueError: いずれかの適応度がNoneの場合（すべての個体が評価されていない）
            RuntimeError: 最大試行回数後に有効な子を生成できなかった場合
        """
        logging.info(f"## Updating {self.agent_id} population")

        # selection
        if any(fitness is None for fitness in self.fitnesses):
            raise ValueError("All fitnesses must be set before updating the population.")
        fitnesses_ = np.array(self.fitnesses)
        sorted_args = list(np.argsort(-fitnesses_))
        survivors = sorted_args[:num_survivors]
        non_survivors = sorted_args[num_survivors:]
        logging.info(f"Survivors: {','.join(map(str, survivors))}")
        for robot_id in non_survivors:
            self.structures[robot_id].is_died = True

        # reproduce
        self.generation += 1
        generation_path = os.path.join(self.save_path, f"generation{self.generation:02}")
        os.mkdir(generation_path)

        for child_robot_id in non_survivors[:num_reproductions]:
            child_save_path = os.path.join(generation_path, f"id{child_robot_id:02}")
            num_attempts = 100
            for _ in range(num_attempts):
                parent_robot_id = random.choice(survivors)
                child = mutate(self.structures[parent_robot_id], child_save_path, self.population_structure_hashes)
                if child is not None:
                    break
            else:
                raise RuntimeError("Failed to generate a child.")

            logging.info(f"Reproduced {parent_robot_id} -> {child_robot_id}")
            self.structures[child_robot_id] = child
            self.population_structure_hashes[hashable(child.body)] = True

        return non_survivors

    def get_training_indices(self) -> List[int]:
        """まだ訓練されていない個体のインデックスを取得する。

        Returns:
            未訓練の構造のインデックスのリスト
        """
        indices = [idx for idx, structure in enumerate(self.structures) if not structure.is_trained]
        return indices

    def get_evaluation_indices(self) -> List[int]:
        """評価対象となる個体のインデックスを取得する。

        Returns:
            訓練済みで死亡とマークされていない構造のインデックスのリスト
        """
        indices = [
            idx for idx, structure in enumerate(self.structures) if structure.is_trained and not structure.is_died
        ]
        return indices

    def set_score(self, self_robot_id: int, opponent_robot_id: int, score: float) -> None:
        """対戦相手に対するロボットの評価スコアを記録する。

        Args:
            self_robot_id: この集団内のロボットのインデックス
            opponent_robot_id: 対戦相手のロボットのID
            score: 評価から得られたパフォーマンススコア
        """
        self.structures[self_robot_id].set_score(opponent_robot_id, score)

    def delete_score(self, opponent_robot_id: int) -> None:
        """すべての構造から特定の対戦相手に対するスコアを削除する。

        これは通常、対戦相手が死亡し、適応度計算に寄与すべきでなくなった
        場合に呼び出される。

        Args:
            opponent_robot_id: スコアを削除すべき対戦相手のID
        """
        for structure in self.structures:
            if structure.has_fought(opponent_robot_id):
                structure.delete_score(opponent_robot_id)

    @property
    def fitnesses(self) -> List[Optional[float]]:
        """集団内のすべての個体の適応度を取得する。

        Returns:
            適応度のリスト（個体が死亡または未評価の場合はNone）
        """
        return [structure.fitness for structure in self.structures]

    def dump_fitnesses(self) -> None:
        """現在の世代の適応度をCSVログファイルに追記する。"""
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.generation] + self.fitnesses)

    def __getitem__(self, index: int) -> Structure:
        """インデックスで構造にアクセスする。

        Args:
            index: 集団内の構造のインデックス

        Returns:
            指定されたインデックスの構造
        """
        return self.structures[index]
