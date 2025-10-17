import glob
import json
import os
from typing import Dict, Literal, Optional

import numpy as np
from evogym import draw, get_full_connectivity, get_uniform, has_actuator, hashable, is_connected  # type: ignore

from alg.coea.coea_utils import StructureMetadata


class Structure:
    """ロボットの形態と訓練状態を表すクラス。

    このクラスはロボットの物理的な構造（body と connections）と、
    訓練状態、評価スコア、適応度などのメタデータを管理する。

    Attributes:
        save_path: 構造データが保存されるディレクトリパス
        body: ロボットの形態を表すnumpy配列（グリッド形式）
        connections: ロボットのボクセル間の接続を表すnumpy配列
        metadata: 訓練状態、死亡フラグ、評価スコアを含むメタデータ
    """

    def __init__(self, save_path: str, body: np.ndarray, connections: np.ndarray, save: bool = True):
        """ロボット構造を初期化する。

        Args:
            save_path: 構造データを保存するディレクトリパス
            body: ロボットの形態を表す配列
            connections: ボクセル間の接続を表す配列
            save: Trueの場合、ディレクトリを作成してデータを保存
                  Falseの場合、既存のメタデータを読み込む
        """

        self.save_path = save_path
        self.body = body
        self.connections = connections

        if save:
            os.mkdir(self.save_path)
            np.save(os.path.join(self.save_path, "body.npy"), body)
            np.save(os.path.join(self.save_path, "connections.npy"), connections)
            self.metadata = StructureMetadata(is_trained=False, is_died=False)
            self.dump_metadata()
        else:
            with open(os.path.join(self.save_path, "metadata.json"), "r") as f:
                metadata_dict = json.load(f)
            self.metadata = StructureMetadata(**metadata_dict)

    def get_latest_controller_path(self) -> str:
        """最新の訓練済みコントローラのファイルパスを取得する。

        Returns:
            最も新しいコントローラファイルのパス

        Raises:
            AssertionError: コントローラファイルが見つからない場合
        """
        controller_paths = sorted(glob.glob(os.path.join(self.save_path, "controller_*.pt")))
        assert controller_paths, f"Controller for {self.save_path} is not found."
        return max(controller_paths, key=os.path.getctime)

    @classmethod
    def from_save_path(cls, save_path: str) -> "Structure":
        """保存されたディレクトリからStructureインスタンスを読み込む。

        Args:
            save_path: 構造データが保存されているディレクトリパス

        Returns:
            読み込まれたStructureインスタンス
        """
        body = np.load(os.path.join(save_path, "body.npy"))
        connections = np.load(os.path.join(save_path, "connections.npy"))
        return cls(save_path, body, connections, save=False)

    def has_fought(self, opponent_id: int) -> bool:
        """指定された対戦相手と既に対戦したかを確認する。

        Args:
            opponent_id: 対戦相手のID

        Returns:
            対戦済みの場合True、そうでなければFalse
        """
        return opponent_id in self.metadata.scores

    def set_score(self, opponent_id: int, score: float) -> None:
        """対戦相手に対するスコアを記録する。

        Args:
            opponent_id: 対戦相手のID
            score: 評価スコア
        """
        self.metadata.scores[opponent_id] = score
        self.dump_metadata()

    def delete_score(self, opponent_id: int) -> None:
        """特定の対戦相手に対するスコアを削除する。

        Args:
            opponent_id: スコアを削除する対戦相手のID
        """
        del self.metadata.scores[opponent_id]
        self.dump_metadata()

    @property
    def fitness(self) -> Optional[float]:
        """ロボットの適応度を計算する。

        適応度は、記録されたすべての対戦スコアの平均値として計算される。
        ロボットが死亡している場合、またはスコアが記録されていない場合はNoneを返す。

        Returns:
            適応度（スコアの平均値）、または計算不可の場合はNone
        """
        if self.is_died:
            return None
        if not self.metadata.scores:
            return None
        values = list(self.metadata.scores.values())
        return float(np.mean(values))

    @property
    def is_trained(self) -> bool:
        """ロボットが訓練済みかどうかを取得する。

        Returns:
            訓練済みの場合True、そうでなければFalse
        """
        return self.metadata.is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        """ロボットの訓練状態を設定し、メタデータを保存する。

        Args:
            value: 訓練済みフラグ
        """
        self.metadata.is_trained = value
        self.dump_metadata()

    @property
    def is_died(self) -> bool:
        """ロボットが死亡（淘汰）されたかどうかを取得する。

        Returns:
            死亡している場合True、そうでなければFalse
        """
        return self.metadata.is_died

    @is_died.setter
    def is_died(self, value: bool) -> None:
        """ロボットの死亡状態を設定し、メタデータを保存する。

        Args:
            value: 死亡フラグ
        """
        self.metadata.is_died = value
        self.dump_metadata()

    def dump_metadata(self) -> None:
        """メタデータをJSONファイルに保存する。"""
        with open(os.path.join(self.save_path, "metadata.json"), "w") as f:
            json.dump(self.metadata.model_dump(), f, indent=4)


def mutate(
    structure: Structure,
    child_save_path: str,
    population_structure_hashes: Dict[str, bool],
    mutation_rate: float = 0.1,
    num_attempts: int = 10,
) -> Optional[Structure]:
    """親構造を突然変異させて新しい子構造を生成する。

    各ボクセルが一定の確率で突然変異し、生成された構造が
    接続性とアクチュエータの有無の条件を満たし、かつ
    集団内に重複しない場合に新しい構造として返される。

    Args:
        structure: 親となる構造
        child_save_path: 子構造を保存するディレクトリパス
        population_structure_hashes: 集団内の既存構造のハッシュセット
        mutation_rate: 各ボクセルが変異する確率（デフォルト: 0.1）
        num_attempts: 有効な子を生成するための最大試行回数（デフォルト: 10）

    Returns:
        生成された有効な子構造、または生成に失敗した場合はNone
    """

    body = structure.body.copy()

    pd = get_uniform(5)
    pd[0] = 0.6

    for n in range(num_attempts):
        for i in range(body.shape[0]):
            for j in range(body.shape[1]):
                mutation = [mutation_rate, 1 - mutation_rate]
                if draw(mutation) == 0:
                    body[i][j] = draw(pd)

        if is_connected(body) and has_actuator(body) and hashable(body) not in population_structure_hashes:
            connections = get_full_connectivity(body)
            return Structure(child_save_path, body, connections)

    return None


class DummyRobotStructure:
    """固定された形態を持つダミーロボット構造。

    このクラスは進化しない固定の対戦相手として使用される。
    事前定義された形態タイプから選択でき、訓練や適応度計算のための
    固定ベースラインとして機能する。

    Attributes:
        body_type: ロボット形態のタイプ
        body: ロボットの形態を表すnumpy配列
        connections: ボクセル間の接続を表すnumpy配列
    """

    def __init__(self, body_type: Literal["rigid_4x4", "soft_4x4", "rigid_5x5", "soft_5x5"]):
        """指定されたタイプのダミーロボット構造を初期化する。

        Args:
            body_type: ロボット形態のタイプ
                - "rigid_4x4": 4x4の剛体ボクセル
                - "soft_4x4": 4x4の柔軟ボクセル
                - "rigid_5x5": 5x5の剛体ボクセル
                - "soft_5x5": 5x5の柔軟ボクセル

        Raises:
            ValueError: 無効なbody_typeが指定された場合
        """

        self.body_type = body_type

        if body_type == "rigid_4x4":
            self.body = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                ]
            )
        elif body_type == "soft_4x4":
            self.body = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [2, 2, 2, 2, 0],
                    [2, 2, 2, 2, 0],
                    [2, 2, 2, 2, 0],
                    [2, 2, 2, 2, 0],
                ]
            )
        elif body_type == "rigid_5x5":
            self.body = np.ones((5, 5))
        elif body_type == "soft_5x5":
            self.body = np.full((5, 5), 2)
        else:
            raise ValueError(f"Invalid body_type: {body_type}")

        self.connections = get_full_connectivity(self.body)
