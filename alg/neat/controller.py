import pickle
from typing import Any, Optional, Tuple

import neat  # type: ignore
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd  # type: ignore

from alg.controller import Controller


class NEATController(Controller):
    """NEAT-Python用のController実装

    このクラスはNEAT-Pythonで訓練されたFeedForwardNetworkをController抽象クラスとして
    ラップし、numpy配列ベースの評価インターフェースを提供します。

    Attributes
    ----------
    net : neat.nn.FeedForwardNetwork
        NEAT FeedForwardNetwork
    """

    def __init__(self, net: neat.nn.FeedForwardNetwork):
        """
        NEATControllerを初期化する

        Parameters
        ----------
        net : neat.nn.FeedForwardNetwork
            NEAT FeedForwardNetwork
        """
        self.net = net

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        行動を選択する（評価時用のnumpy版、Controllerインターフェース実装）

        Parameters
        ----------
        observation : np.ndarray
            観測値（shape: (obs_dim,) or (batch_size, obs_dim)）
        deterministic : bool, optional
            確定的方策を用いるかどうか, by default False
            Note: NEATは決定論的なネットワークなので、このパラメータは無視される

        Returns
        -------
        action : np.ndarray
            行動（shape: (action_dim,) or (batch_size, action_dim)）
        """
        # 1次元の場合（単一観測）
        if observation.ndim == 1:
            return np.array(self.net.activate(observation))

        # 2次元の場合（バッチ処理）
        actions = []
        for obs in observation:
            actions.append(self.net.activate(obs))
        return np.array(actions)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> Tuple["NEATController", None]:
        """
        ファイルからNEATControllerをロードする（Controllerインターフェース実装）

        Parameters
        ----------
        path : str
            neat-checkpointファイルのパス
        **kwargs : dict
            config_path (str): NEAT configファイルのパス（必須）

        Returns
        -------
        controller : NEATController
            ロードされたコントローラー
        normalization_stats : None
            NEATは観測値の正規化を行わないため、常にNone

        Raises
        ------
        ValueError
            config_pathが提供されていない場合、またはベストgenomeが見つからない場合
        """
        config_path: Optional[str] = kwargs.get("config_path")

        if config_path is None:
            raise ValueError("config_path must be provided")

        # NEAT configをロード
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        # checkpointからpopulationを復元
        p = neat.Checkpointer.restore_checkpoint(path)

        # ベストgenomeを取得
        best_genome = None
        best_fitness = float("-inf")
        for genome_id, genome in p.population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome

        if best_genome is None:
            raise ValueError(f"No genome with fitness found in checkpoint: {path}")

        # ネットワーク生成
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)

        return cls(net), RunningMeanStd()
