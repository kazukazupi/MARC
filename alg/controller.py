from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class Controller(ABC):
    """制御器の抽象クラス（numpy.ndarrayベース）

    このクラスは、PPOベースの制御器やNEAT-Pythonベースの制御器など、
    異なるアルゴリズムで訓練された制御器の共通インターフェースを提供します。

    評価時には、観測値と行動はnumpy.ndarrayで扱われます。
    """

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """行動を選択する

        Parameters
        ----------
        observation : np.ndarray
            観測値（1次元または2次元配列）
            - 1次元の場合: shape (obs_dim,)
            - 2次元の場合: shape (batch_size, obs_dim)
        deterministic : bool, optional
            確定的方策を用いるかどうか, by default False
            - True: 決定論的な行動（モード、平均値など）
            - False: 確率的な行動（サンプリング）

        Returns
        -------
        action : np.ndarray
            行動（観測値と同じ次元構造）
            - observation が1次元なら shape (action_dim,)
            - observation が2次元なら shape (batch_size, action_dim)
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, path: str, **kwargs) -> Tuple["Controller", Any]:
        """ファイルから制御器をロードする

        Parameters
        ----------
        path : str
            制御器の保存先ファイルパス
        **kwargs : dict
            アルゴリズム固有の追加パラメータ
            例: device (torch.device) for PPO

        Returns
        -------
        controller : Controller
            ロードされた制御器インスタンス
        normalization_stats : Any
            正規化統計量（obs_rmsなど）
            正規化が不要な場合はNoneを返す
        """
        pass
