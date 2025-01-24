from typing import Generator, NamedTuple, Optional

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    value_preds: torch.Tensor
    returns: torch.Tensor
    old_action_log_probs: torch.Tensor
    adv_targets: torch.Tensor


class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_dim: int,
        action_dim: int,
    ):
        """
        収集した経験を保存するためのストレージ

        Parameters
        ----------
            num_steps: int, ストレージに保存するステップ数
            num_processes: int, 環境の並列プロセス数
            obs_dim: int, 観測空間の次元数
            action_dim: int, 行動空間の次元数
        """

        self.obs = torch.zeros(num_steps + 1, num_processes, obs_dim)  # 観測
        self.rewards = torch.zeros(num_steps, num_processes, 1)  # 報酬
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)  # 価値関数の予測値
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)  # 期待収益
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)  # 行動の対数確率
        self.actions = torch.zeros(num_steps, num_processes, action_dim)  # 行動
        self.masks = torch.ones(num_steps + 1, num_processes, 1)  # エピソードが終了したかどうかのマスク
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)  # タイムリミットによる終了かどうかのマスク

        self.num_steps = num_steps  # ストレージに保存するステップ数
        self.step = 0  # 現在のステップ数

    def to(self, device: torch.device):
        """
        デバイスを指定する。
        """
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: torch.Tensor,
    ):
        """
        1タイムステップ分の経験をストレージに保存する。
        """

        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        モデルのパラメータ更新後に行う処理。
        """
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True,
    ):
        """
        各時刻の期待収益を計算する。

        Parameters
        ----------
            next_value: torch.Tensor, 次の状態価値
            use_gae: bool, GAEを使うかどうか
            gamma: float, 割引率
            gae_lambda: float, GAEのλ
            use_proper_time_limits: bool, タイムリミットによる終了を区別するかどうか
        """

        if use_gae:
            """
            GAEに基づき、時刻tにおけるアドバンテージ関数A_tを、以下のように計算する。
            A_t = delta_t + (gamma * gae_lambda)^1 * delta_{t+1} + ... + (gamma * gae_lambda)^{T-t-1} * delta_{T-1}
                = delta_t + gamma * gae_lambda * A_{t+1}
            これを元に、時刻tにおける期待収益G_tを計算する。
            G_t = A_t + V_t
            """
            self.value_preds[-1] = next_value
            gae = torch.tensor(0.0)
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                if use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            """
            時刻tにおける期待収益G_tを、以下のように計算する。
            G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * V_T
                = r_t + gamma * G_{t+1}
            """
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                return_ = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
                if use_proper_time_limits:
                    return_ = (
                        return_ * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                    )
                self.returns[step] = return_

    def feed_forward_generator(
        self,
        advantages: torch.Tensor,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Generator[Batch, None, None]:
        """
        ppoの更新に用いるデータをミニバッチに分割して返すジェネレータの生成。

        Parameters
        ----------
            advantages: torch.Tensor, アドバンテージ関数
            num_mini_batch: Optional[int], ミニバッチの数
            mini_batch_size: Optional[int], ミニバッチのサイズ
        """

        num_steps, num_processes = self.rewards.size()[:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert num_mini_batch is not None, "Either mini_batch_size or num_mini_batch must be given"
            assert batch_size >= num_mini_batch, f"batch_size({batch_size}) >= num_mini_batch({num_mini_batch})"
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:

            batch = Batch(
                obs=self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                actions=self.actions.view(-1, self.actions.size(-1))[indices],
                value_preds=self.value_preds[:-1].view(-1, 1)[indices],
                returns=self.returns[:-1].view(-1, 1)[indices],
                old_action_log_probs=self.action_log_probs.view(-1, 1)[indices],
                adv_targets=advantages.view(-1, 1)[indices],
            )

            yield batch
