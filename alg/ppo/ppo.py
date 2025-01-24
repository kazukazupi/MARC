from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from alg.ppo.model import Agent
from alg.ppo.storage import RolloutStorage


class PPO:
    def __init__(
        self,
        agent: Agent,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: float,
        eps: float,
        max_grad_norm: float,
        use_clipped_value_loss: bool = True,
    ):
        """
        PPOによりエージェントの更新を行うクラス

        Parameters
        ----------
        agent : Agent, 実体は方策と価値関数
        clip_param : float, PPOのクリッピングパラメータ
        ppo_epoch : int, updateメソッドにおけるエポック数
        num_mini_batch : int, updateメソッドにおけるミニバッチ数
        value_loss_coef : float, 価値関数の損失の係数
        entropy_coef : float, エントロピーの係数
        lr : float, 学習率
        eps : float, Adamのイプシロン
        max_grad_norm : float, 勾配の最大ノルム
        use_clipped_value_loss : bool, 価値関数更新にクリッピングした損失を使用するかどうか
        """
        self.agent = agent
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(agent.parameters(), lr=lr, eps=eps)
        self.max_grad_norm = max_grad_norm

    def update(self, storage: RolloutStorage) -> Tuple[float, float, float]:
        """
        方策と価値関数の更新を行う

        Parameters
        ----------
            storage : RolloutStorage, バッファ

        Returns
        -------
            value_loss_epoch : float, 価値関数の損失
            action_loss_epoch : float, 方策の損失
            dist_entropy_epoch : float, エントロピー
        """

        # アドバンテージの計算
        advantages = storage.returns[:-1] - storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        for _ in range(self.ppo_epoch):
            data_generator = storage.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                obs_batch = sample.obs
                actions_batch = sample.actions
                value_preds_batch = sample.value_preds
                return_batch = sample.returns
                old_action_log_prob_batch = sample.old_action_log_probs
                adv_targets = sample.adv_targets

                values, action_log_probs, dist_entropy = self.agent.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_prob_batch)
                surr1 = ratio * adv_targets
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targets

                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
