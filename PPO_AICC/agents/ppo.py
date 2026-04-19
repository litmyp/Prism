import os
import torch
from torch import optim
from torch import nn
import numpy as np
from typing import List, Tuple
from types import SimpleNamespace
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from config.config import Config
from models.actor_critic import ActorCritic
from .utils import random_sample, AsyncronousRollouts, flatten
from .base import BaseAgent

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None


class PPO(BaseAgent):
    def __init__(self, config: Config, env: VecEnv):
        BaseAgent.__init__(self, config, env)
        self.model = ActorCritic(self.env.observation_space, self.config).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate, eps=1e-5)
        self.rollout = AsyncronousRollouts(self.config)
        self.use_lut = bool(self.config.agent.get('lut', False))
        self.lut_path = self.config.agent.get('lut_path', None)
        self.lut_table = None
        self.lut_obs_edges = None
        self.lut_action_weights = None
        self.lut_probs = None
        self.compare_nn = bool(self.config.agent.get('compare_nn', False))
        self.nn_loaded_for_compare = False
        self.distill_requested = bool(self.config.agent.get('distill', False))
        self.use_distill = self.distill_requested
        self.distill_output_type = self.config.agent.get('distill_output_type', 'discrete')
        self.distill_model_path = self.config.agent.get('distill_model', None)
        self.distill_model = None
        if self.use_lut:
            # LUT 推理优先，启用时关闭蒸馏/模型加载
            self.use_distill = False

        if self.config.agent.evaluate:
            if self.use_lut:
                try:
                    self._load_lut()
                except Exception as e:
                    print(f"加载 LUT 失败，将回退使用模型/蒸馏: {e}")
                    self.use_lut = False
                    self.use_distill = self.distill_requested
                else:
                    if self.compare_nn:
                        try:
                            self.load_model()
                            self.nn_loaded_for_compare = True
                        except Exception as e:
                            print(f"加载用于对比的 PPO 权重失败，将只使用 LUT: {e}")

            if self.use_lut:
                pass
            elif self.use_distill:
                try:
                    self._load_distill_model()
                except Exception as e:
                    print(f"加载蒸馏模型失败，将回退使用 PPO 权重: {e}")
                    self.use_distill = False
                    self.load_model()
            else:
                self.load_model()

    def test(self) -> None:
        timesteps = 0

        state, info = self.env.reset()

        with torch.no_grad():
            while True:
                if state.numel() ==0:
                    print("state为空张量, 结束")
                    break
                print(f"states:{state}")# 打印action. adpg_reward值
                action_for_env = None
                if self.use_lut and self.lut_table is not None:
                    lut_action = self._lut_act(state)
                    action_for_env = self._parse_action(lut_action.cpu())
                    if self.compare_nn and self.nn_loaded_for_compare:
                        nn_action = self.model.act(state)
                        nn_action_cpu = nn_action.detach().cpu()
                        nn_parsed = self._parse_action(nn_action_cpu.clone())
                        print(f"LUT动作idx: {lut_action.detach().cpu().tolist()} 解析: {action_for_env} | NN动作idx: {nn_action_cpu.tolist()} 解析: {nn_parsed}")
                    action = lut_action
                elif self.use_distill and self.distill_model is not None:
                    action = self._distill_act(state)
                    action_for_env = self._parse_action(action.cpu())
                else:
                    action = self.model.act(state)
                    action_for_env = self._parse_action(action.cpu())
                state, reward, done, infos = self.env.step(action_for_env)

                self.log_data(timesteps, infos)

                timesteps += state.shape[0]

    def train(self) -> Tuple[float, float, float]:
        timesteps = 0

        state, info = self.env.reset()
        reward = done = torch.tensor([0. for _ in range(state.shape[0])])
        policy_loss = v_loss = entropy_loss = -1
        num_updates = 0

        while num_updates < self.config.training.max_num_updates:
            rollouts = []
            final_states = []
            while len(rollouts) < self.config.agent.ppo.rollouts_per_batch:
                # if not torch.isfinite(state).all():
                #     print("检测到有nan/inf,state信息如下:")
                #     print(state)
                #     break
                rollout = self.rollout.add(dict(state=state, reward=reward, mask=1. - done), info, True)
                if len(rollout) > 0:
                    for r in rollout:
                        rollouts.append(r[0])
                        final_states.append(r[1])

                #检查数据是不是需要归一化
                # print(f"Date check {state}")
                value, action, action_log_probs = self.model(state)

                self.rollout.add(dict(value=value, action=action, action_log_probs=action_log_probs), info, False)

                state, reward, done, infos = self.env.step(self._parse_action(action.cpu().detach()))

                self.log_data(timesteps, infos)

                timesteps += 1

            num_updates += 1
            states, actions, log_probs, returns, advantages = self._process_data(rollouts, final_states)

            assert torch.isfinite(states).all(), f"states 包含inf/NaN: {states}"
            assert torch.isfinite(actions).all(), f"actions 包含inf/NaN: {actions}"
            assert torch.isfinite(returns).all(), f"returns 包含inf/NaN: {returns}"
            assert torch.isfinite(advantages).all(), f"advantages 包含inf/NaN: {advantages}"
            assert torch.isfinite(log_probs).all(), f"log_probs 包含inf/NaN: {log_probs}"

            policy_loss, v_loss, entropy_loss = self._calculate_loss(states, actions, log_probs, returns, advantages)
            avg_return = returns.mean().item()

            print(f"Policy Update {num_updates}/{self.config.training.max_num_updates} (Steps: {timesteps})")
            print(f"  >> [Reward] Avg Reward: {avg_return:.5f}")   
            print(f"  >> [Loss]  Policy loss: {policy_loss:.5f} | Value loss: {v_loss:.5f} | Entropy loss: {entropy_loss:.5f}")
            print(20*'-')
            if self.config.logging.wandb is not None:
                self.config.logging.wandb.log(
                    {"Policy loss": policy_loss, "Value loss": v_loss, "Entropy loss": entropy_loss},
                    step=timesteps
                )
            self.save_model(checkpoint=timesteps)

        # 训练结束也保存一下
        self.save_model(checkpoint=timesteps)
        return policy_loss, v_loss, entropy_loss

    def _parse_action(self, actions: torch.tensor) -> float:
        actions = actions.view(-1).numpy().tolist()
        for i, action in enumerate(actions):
            if self.config.agent.ppo.discrete_actions:
                action = self.config.agent.ppo.action_weights[action]
            else:
                action = np.tanh(action)
                if action < 0:
                    action = 1. / (1. - action * self.config.agent.ppo.action_multiplier_dec)
                else:
                    action = 1. + action * self.config.agent.ppo.action_multiplier_inc
            actions[i] = action
        return actions

    def _process_data(
            self,
            rollouts: List[SimpleNamespace],
            final_states: List[torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:

        batch_size = len(final_states)

        final_states = torch.cat(final_states)
        with torch.no_grad():
            final_values = self.model.critic(final_states)

        states = []
        actions = []
        log_probs = []
        returns = []
        advantages = []

        for batch_index in range(batch_size):
            reward_to_go = final_values[batch_index]
            rollouts[batch_index].value.append(reward_to_go)

            states.append(rollouts[batch_index].state)
            actions.append(rollouts[batch_index].action)
            log_probs.append(rollouts[batch_index].action_log_probs)

            advantages.append([])
            returns.append([])

            adv = 0
            for i in reversed(range(len(actions[-1]))):
                reward_to_go = rollouts[batch_index].reward[i] + self.config.agent.discount * rollouts[batch_index].mask[i] * reward_to_go
                if not self.config.agent.ppo.use_gae:
                    adv = reward_to_go - rollouts[batch_index].value[i].detach()
                else:
                    td_error = rollouts[batch_index].reward[i] + self.config.agent.discount * rollouts[batch_index].mask[i] * rollouts[batch_index].value[i + 1] - rollouts[batch_index].value[i]
                    adv = td_error + adv * self.config.agent.ppo.gae_tau * self.config.agent.discount * rollouts[batch_index].mask[i]
                advantages[-1].insert(0, adv)
                returns[-1].insert(0, reward_to_go)

        states = torch.cat(flatten(states)).detach()
        actions = torch.cat(flatten(actions)).detach()
        log_probs = torch.cat(flatten(log_probs)).detach()
        returns = torch.cat(flatten(returns)).detach()
        advantages = torch.cat(flatten(advantages)).detach()

        return states, actions, log_probs, returns, advantages

    def _calculate_loss(
            self,
            states: torch.tensor,
            actions: torch.tensor,
            old_log_probs: torch.tensor,
            returns: torch.tensor,
            advantages: torch.tensor
    ) -> Tuple[float, float, float]:

        policy_loss = 0
        entropy_loss = 0
        value_loss = 0

        # for name, p in self.model.named_parameters():
            # if torch.isnan(p).any():
            #     print("calculate_loss前检查到parameter NaN:", name)
            #     break

        for _ in range(self.config.agent.ppo.params.ppo_optimization_epochs):
            sample = random_sample(np.arange(self.config.agent.ppo.rollout_length), self.config.agent.ppo.params.ppo_batch_size)
            for batch_indices in sample:
                batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.config.device)
                sampled_states, sampled_actions, sampled_old_log_probs, sampled_returns, sampled_advantages = (arr[batch_indices] for arr in [states, actions, old_log_probs, returns, advantages])

                assert torch.isfinite(sampled_states).all(), f"sampled_states 包含inf/NaN: {sampled_states}"#
                assert torch.isfinite(sampled_actions).all(), f"sampled_actions 包含inf/NaN: {sampled_actions}"#
                assert torch.isfinite(sampled_old_log_probs).all(), f"sampled_old_log_probs 包含inf/NaN: {sampled_old_log_probs}"#
                assert torch.isfinite(sampled_returns).all(), f"sampled_returns 包含inf/NaN: {sampled_returns}"#
                assert torch.isfinite(sampled_advantages).all(), f"sampled_advantages 包含inf/NaN: {sampled_advantages}"#
                
                v, log_probs, entropy = self.model.evaluate(sampled_states, sampled_actions)

                assert torch.isfinite(v).all(), f"v 包含inf/NaN: {v}"#
                assert torch.isfinite(log_probs).all(), f"log_probs 包含inf/NaN: {log_probs}"#
                assert torch.isfinite(entropy).all(), f"entropy 包含inf/NaN: {entropy}"#

                ratio = (log_probs - sampled_old_log_probs).exp()

                assert torch.isfinite(ratio).all(),f"ratio 包含inf/NaN: {ratio}"#

                pg_obj1 = ratio * sampled_advantages
                pg_obj2 = ratio.clamp(1.0 - self.config.agent.ppo.params.ppo_ratio_clip, 1.0 + self.config.agent.ppo.params.ppo_ratio_clip) * sampled_advantages
                pg_loss = torch.min(pg_obj1, pg_obj2).mean()

                v_loss = 0.5 * torch.square(sampled_returns - v).mean()

                loss = - pg_loss - self.config.agent.ppo.entropy_coeff * entropy + self.config.agent.ppo.baseline_coeff * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.optimizer.step()

                policy_loss += pg_loss.item() / (self.config.agent.ppo.params.ppo_optimization_epochs * self.config.agent.ppo.params.ppo_batch_size)
                entropy_loss += entropy.item() / (self.config.agent.ppo.params.ppo_optimization_epochs * self.config.agent.ppo.params.ppo_batch_size)
                value_loss += v_loss.item() / (self.config.agent.ppo.params.ppo_optimization_epochs * self.config.agent.ppo.params.ppo_batch_size)

        return policy_loss, value_loss, entropy_loss
    
    def _load_lut(self):
        if not self.lut_path or self.lut_path in (-1, "None", None):
            raise FileNotFoundError("lut_path 未配置或无效")
        if not os.path.exists(self.lut_path):
            raise FileNotFoundError(f"LUT 文件不存在: {self.lut_path}")
        data = np.load(self.lut_path)
        self.lut_table = data['lut']
        self.lut_obs_edges = data['obs_edges']
        self.lut_action_weights = data['action_weights']
        self.lut_probs = data.get('lut_probs', None)

        cfg_weights = np.array(self.config.agent.ppo.action_weights, dtype=float)
        if len(cfg_weights) != len(self.lut_action_weights) or not np.allclose(cfg_weights, self.lut_action_weights):
            print(f"警告: LUT 内 action_weights={self.lut_action_weights} 与配置 {cfg_weights} 不一致，索引按 LUT 内部值映射，输出动作按配置权重执行")

        if self.config.env.history_length != 2:
            print(f"警告: 当前 history_length={self.config.env.history_length}，LUT 期望值为 2")
        expected_features = ['action', 'adpg_reward']
        if list(self.config.agent.agent_features) != expected_features:
            print(f"警告: 当前 agent_features={self.config.agent.agent_features}，LUT 期望 {expected_features}")

    def _lut_act(self, state: torch.tensor) -> torch.tensor:
        """
        使用预先生成的 LUT 进行动作推理。
        目前假设 history_length=2 且 agent_features=['action', 'adpg_reward']。
        """
        if self.lut_table is None or self.lut_obs_edges is None or self.lut_action_weights is None:
            raise RuntimeError("LUT 未正确加载")

        state_np = state.detach().cpu().numpy()
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        if state_np.shape[1] < 4:
            raise ValueError(f"LUT 输入维度不足，收到 {state_np.shape}")

        actions_idx = []

        for row in state_np:
            a0, r0, a1, r1 = row[:4]

            a0_idx = int(np.argmin(np.abs(self.lut_action_weights - float(a0))))
            a1_idx = int(np.argmin(np.abs(self.lut_action_weights - float(a1))))

            r0_bin = int(np.searchsorted(self.lut_obs_edges, float(r0), side='right') - 1)
            r1_bin = int(np.searchsorted(self.lut_obs_edges, float(r1), side='right') - 1)
            r0_bin = int(np.clip(r0_bin, 0, len(self.lut_obs_edges) - 2))
            r1_bin = int(np.clip(r1_bin, 0, len(self.lut_obs_edges) - 2))

            if self.lut_probs is not None:
                probs = self.lut_probs[a0_idx, r0_bin, a1_idx, r1_bin]
                probs = np.asarray(probs, dtype=float)
                s = probs.sum()
                if s > 0:
                    probs = probs / s
                act_idx = int(np.random.choice(len(probs), p=probs))
            else:
                act_idx = int(self.lut_table[a0_idx, r0_bin, a1_idx, r1_bin])
            actions_idx.append(act_idx)

        return torch.tensor(actions_idx, dtype=torch.long)

    def _load_distill_model(self):
        if CatBoostClassifier is None:
            raise ImportError("未安装 catboost，无法加载蒸馏模型")
        if not self.distill_model_path or self.distill_model_path in (-1, "None", None):
            raise FileNotFoundError("distill_model 未配置有效路径")
        if not os.path.exists(self.distill_model_path):
            raise FileNotFoundError(f"蒸馏模型不存在: {self.distill_model_path}")
        # 优先按分类模型加载，不行则退回回归
        # try:
        #     model = CatBoostClassifier()
        #     model.load_model(self.distill_model_path)
        #     self.distill_model = model
        # except Exception:
        #     model = CatBoostRegressor()
        #     model.load_model(self.distill_model_path)
        #     self.distill_model = model

        # 新实现：根据 distill_output_type/动作空间偏好优先选择模型类型，并在失败时自动回退
        prefer_regression = (self.distill_output_type == 'continuous') or (not self.config.agent.ppo.discrete_actions)
        load_order = [CatBoostRegressor, CatBoostClassifier] if prefer_regression else [CatBoostClassifier, CatBoostRegressor]

        last_err = None
        for cls in load_order:
            try:
                model = cls()
                model.load_model(self.distill_model_path)

                # 若以分类方式加载但模型元数据显示无类别/回归损失，则视为不匹配，继续尝试回退
                params = model.get_all_params()
                loss_fn = params.get('loss_function', '') if params else ''
                classes_count = params.get('classes_count', None) if params else None
                if isinstance(model, CatBoostClassifier) and (loss_fn.upper() == 'RMSE' or classes_count == 0):
                    raise ValueError("模型为回归损失，分类器载入可能异常，尝试回退")

                self.distill_model = model
                return
            except Exception as e:
                last_err = e
                continue

        # 若所有尝试均失败，抛出最后的错误信息
        raise last_err

    def _distill_act(self, state: torch.tensor) -> torch.tensor:
        """
        使用蒸馏后的 CatBoost 模型做动作推理。
        默认输出离散动作 id（distill_output_type=discrete）；若为 continuous 则认为输出原始连续动作。
        """
        state_np = state.detach().cpu().numpy()
        if self.distill_output_type == 'discrete' or self.config.agent.ppo.discrete_actions:
            weights = np.array(self.config.agent.ppo.action_weights, dtype=float)

            # 硬标签蒸馏：直接使用模型输出的离散动作索引
            preds = self.distill_model.predict(state_np)
            preds = np.array(preds).reshape(-1)
            idxs = [int(np.clip(int(round(p)), 0, len(weights) - 1)) for p in preds]
            return torch.tensor(idxs, dtype=torch.long)

        preds = self.distill_model.predict(state_np)
        preds = np.array(preds).reshape(-1)
        return torch.tensor(preds, dtype=torch.float32)

    def act(self, state: torch.tensor) -> torch.tensor:
        return torch.tanh(self.model.act(state))
