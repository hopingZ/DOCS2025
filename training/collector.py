import random
import threading
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from data.env_seed_generator import generate_process_time_info, generate_orders
from models.pncn import PNCN
from training.pn_env import PNEnv
from training.train import get_logits_batch
from replay_buffer import ReplayBuffer


class Collector:
    def __init__(self, old_policy_net, instance_settings, group_size, greedy_guiding, replay_buffer_size):
        self.old_policy_net = old_policy_net
        self.old_policy_net.eval()
        self.instance_settings = instance_settings
        self.greedy_guiding = greedy_guiding
        self.group_size = group_size + int(self.greedy_guiding)
        self.pn_envs = [PNEnv() for _ in range(self.group_size)]
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.global_state_dict = None
        self.global_state_dict_lock = threading.Lock()
        self.last_total_rewards = deque(maxlen=10)

    def run(self, episodes_between_updates):
        while True:
            for _ in range(episodes_between_updates):
                self.collect()
            with self.global_state_dict_lock:
                if self.global_state_dict is not None:
                    self.old_policy_net.load_state_dict(self.global_state_dict)

    def collect(self):
        setting = random.choice(self.instance_settings)
        num_types = setting['num_types']
        num_stages = setting['num_stages']
        num_machines_per_stage = setting['num_machines_per_stage']
        process_time_lb = setting['process_time_lb']
        process_time_rb = setting['process_time_rb']
        num_orders = setting['num_orders']
        lam = setting['lam']
        due_lb = setting['due_lb']
        due_rb = setting['due_rb']

        process_time_info = generate_process_time_info(
            num_types,
            num_stages,
            num_machines_per_stage,
            process_time_lb,
            process_time_rb
        )
        orders = generate_orders(num_orders, lam, due_lb, due_rb, num_types)

        due_time_of_order_ = {
            order['order_id']: order['due_date']
            for order in orders
        }

        accumulated_rewards = [0 for _ in range(self.group_size)]
        trajectories = [[] for _ in range(self.group_size)]
        done_flags = [False for _ in range(self.group_size)]

        for pn_env in self.pn_envs:
            pn_env.reset(process_time_info, num_types, num_stages, orders)

        while True:
            # 使用当前模型采样动作
            with torch.no_grad():
                if self.greedy_guiding:
                    pn_envs = self.pn_envs[:-1]
                else:
                    pn_envs = self.pn_envs

                logits = get_logits_batch(
                    [pn_env.cur_pn_state for pn_env in pn_envs],
                    [pn_env.due_time_of_order_ for pn_env in pn_envs],
                    self.old_policy_net
                )  # [10, 255, 1]
                mask = torch.full_like(logits, fill_value=-torch.inf)

                for env_idx, pn_env in enumerate(pn_envs):
                    for transition_idx, transition_name in enumerate(pn_env.cur_pn_state.transition_names):
                        if transition_name in pn_env.cur_enable_transitions:
                            mask[env_idx, transition_idx, 0] = 0.

                probs = nn.functional.softmax(logits + mask, dim=-2)
                probs[probs.isnan()] = 1 / probs.shape[1]
                dist = Categorical(probs.squeeze())
                actions = dist.sample()

            if self.greedy_guiding:
                pn_env = self.pn_envs[-1]
                enable_transitions = pn_env.cur_enable_transitions
                if any([enable_transition.startswith('begin') for enable_transition in enable_transitions]):
                    begin_transition_names = [
                        enable_transition for enable_transition in enable_transitions if
                        enable_transition.startswith('begin')
                    ]
                    cost_times = [
                        pn_env.cur_pn_state.delay_of_place_named['_'.join(begin_transition_name.split('_')[1:-1])]
                        for begin_transition_name in begin_transition_names]
                    firing_transition_name = begin_transition_names[np.argmin(cost_times).item()]

                else:
                    working_place_names = [enable_transition.replace('end_', '') for enable_transition in
                                           enable_transitions]
                    left_times = [pn_env.cur_pn_state.delay_of_place_named[place_name] -
                                  (pn_env.cur_pn_state.x_of_place_named[
                                       place_name] if place_name in pn_env.cur_pn_state.x_of_place_named else 0)
                                  for place_name in working_place_names]
                    firing_transition_name = enable_transitions[np.argmin(left_times).item()]

                action = pn_env.cur_pn_state.transition_names.index(firing_transition_name)
                prob = torch.zeros(probs.shape[0] + 1, max(action + 1, probs.shape[1]), 1)
                prob[-1, action] = 1.0
                prob[:-1, :probs.shape[1]] = probs.clone()
                actions = torch.cat([actions, torch.tensor([action]).to(actions)])
                # probs = torch.cat([probs, prob[None, :].to(actions)])
                probs = prob.to(actions.device)

            # 更新环境并存储
            for env_idx in range(len(self.pn_envs)):
                if done_flags[env_idx]:
                    continue

                action = actions[env_idx]
                firing_transition_name = self.pn_envs[env_idx].cur_pn_state.transition_names[action.item()]

                last_pn_state = deepcopy(self.pn_envs[env_idx].cur_pn_state)
                enable_transition_idxs = [
                    last_pn_state.transition_names.index(transition_name)
                    for transition_name in self.pn_envs[env_idx].cur_enable_transitions
                ]
                _, reward, done, info = self.pn_envs[env_idx].step(firing_transition_name)

                if not done_flags[env_idx] and "ignore" not in info:
                    trajectories[env_idx].append((
                            last_pn_state,
                            due_time_of_order_,
                            enable_transition_idxs,
                            action.item(),
                            probs[env_idx, action.item()].item(),
                        )
                    )

                accumulated_rewards[env_idx] += reward

                if done:
                    done_flags[env_idx] = True

            if all(done_flags):
                break

        accumulated_rewards = np.array(accumulated_rewards)
        self.last_total_rewards.append(accumulated_rewards.mean())
        level = (sum(self.last_total_rewards) / len(self.last_total_rewards)) / self.instance_settings[0]['num_orders']
        print(sum(self.last_total_rewards) / len(self.last_total_rewards), '/', self.instance_settings[0]['num_orders'])
        if level > 0.5:
            self.instance_settings[0]['num_orders'] += 1
            self.last_total_rewards.clear()
            torch.save(self.old_policy_net.state_dict(), 'old_policy_net.pth')
        print(accumulated_rewards)
        advantages = (accumulated_rewards - accumulated_rewards.mean()) / (accumulated_rewards.std() + 1e-6)

        for env_idx in range(len(self.pn_envs)):
            random.shuffle(trajectories[env_idx])
            for transition in trajectories[env_idx]:
                self.replay_buffer.add_transition(
                    state=transition[0],
                    due_time_of_order_=transition[1],
                    enable_transition_idxs=transition[2],
                    action=transition[3],
                    action_prob=transition[4],
                    advantage=advantages[env_idx]
                )
        print("current buffer size", len(self.replay_buffer.buffer))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    old_policy_net = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=512, expand_ratio=0.25,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=512,
    ).to(device)
    old_policy_net.device = device
    old_policy_net.eval()

    instance_settings = [
        {
            "num_types": 5,
            "num_stages": 5,
            "num_machines_per_stage": 5,
            "lam": 0.005,
            "num_orders": 20,
            "process_time_lb": 20,
            "process_time_rb": 500,
            "due_lb": 200,
            "due_rb": 2500,
        },
    ]
    group_size = 10
    greedy_guiding = True
    replay_buffer_size = 100000

    collector = Collector(
        old_policy_net,
        instance_settings,
        group_size,
        greedy_guiding,
        replay_buffer_size
    )

    collector.run(
        old_policy_net.state_dict(),
        None,
        10000
    )


if __name__ == "__main__":
    main()

