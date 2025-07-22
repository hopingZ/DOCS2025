from copy import deepcopy

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from data.env_seed_generator import generate_process_time_info, generate_orders, load_env_seed_from_txt, \
    get_num_types_and_stages
from training.pn_env import PNEnv
from training.state_to_numpy import pnstate_to_vectors


def benchmark(model, fp):
    process_time_info, orders = load_env_seed_from_txt(fp)
    num_types, num_stages = get_num_types_and_stages(process_time_info)
    pn_env = PNEnv()
    pn_env.reset(process_time_info, num_types, num_stages, orders)

    accumulated_reward = 0
    while True:
        '''with torch.no_grad():
            logits = get_logits(
                pn_env.cur_pn_state,
                pn_env.due_time_of_order_,
                model
            )
            mask = torch.full_like(logits, fill_value=-torch.inf)

            for transition_idx, transition_name in enumerate(pn_env.cur_pn_state.transition_names):
                if transition_name in pn_env.cur_enable_transitions:
                    mask[0, transition_idx, 0] = 100. if 'begin' in transition_name else 0.  # TODO

            action = (logits + mask).squeeze().argmax().item()'''

        enable_transitions = pn_env.cur_enable_transitions
        if any([enable_transition.startswith('begin') for enable_transition in enable_transitions]):
            begin_transition_names = [
                enable_transition for enable_transition in enable_transitions if
                enable_transition.startswith('begin')
            ]
            cost_times = [pn_env.cur_pn_state.delay_of_place_named['_'.join(begin_transition_name.split('_')[1:-1])]
                          for begin_transition_name in begin_transition_names]
            firing_transition_name = begin_transition_names[np.argmin(cost_times).item()]
        else:
            working_place_names = [enable_transition.replace('end_', '') for enable_transition in enable_transitions]
            left_times = [pn_env.cur_pn_state.delay_of_place_named[place_name] -
                          (pn_env.cur_pn_state.x_of_place_named[place_name] if place_name in pn_env.cur_pn_state.x_of_place_named else 0)
                          for place_name in working_place_names]
            firing_transition_name = enable_transitions[np.argmin(left_times).item()]

        # firing_transition_name = pn_env.cur_pn_state.transition_names[action]
        # print(firing_transition_name)
        _, reward, done, info = pn_env.step(firing_transition_name)
        accumulated_reward += reward

        if done:
            break

    return accumulated_reward


def get_logits(pn_state, due_time_of_order_, model):
    pn_state_npy = pnstate_to_vectors(pn_state, due_time_of_order_)
    C_pre = pn_state_npy["C_pre"]
    C_post = pn_state_npy["C_post"]
    delay = pn_state_npy["delay"]

    C_stack = np.zeros((4,) + C_pre.shape)
    # TP -> T
    C_stack[0][delay != 0] = C_pre[delay != 0]
    # TP <- T
    C_stack[1][delay != 0] = C_post[delay != 0]
    # T <- NTP
    C_stack[2][delay == 0] = C_pre[delay == 0]
    # T -> NTP
    C_stack[3][delay == 0] = C_post[delay == 0]
    C_t_stack = C_stack.transpose([0, 2, 1]).copy()
    C_t_stack = torch.tensor(C_t_stack, dtype=torch.float32)
    C_stack = torch.tensor(C_stack, dtype=torch.float32)

    m = pn_state_npy["m"]
    x = pn_state_npy["x"]
    delay = pn_state_npy["delay"]
    rest = pn_state_npy["rest"]
    rest[rest == np.inf] = delay.sum()
    nn_input = np.stack([m, x, delay, rest], axis=-1)
    nn_input = torch.tensor(nn_input, dtype=torch.float32)
    nn_input = nn_input.unsqueeze(0)

    nn_input = nn_input.to(model.device)
    C_t_stack = C_t_stack.to(model.device)
    C_stack = C_stack.to(model.device)
    attention_mask = torch.ones([1, C_pre.shape[0] + C_pre.shape[1]]).to(model.device)
    logits = model(nn_input, C_t_stack, C_stack, attention_mask)
    return logits


def get_logits_batch(pn_state, due_time_of_order_, model):
    pn_state_batch = [pnstate_to_vectors(pn_state[i], due_time_of_order_[i]) for i in range(len(pn_state))]
    C_pre = [b["C_pre"] for b in pn_state_batch]
    C_post = [b["C_post"] for b in pn_state_batch]
    delay = [b["delay"] for b in pn_state_batch]

    C_stacks = []
    C_t_stacks = []
    for i in range(len(pn_state)):
        C_stack = np.zeros((4,) + C_pre[i].shape)
        # TP -> T
        C_stack[0][delay[i] != 0] = C_pre[i][delay[i] != 0]
        # TP <- T
        C_stack[1][delay[i] != 0] = C_post[i][delay[i] != 0]
        # T <- NTP
        C_stack[2][delay[i] == 0] = C_pre[i][delay[i] == 0]
        # T -> NTP
        C_stack[3][delay[i] == 0] = C_post[i][delay[i] == 0]

        C_t_stack = C_stack.transpose([0, 2, 1]).copy()
        C_t_stack = torch.tensor(C_t_stack, dtype=torch.float32)
        C_stack = torch.tensor(C_stack, dtype=torch.float32)
        C_stacks.append(C_stack)
        C_t_stacks.append(C_t_stack)

    ms = [b["m"] for b in pn_state_batch]
    xs = [b["x"] for b in pn_state_batch]
    delays = [b["delay"] for b in pn_state_batch]
    rests = [b["rest"] for b in pn_state_batch]

    for i in range(len(pn_state_batch)):
        rests[i][rests[i] == np.inf] = delay[i].sum()

    p_lens = [ont_C_pre.shape[0] for ont_C_pre in C_pre]
    t_lens = [ont_C_pre.shape[1] for ont_C_pre in C_pre]
    p_max_len = max(p_lens)
    t_max_len = max(t_lens)

    m = np.zeros((len(pn_state), p_max_len), dtype=int)
    x = np.zeros((len(pn_state), p_max_len), dtype=float)
    delay = np.zeros((len(pn_state), p_max_len), dtype=float)
    rest = np.zeros((len(pn_state), p_max_len), dtype=float)
    C_t_stack = np.zeros((len(pn_state), 4, t_max_len, p_max_len))
    C_stack = np.zeros((len(pn_state), 4, p_max_len, t_max_len))
    p_attention_mask = np.zeros((len(pn_state), p_max_len), dtype=bool)
    t_attention_mask = np.zeros((len(pn_state), t_max_len), dtype=bool)

    for i in range(len(pn_state)):
        m[i, :p_lens[i]] = ms[i]
        x[i, :p_lens[i]] = xs[i]
        delay[i, :p_lens[i]] = delays[i]
        rest[i, :p_lens[i]] = rests[i]
        C_stack[i, :, :p_lens[i], :t_lens[i]] = C_stacks[i]
        C_t_stack[i, :, :t_lens[i], :p_lens[i]] = C_t_stacks[i]
        p_attention_mask[i, :p_lens[i]] = True
        t_attention_mask[i, :t_lens[i]] = True

    attention_mask = np.concatenate([t_attention_mask, p_attention_mask], axis=1)
    nn_input = np.stack([m, x, delay, rest], axis=-1)
    nn_input = torch.tensor(nn_input, dtype=torch.float32)
    C_t_stack = torch.tensor(C_t_stack, dtype=torch.float32)
    C_stack = torch.tensor(C_stack, dtype=torch.float32)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)

    nn_input = nn_input.to(model.device)
    C_t_stack = C_t_stack.to(model.device)
    C_stack = C_stack.to(model.device)
    attention_mask = attention_mask.to(model.device)
    logits = model(
        nn_input, C_t_stack, C_stack,
        attention_mask=attention_mask,  # [B, |T| + |P|]
    )
    return logits


def train():
    from pathlib import Path
    for fp in Path('../schedulePlat/data/instance/competition/').glob('*.txt'):
        accumulated_reward = benchmark(None, fp)
        print(fp, accumulated_reward)
    import random
    from models.pncn import PNCN
    from torch.distributions import Categorical

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_envs = 10

    model = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=512, expand_ratio=0.25,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=512,
    ).to(device)
    model.device = device

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    pn_envs = [PNEnv() for _ in range(num_envs)]

    settings = [
        {
            "num_types": 5,
            "num_stages": 5,
            "num_machines_per_stage": 5,
            "lam": 0.0083,
            "num_orders": 50,
            "process_time_lb": 60,
            "process_time_rb": 120,
            "due_lb": 700,
            "due_rb": 1200,
        },
        {
            "num_types": 5,
            "num_stages": 5,
            "num_machines_per_stage": 5,
            "lam": 0.016,
            "num_orders": 50,
            "process_time_lb": 60,
            "process_time_rb": 120,
            "due_lb": 700,
            "due_rb": 1200,
        },
        {
            "num_types": 5,
            "num_stages": 5,
            "num_machines_per_stage": 5,
            "lam": 0.03,
            "num_orders": 50,
            "process_time_lb": 60,
            "process_time_rb": 120,
            "due_lb": 700,
            "due_rb": 1200,
        },
    ]

    num_turns = 0
    temperature = 1.0
    while True:
        # 根据环境参数生成 seed
        setting = random.choice(settings)
        num_types = setting["num_types"]
        num_stages = setting["num_stages"]
        num_machines_per_stage = setting["num_machines_per_stage"]
        process_time_lb = setting["process_time_lb"]
        process_time_rb = setting["process_time_rb"]
        lam = setting["lam"]
        num_orders = setting["num_orders"]
        due_lb = setting["due_lb"]
        due_rb = setting["due_rb"]

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

        # 收集训练数据
        print('========= collecting =========')
        accumulated_rewards = [0 for _ in range(num_envs)]
        trajectories = [[] for _ in range(num_envs)]
        done_flags = [False for _ in range(num_envs)]

        for pn_env in pn_envs:
            pn_env.reset(process_time_info, num_types, num_stages, orders)

        model.eval()
        while True:
            # 使用当前模型采样动作
            with torch.no_grad():
                logits = get_logits_batch(
                    [pn_env.cur_pn_state for pn_env in pn_envs],
                    [pn_env.due_time_of_order_ for pn_env in pn_envs],
                    model
                )
                mask = torch.full_like(logits, fill_value=-torch.inf)

                for env_idx, pn_env in enumerate(pn_envs):
                    for transition_idx, transition_name in enumerate(pn_env.cur_pn_state.transition_names):
                        if transition_name in pn_env.cur_enable_transitions:
                            mask[env_idx, transition_idx, 0] = 0.

                probs = nn.functional.softmax(logits / temperature + mask, dim=-2)
                probs[probs.isnan()] = 1 / probs.shape[1]
                dist = Categorical(probs.squeeze())
                actions = dist.sample()

            # 更新环境并存储
            for env_idx in range(len(pn_envs)):
                if done_flags[env_idx]:
                    continue

                action = actions[env_idx]
                firing_transition_name = pn_envs[env_idx].cur_pn_state.transition_names[action.item()]

                last_pn_state = deepcopy(pn_envs[env_idx].cur_pn_state)
                _, reward, done, info = pn_envs[env_idx].step(firing_transition_name)

                if not done_flags[env_idx] and "ignore" not in info:
                    trajectories[env_idx].append((
                            last_pn_state,
                            action.item(),
                            reward,
                        )
                    )

                accumulated_rewards[env_idx] += reward

                if done:
                    done_flags[env_idx] = True

            if all(done_flags):
                break

        print(accumulated_rewards)
        accumulated_rewards = np.array(accumulated_rewards)
        if accumulated_rewards.std() < 1:
            temperature += 0.2
            continue
        else:
            temperature = 1.0

        advantages = (accumulated_rewards - accumulated_rewards.mean()) / (accumulated_rewards.std() + 1e-6)
        print(accumulated_rewards.std())

        update_steps = 20
        batch_size = 64

        model.train()
        optimizer.zero_grad()
        print('========= training =========')
        for _ in tqdm(range(update_steps)):
            input_states = []
            advantage_labels = []
            action_labels = []
            for item_idx in range(batch_size):
                env_idx = random.randint(0, num_envs - 1)
                pn_state, action, reward = random.choice(trajectories[env_idx])
                advantage_labels.append(advantages[env_idx])
                action_labels.append(action)
                input_states.append(pn_state)

            logits = get_logits_batch(input_states, [due_time_of_order_] * batch_size, model)
            loss = -torch.tensor(advantage_labels, dtype=torch.float).to(model.device) @ \
                torch.softmax(logits, dim=-2)[range(batch_size), action_labels]
            loss = loss / update_steps
            print(loss)

            loss.backward()

        optimizer.step()


        if num_turns % 5 == 0:
            model.eval()
            accumulated_rewards = benchmark(model, '../schedulePlat/data/instance/competition/num1000_lam0.03_change0__8.txt')
            # accumulated_rewards = benchmark(model, './schedulePlat/data/instance/competition/num1000_lam0.03_change0__8.txt')
            print(accumulated_rewards)
        num_turns += 1


if __name__ == '__main__':
    train()
