from copy import deepcopy

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from data.env_seed_generator import generate_process_time_info, generate_orders
from training.pn_env import PNEnv
from training.state_to_numpy import pnstate_to_vectors


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
    logits = model(nn_input, C_t_stack, C_stack)
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
    logits = model(
        nn_input, C_t_stack, C_stack,
        attention_mask=attention_mask,  # [B, |T| + |P|]
    )
    return logits


def train():
    import random
    from models.pncn import PNCN
    from torch.distributions import Categorical

    model = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=3, hidden_channel=128, expand_ratio=0.25,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=128,
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    num_envs = 10
    pn_envs = [PNEnv() for _ in range(num_envs)]

    num_types, num_stages = 5, 5
    num_machines_per_stage = 5
    '''lam = 0.0083
    num_orders = 30
    process_time_lb, process_time_rb = 121, 140
    due_lb, due_rb = 800, 1200'''

    lam = 0.03
    num_orders = 20
    process_time_lb, process_time_rb = 100, 160
    due_lb, due_rb = 800, 1200

    while True:
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

        for pn_env in pn_envs:
            pn_env.reset(process_time_info, num_types, num_stages, orders)

        accumulated_rewards = [0 for _ in range(num_envs)]
        trajectories = [[] for _ in range(num_envs)]
        done_flags = [False for _ in range(num_envs)]

        print('========= collecting =========')
        model.eval()
        while True:
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

            probs = nn.functional.softmax(logits + mask, dim=-2)
            probs[probs.isnan()] = 1 / probs.shape[1]
            dist = Categorical(probs.squeeze())
            actions = dist.sample()
            for env_idx in range(len(pn_envs)):
                if done_flags[env_idx]:
                    continue

                action = actions[env_idx]
                firing_transition_name = pn_envs[env_idx].cur_pn_state.transition_names[action.item()]

                last_pn_state = deepcopy(pn_envs[env_idx].cur_pn_state)
                _, reward, done, info = pn_envs[env_idx].step(firing_transition_name)
                if not done_flags[env_idx] and "ignore" not in info and firing_transition_name.startswith('begin'):
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
        advantages = (accumulated_rewards - accumulated_rewards.mean()) / (accumulated_rewards.std() + 1e-6)

        update_steps = 2
        batch_size = 64

        model.train()
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
            loss = -torch.tensor(advantage_labels, dtype=torch.float) @ \
                torch.softmax(logits, dim=-2)[range(batch_size), action_labels]
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train()
