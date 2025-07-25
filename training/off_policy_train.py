import threading
import torch
from torch import optim
from train import get_logits_batch


def train():
    from collector import Collector
    from models.pncn import PNCN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=128, expand_ratio=1,
        num_transformer_layers=1,
        num_attention_heads=4,
        transformer_intermediate_size=128,
    ).to(device)
    old_policy_net = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=128, expand_ratio=1,
        num_transformer_layers=1,
        num_attention_heads=4,
        transformer_intermediate_size=128,
    ).to(device)
    old_policy_net.device = device
    old_policy_net.eval()

    instance_settings = [
        {
            "num_types": 5,
            "num_stages": 5,
            "num_machines_per_stage": 5,
            "lam": 0.005,
            "num_orders": 5,
            "process_time_lb": 20,
            "process_time_rb": 500,
            "due_lb": 200,
            "due_rb": 2500,
        },
    ]
    group_size = 10
    greedy_guiding = False
    replay_buffer_size = 5000

    collector = Collector(
        old_policy_net,
        instance_settings,
        group_size,
        greedy_guiding,
        replay_buffer_size
    )

    # warmup
    collector.collect()

    threading.Thread(target=collector.run, args=(1,)).start()

    # ========
    # train
    # ========
    batch_size = 64

    model.device = device
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    accumulate_steps = 4
    num_steps = 0
    while True:
        for _ in range(accumulate_steps):
            batch = collector.replay_buffer.sample_batch(batch_size)
            pn_state = [b['state'] for b in batch]
            due_time_of_order_ = [b['due_time_of_order_'] for b in batch]
            enable_transition_idxs = [b['enable_transition_idxs'] for b in batch]
            action = [b['action'] for b in batch]
            action_prob = [b['action_prob'] for b in batch]
            advantage = [b['advantage'] for b in batch]
            advantage = torch.tensor(advantage, dtype=torch.float32, device=device)
            action_prob = torch.tensor(action_prob, dtype=torch.float32, device=device)
            action_labels = torch.tensor(action, dtype=torch.long, device=device)

            logits = get_logits_batch(pn_state, due_time_of_order_, model)
            mask = torch.full_like(logits, fill_value=-torch.inf)
            for i, enable_transition_idx in enumerate(enable_transition_idxs):
                mask[i, enable_transition_idx] = 0.

            log_prob = torch.softmax(logits + mask, dim=-2)[range(batch_size), action_labels].log().squeeze()
            old_log_prob = action_prob.log()
            ratio = torch.exp(log_prob - old_log_prob)
            policy_loss_1 = advantage * ratio
            policy_loss_2 = advantage * torch.clamp(ratio, 0.8, 1.2)
            loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            print(loss)
            # loss = -log_prob * advantage
            # print(loss)

            (loss / accumulate_steps).backward()

        optimizer.step()
        optimizer.zero_grad()

        if num_steps % 1 == 0:
            with collector.global_state_dict_lock:
                collector.global_state_dict = model.state_dict()

        num_steps += 1


if __name__ == '__main__':
    train()

