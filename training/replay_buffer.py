import random
from collections import deque
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_transition(self, state, due_time_of_order_, enable_transition_idxs, action, action_prob, advantage):
        transition = {
            'state': state,
            'due_time_of_order_': due_time_of_order_,
            'enable_transition_idxs': enable_transition_idxs,
            'action': action,
            'action_prob': action_prob,
            'advantage': advantage,
        }
        self.buffer.append(transition)

    def sample_batch(self, batch_size):
        """采样"""
        batch = random.sample(self.buffer, batch_size)
        return batch
