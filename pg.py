from torch.distributions import Categorical
import random
import threading
from collections import deque
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from torch import nn, optim
from data.env_seed_generator import generate_process_time_info, generate_orders
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List
from copy import deepcopy
import torch
import numpy as np


def pnstate_to_vectors(pn_state, due_time_of_order_):
    """
    将PNState对象转换为矩阵/向量表示的Petri Net
    - C_pre: 前置关联矩阵[places x transitions]
    - C_post: 后置关联矩阵[places x transitions]
    - m: marking向量[places]
    - x: 已加工时间向量[places]
    - delay: 加工时长向量[places]
    - rest: 剩余时间向量[places]
    """
    # 建立名称到索引的映射
    place_name_to_idx = {name: idx for idx, name in enumerate(pn_state.place_names)}
    transition_name_to_idx = {name: idx for idx, name in enumerate(pn_state.transition_names)}

    num_places = len(pn_state.place_names)
    num_transitions = len(pn_state.transition_names)

    # 初始化输出数据结构
    C_pre = np.zeros((num_places, num_transitions), dtype=int)
    C_post = np.zeros((num_places, num_transitions), dtype=int)
    m = np.zeros(num_places, dtype=int)
    x = np.zeros(num_places)
    delay = np.zeros(num_places)
    rest = np.full(num_places, np.inf)  # 默认剩余时间无限

    # 构建关联矩阵
    for trans_name, pre_places in pn_state.pre_place_names_of_transition_named.items():
        trans_idx = transition_name_to_idx[trans_name]
        for place_name in pre_places:
            if place_name in place_name_to_idx:
                C_pre[place_name_to_idx[place_name], trans_idx] = 1

    for trans_name, post_places in pn_state.post_place_names_of_transition_named.items():
        trans_idx = transition_name_to_idx[trans_name]
        for place_name in post_places:
            if place_name in place_name_to_idx:
                C_post[place_name_to_idx[place_name], trans_idx] = 1

    # 填充状态向量
    for place_name, token_count in pn_state.m_of_place_named.items():
        if place_name in place_name_to_idx:
            idx = place_name_to_idx[place_name]
            m[idx] = token_count

    for place_name, token_count in pn_state.x_of_place_named.items():
        # 处理加工时间
        if place_name in pn_state.x_of_place_named:
            idx = place_name_to_idx[place_name]
            x[idx] = pn_state.x_of_place_named[place_name]

    # 计算剩余时间（订单交期 - 当前时间）
    for place_name, order_id in pn_state.order_of_place_named.items():
        due_date = due_time_of_order_[order_id]
        idx = place_name_to_idx[place_name]
        rest[idx] = due_date - pn_state.cur_time

    for place_name, delay_time in pn_state.delay_of_place_named.items():
        idx = place_name_to_idx[place_name]
        delay[idx] = delay_time

    return {
        'C_pre': C_pre,
        'C_post': C_post,
        'm': m,
        'x': x,
        'delay': delay,
        'rest': rest,
        'place_names': pn_state.place_names,
        'transition_names': pn_state.transition_names
    }


@dataclass
class PNState:
    place_names: List[str] = field(default_factory=list)
    transition_names: List[str] = field(default_factory=list)
    post_place_names_of_transition_named: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    pre_place_names_of_transition_named: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    m_of_place_named: Dict[str, int] = field(default_factory=dict)
    x_of_place_named: Dict[str, int] = field(default_factory=dict)
    delay_of_place_named: Dict[str, int] = field(default_factory=dict)
    order_of_place_named: Dict[str, str] = field(default_factory=dict)
    cur_time: float = 0.0


def get_init_pn_state(
    process_time_info, num_types, num_stages, init_orders,
):
    place_names = []
    transition_names = []
    delay_of_place_named = {}
    pre_place_names_of_transition_named = defaultdict(list)
    post_place_names_of_transition_named = defaultdict(list)

    # 机器资源库所
    for stage_idx in range(num_stages):
        for machine_idx in process_time_info[stage_idx].keys():
            place_names.append(f"M{machine_idx}")

    # 订单池模板库所
    for type_idx in range(num_types):
        place_names.append(f"T{type_idx}_pool")

    # 等待池模板库所
    for stage_idx in range(1, num_stages):
        for type_idx in range(num_types):
            place_names.append(f"T{type_idx}_S{stage_idx - 1}-S{stage_idx}_waiting")

    # 加工流程建模
    for stage_idx in range(num_stages):
        for machine_idx in process_time_info[stage_idx]:
            for type_idx in range(num_types):
                place_name = f"S{stage_idx}_T{type_idx}_M{machine_idx}"
                place_names.append(place_name)
                delay_of_place_named[place_name] = process_time_info[stage_idx][machine_idx][type_idx]
                begin_transition_name = f"begin_{place_name}"
                end_transition_name = f"end_{place_name}"
                transition_names.append(begin_transition_name)
                transition_names.append(end_transition_name)
                pre_place_names_of_transition_named[begin_transition_name].append(f"M{machine_idx}")
                post_place_names_of_transition_named[begin_transition_name].append(place_name)
                pre_place_names_of_transition_named[end_transition_name].append(place_name)
                post_place_names_of_transition_named[end_transition_name].append(f"M{machine_idx}")

                if stage_idx + 1 < num_stages:
                    waiting_place_name = f"T{type_idx}_S{stage_idx}-S{stage_idx + 1}_waiting"
                    post_place_names_of_transition_named[end_transition_name].append(waiting_place_name)

                if stage_idx > 0:
                    waiting_place_name = f"T{type_idx}_S{stage_idx - 1}-S{stage_idx}_waiting"
                    pre_place_names_of_transition_named[begin_transition_name].append(waiting_place_name)

    for type_idx in range(num_types):
        transition_name_prefix = f'begin_S0_T{type_idx}'
        for transition_name in transition_names:
            if transition_name.startswith(transition_name_prefix):
                pre_place_names_of_transition_named[transition_name].append(f"T{type_idx}_pool")

    first_order = init_orders[-1]
    cur_time = first_order['arrival_time']
    assert all(order['arrival_time'] == cur_time for order in init_orders)

    # 闲置订单，从等待池模板拷贝存放
    for order in init_orders:
        waiting_place_name = f"T{order['product_type']}_pool"
        new_waiting_place_name = waiting_place_name + f"_O{order['order_id']}"  # np
        place_names.append(new_waiting_place_name)
        for transition_name in transition_names:
            if waiting_place_name in pre_place_names_of_transition_named[transition_name]:
                new_transition_name = transition_name + f"_O{order['order_id']}"
                transition_names.append(new_transition_name)
                post_place_names_of_transition_named[new_transition_name] = deepcopy(
                    post_place_names_of_transition_named[transition_name])
                pre_place_names_of_transition_named[new_transition_name] = deepcopy(
                    pre_place_names_of_transition_named[transition_name])
                pre_place_names_of_transition_named[new_transition_name].remove(waiting_place_name)
                pre_place_names_of_transition_named[new_transition_name].append(new_waiting_place_name)

    m_of_place_named = defaultdict(int)
    x_of_place_named = defaultdict(int)
    order_of_place_named = defaultdict(str)

    # 先预设机器资源都可用
    for stage_idx in range(num_stages):
        for machine_idx in process_time_info[stage_idx].keys():
            m_of_place_named[f"M{machine_idx}"] = 1

    for order in init_orders:
        # 闲置中的订单，放在订单专属的等待池
        waiting_place_name = f"T{order['product_type']}_pool"
        place_name = waiting_place_name + f"_O{order['order_id']}"
        if m_of_place_named[place_name] == 0:
            m_of_place_named[place_name] = 1
            order_of_place_named[place_name] = order["order_id"]
        # 不闲置的订单，放在加工流程中，机器资源被占用

    pn_state = PNState(
        place_names=place_names,
        transition_names=transition_names,
        post_place_names_of_transition_named=post_place_names_of_transition_named,
        pre_place_names_of_transition_named=pre_place_names_of_transition_named,
        m_of_place_named=m_of_place_named,
        x_of_place_named=x_of_place_named,
        delay_of_place_named=delay_of_place_named,
        order_of_place_named=order_of_place_named,
        cur_time=cur_time,
    )

    return pn_state


class PNEnv:
    def __init__(self):
        self.cur_pn_state = None
        self.orders = None
        self.cur_enable_transitions = None
        self.due_time_of_order_ = None

    def reset(
            self,
            process_time_info,
            num_types,
            num_stages,
            orders,
    ):
        self.due_time_of_order_ = {
            order['order_id']: order['due_date']
            for order in orders
        }
        orders = orders[::-1]
        init_orders = []
        cur_time = orders[-1]['arrival_time']
        while orders and orders[-1]['arrival_time'] == cur_time:
            init_orders.append(orders.pop())
        self.cur_pn_state = get_init_pn_state(
            process_time_info, num_types, num_stages, init_orders,
        )
        self.orders = orders
        self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)
        return self.cur_pn_state

    @staticmethod
    def get_next_firing_end_transition(
            firing_transition_name,
            pn_state,
    ):
        place_names = pn_state.place_names
        transition_names = pn_state.transition_names
        post_place_names_of_transition_named = pn_state.post_place_names_of_transition_named
        pre_place_names_of_transition_named = pn_state.pre_place_names_of_transition_named
        m_of_place_named = pn_state.m_of_place_named
        x_of_place_named = pn_state.x_of_place_named
        delay_of_place_named = pn_state.delay_of_place_named
        order_of_place_named = pn_state.order_of_place_named
        cur_time = pn_state.cur_time

        # ========================
        # marking 处理
        # ========================
        src_place_names = pre_place_names_of_transition_named[firing_transition_name]
        assert len(src_place_names) == 1
        src_place_name = src_place_names[0]
        tar_place_names = post_place_names_of_transition_named[firing_transition_name]
        new_m_of_place_named = m_of_place_named.copy()
        new_order_of_place_named = order_of_place_named.copy()

        order = order_of_place_named[src_place_name]
        new_order_of_place_named.pop(src_place_name)
        new_m_of_place_named[src_place_name] -= 1
        if new_m_of_place_named[src_place_name] == 0:
            new_m_of_place_named.pop(src_place_name)

        tar_pool_place_name = None
        for tar_place_name in tar_place_names:
            if tar_place_name not in new_m_of_place_named:
                new_m_of_place_named[tar_place_name] = 1
            else:
                new_m_of_place_named[tar_place_name] += 1
            if 'waiting' in tar_place_name or 'pool' in tar_place_name:
                tar_pool_place_name = tar_place_name
                new_order_of_place_named[tar_place_name + f'_O{order}'] = order

        # ========================
        # 时间处理
        # ========================
        if src_place_name not in x_of_place_named:
            x_of_place_named[src_place_name] = 0
        timestep = max(0, delay_of_place_named[src_place_name] - x_of_place_named[src_place_name])
        new_x_of_place_named = {}
        for place_name in new_m_of_place_named:
            if '_T' in place_name:
                if place_name not in x_of_place_named:
                    x_of_place_named[place_name] = 0
                new_x_of_place_named[place_name] = min(x_of_place_named[place_name] + timestep,
                                                       delay_of_place_named[place_name])

        cur_time += timestep

        # ========================
        # 增加等待池
        # ========================
        new_transition_names = transition_names.copy()
        new_place_names = place_names.copy()
        new_post_place_names_of_transition_named = post_place_names_of_transition_named.copy()
        new_pre_place_names_of_transition_named = pre_place_names_of_transition_named.copy()

        new_delay_of_place_named = delay_of_place_named.copy()

        if tar_pool_place_name is None:
            new_pn_state = PNState(
                place_names=new_place_names,
                transition_names=new_transition_names,
                post_place_names_of_transition_named=new_post_place_names_of_transition_named,
                pre_place_names_of_transition_named=new_pre_place_names_of_transition_named,
                m_of_place_named=new_m_of_place_named,
                x_of_place_named=new_x_of_place_named,
                delay_of_place_named=new_delay_of_place_named,
                order_of_place_named=new_order_of_place_named,
                cur_time=cur_time,
            )
            return new_pn_state

        t_str = tar_pool_place_name.split('_')[0]
        s_str = tar_pool_place_name.split('_')[1].split('-')[-1]

        new_tar_pool_place_name = tar_pool_place_name + f'_O{order}'
        new_place_names.append(new_tar_pool_place_name)

        transition_names_to_copy = [t for t in transition_names if
                                    t.startswith(f'begin_{s_str}_{t_str}') and ('_O' not in t)]
        for transition_name_to_copy in transition_names_to_copy:
            new_transition_name_to_copy = transition_name_to_copy + f'_O{order}'
            new_transition_names.append(new_transition_name_to_copy)
            new_pre_place_names_of_transition_named[new_transition_name_to_copy] = pre_place_names_of_transition_named[
                transition_name_to_copy].copy()
            for i in range(len(new_pre_place_names_of_transition_named[new_transition_name_to_copy])):
                if "waiting" in new_pre_place_names_of_transition_named[new_transition_name_to_copy][i] or "pool" in \
                        new_pre_place_names_of_transition_named[new_transition_name_to_copy][i]:
                    new_pre_place_names_of_transition_named[new_transition_name_to_copy][i] = new_tar_pool_place_name
            new_post_place_names_of_transition_named[new_transition_name_to_copy] = \
            post_place_names_of_transition_named[transition_name_to_copy].copy()

        new_m_of_place_named[new_tar_pool_place_name] = 1
        new_m_of_place_named.pop(tar_pool_place_name)

        new_pn_state = PNState(
            place_names=new_place_names,
            transition_names=new_transition_names,
            post_place_names_of_transition_named=new_post_place_names_of_transition_named,
            pre_place_names_of_transition_named=new_pre_place_names_of_transition_named,
            m_of_place_named=new_m_of_place_named,
            x_of_place_named=new_x_of_place_named,
            delay_of_place_named=new_delay_of_place_named,
            order_of_place_named=new_order_of_place_named,
            cur_time=cur_time,
        )

        return new_pn_state

    @staticmethod
    def get_next_firing_begin_transition(
            firing_transition_name,
            pn_state,
    ):
        place_names = pn_state.place_names
        transition_names = pn_state.transition_names
        post_place_names_of_transition_named = pn_state.post_place_names_of_transition_named
        pre_place_names_of_transition_named = pn_state.pre_place_names_of_transition_named
        m_of_place_named = pn_state.m_of_place_named
        x_of_place_named = pn_state.x_of_place_named
        delay_of_place_named = pn_state.delay_of_place_named
        order_of_place_named = pn_state.order_of_place_named
        cur_time = pn_state.cur_time

        # ========================
        # marking 处理
        # ========================
        src_place_names = pre_place_names_of_transition_named[firing_transition_name]
        tar_place_names = post_place_names_of_transition_named[firing_transition_name]
        assert len(tar_place_names) == 1
        tar_place_name = tar_place_names[0]
        src_pool_place_name = None
        for src_place_name in src_place_names:
            if 'waiting' in src_place_name or 'pool' in src_place_name:
                src_pool_place_name = src_place_name
                break
        new_m_of_place_named = m_of_place_named.copy()
        new_order_of_place_named = order_of_place_named.copy()

        order = order_of_place_named[src_pool_place_name]
        new_order_of_place_named[tar_place_name] = order
        new_order_of_place_named.pop(src_pool_place_name)

        for src_place_name in src_place_names:
            new_m_of_place_named[src_place_name] -= 1
            if new_m_of_place_named[src_place_name] == 0:
                new_m_of_place_named.pop(src_place_name)

        new_m_of_place_named[tar_place_name] = 1

        # ========================
        # 移除等待池
        # ========================
        new_transition_names = transition_names.copy()
        new_place_names = place_names.copy()
        new_post_place_names_of_transition_named = post_place_names_of_transition_named.copy()
        new_pre_place_names_of_transition_named = pre_place_names_of_transition_named.copy()
        new_x_of_place_named = x_of_place_named.copy()
        # new_x_of_place_named[tar_place_name] = 0
        new_delay_of_place_named = delay_of_place_named.copy()

        transition_names_to_remove = []
        for transition_name in transition_names:
            if src_pool_place_name in pre_place_names_of_transition_named[transition_name]:
                transition_names_to_remove.append(transition_name)

        for transition_name_to_remove in transition_names_to_remove:
            new_transition_names.remove(transition_name_to_remove)
            new_post_place_names_of_transition_named.pop(transition_name_to_remove)
            new_pre_place_names_of_transition_named.pop(transition_name_to_remove)

        place_name_to_remove = src_pool_place_name
        new_place_names.remove(place_name_to_remove)

        new_pn_state = PNState(
            place_names=new_place_names,
            transition_names=new_transition_names,
            post_place_names_of_transition_named=new_post_place_names_of_transition_named,
            pre_place_names_of_transition_named=new_pre_place_names_of_transition_named,
            m_of_place_named=new_m_of_place_named,
            x_of_place_named=new_x_of_place_named,
            delay_of_place_named=new_delay_of_place_named,
            order_of_place_named=new_order_of_place_named,
            cur_time=cur_time,
        )

        return new_pn_state

    @staticmethod
    def get_enable_transitions(
            pn_state
    ):
        m_of_place_named = pn_state.m_of_place_named
        pre_place_names_of_transition_named = pn_state.pre_place_names_of_transition_named
        enable_transition_names = []
        for transition_name in pre_place_names_of_transition_named:
            pre_place_names = pre_place_names_of_transition_named[transition_name]
            enable = True
            for pre_place_name in pre_place_names:
                if pre_place_name not in m_of_place_named or m_of_place_named[pre_place_name] < 1:
                    enable = False
                    break
            if enable:
                enable_transition_names.append(transition_name)

        return enable_transition_names

    def step(self, transition_name):
        assert transition_name in self.cur_enable_transitions, f"transition {transition_name} is not enable"

        reward = 0
        done = False

        if transition_name.startswith('begin'):
            self.cur_pn_state = self.get_next_firing_begin_transition(transition_name, self.cur_pn_state)
            self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)
            done = False
            return self.cur_pn_state, reward, done, {}

        # ==================
        # 处理 end transition
        # ==================
        info = {}
        next_pn_state = self.get_next_firing_end_transition(transition_name, self.cur_pn_state)
        if self.orders and self.orders[-1]['arrival_time'] < next_pn_state.cur_time:
            # =============
            # 变迁激发失效
            # =============
            new_orders = []
            new_time = self.orders[-1]['arrival_time']
            while self.orders and self.orders[-1]['arrival_time'] == new_time:
                new_orders.append(self.orders.pop())

            self.cur_pn_state = self.get_pn_state_added_new_orders(
                pn_state=self.cur_pn_state,
                new_orders=new_orders,
                timestep=new_time - self.cur_pn_state.cur_time,
            )

            info = {"ignore": True}
        else:
            if (transition_name.startswith('end') and
                    len(self.cur_pn_state.post_place_names_of_transition_named[transition_name]) == 1):
                place_name = transition_name.split('end_')[-1]
                reward += (self.cur_pn_state.cur_time <= self.due_time_of_order_[self.cur_pn_state.order_of_place_named[place_name]])
            self.cur_pn_state = next_pn_state

        # 会造成时间推移，所以顺便把已经完成的 process 弹出
        while any([
            self.cur_pn_state.x_of_place_named[place_name] == self.cur_pn_state.delay_of_place_named[place_name]
            for place_name in self.cur_pn_state.x_of_place_named
        ]):
            for place_name in self.cur_pn_state.x_of_place_named:
                if self.cur_pn_state.x_of_place_named[place_name] == self.cur_pn_state.delay_of_place_named[place_name]:
                    transition_name = f"end_{place_name}"
                    if transition_name.startswith('end') and \
                            len(self.cur_pn_state.post_place_names_of_transition_named[transition_name]) == 1:
                        reward += (self.cur_pn_state.cur_time <= self.due_time_of_order_[self.cur_pn_state.order_of_place_named[place_name]])
                    self.cur_pn_state = self.get_next_firing_end_transition(transition_name, self.cur_pn_state)
                    break

        # 过期结构拆除
        transitions_to_remove = []
        places_to_remove = []
        for place_name, order_idx in self.cur_pn_state.order_of_place_named.items():
            due_time = self.due_time_of_order_[order_idx]
            if due_time <= self.cur_pn_state.cur_time and ('waiting' in place_name or 'pool' in place_name):
                places_to_remove.append(place_name)
                overdue_transitions = [overdue_t for overdue_t in self.cur_pn_state.transition_names if overdue_t.endswith(f'_O{order_idx}')]
                transitions_to_remove.extend(overdue_transitions)

        for transition_to_remove in transitions_to_remove:
            self.cur_pn_state.post_place_names_of_transition_named.pop(transition_to_remove)
            self.cur_pn_state.pre_place_names_of_transition_named.pop(transition_to_remove)
            self.cur_pn_state.transition_names.remove(transition_to_remove)
        for place_to_remove in places_to_remove:
            self.cur_pn_state.order_of_place_named.pop(place_to_remove)
            self.cur_pn_state.place_names.remove(place_to_remove)
            self.cur_pn_state.m_of_place_named.pop(place_to_remove)
            if place_to_remove in self.cur_pn_state.x_of_place_named:
                self.cur_pn_state.x_of_place_named.pop(place_to_remove)
            if place_to_remove in self.cur_pn_state.delay_of_place_named:
                self.cur_pn_state.delay_of_place_named.pop(place_to_remove)

        self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)

        def is_all_overdue():
            for cur_enable_transition in self.cur_enable_transitions:
                if cur_enable_transition.startswith('end'):
                    return False

                if self.cur_pn_state.cur_time + self.cur_pn_state.delay_of_place_named[self.cur_pn_state.post_place_names_of_transition_named[cur_enable_transition][0]] <= self.due_time_of_order_[cur_enable_transition.split('_O')[-1]]:
                    return False

            return True

        if not self.cur_enable_transitions or is_all_overdue():
            if self.orders:
                new_orders = []
                new_time = self.orders[-1]['arrival_time']
                while self.orders and self.orders[-1]['arrival_time'] == new_time:
                    new_orders.append(self.orders.pop())

                self.cur_pn_state = self.get_pn_state_added_new_orders(
                    pn_state=self.cur_pn_state,
                    new_orders=new_orders,
                    timestep=new_time - self.cur_pn_state.cur_time,
                )
                self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)
            else:
                done = True

        return self.cur_pn_state, reward, done, info

    @staticmethod
    def get_pn_state_added_new_orders(  # 保证不修改原来的 pn_state
            pn_state,
            new_orders,
            timestep,
    ):
        pn_state = deepcopy(pn_state)

        for place_name in pn_state.m_of_place_named:
            if '_T' not in place_name:
                continue

            # 到这里是 process place
            if place_name not in pn_state.x_of_place_named:
                pn_state.x_of_place_named[place_name] = 0

            pn_state.x_of_place_named[place_name] = min(
                pn_state.x_of_place_named[place_name] + timestep,
                pn_state.delay_of_place_named[place_name]
            )

        # 闲置订单，从等待池模板拷贝存放
        for order in new_orders:
            waiting_place_name = f"T{order['product_type']}_pool"
            new_waiting_place_name = waiting_place_name + f"_O{order['order_id']}"  # np
            pn_state.place_names.append(new_waiting_place_name)
            for transition_name in pn_state.transition_names:
                if waiting_place_name not in pn_state.pre_place_names_of_transition_named[transition_name]:
                    continue

                # 到这里的是跟新增库所相关的变迁
                new_transition_name = transition_name + f"_O{order['order_id']}"
                pn_state.transition_names.append(new_transition_name)
                pn_state.post_place_names_of_transition_named[new_transition_name] = deepcopy(
                    pn_state.post_place_names_of_transition_named[transition_name])
                pn_state.pre_place_names_of_transition_named[new_transition_name] = deepcopy(
                    pn_state.pre_place_names_of_transition_named[transition_name])
                pn_state.pre_place_names_of_transition_named[new_transition_name].remove(waiting_place_name)
                pn_state.pre_place_names_of_transition_named[new_transition_name].append(new_waiting_place_name)

        for order in new_orders:
            # 闲置中的订单，放在订单专属的等待池
            waiting_place_name = f"T{order['product_type']}_pool"
            place_name = waiting_place_name + f"_O{order['order_id']}"
            assert pn_state.m_of_place_named[place_name] == 0
            pn_state.m_of_place_named[place_name] = 1
            pn_state.order_of_place_named[place_name] = order["order_id"]

        pn_state.cur_time += timestep

        return pn_state

    def get_cur_enable_transitions(self):
        return self.get_enable_transitions(self.cur_pn_state)


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


class FinalP2TLayer(nn.Module):
    def __init__(
        self,
        in_place_channel,
        num_classes,
    ):
        super(FinalP2TLayer, self).__init__()
        self.num_classes = num_classes
        self.fc_p2t = nn.Linear(
            in_features=in_place_channel,
            out_features=num_classes * 2 * int(in_place_channel),
            bias=False
        )

    def forward(
        self,
        C_t_stack,
        place_features
    ):
        tmp = self.fc_p2t(place_features).view(
            place_features.shape[0],
            place_features.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        return torch.matmul(C_t_stack, tmp).sum(1)


class PNCLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_classes,
        expand_ratio,
        act=F.leaky_relu
    ):
        super(PNCLayer, self).__init__()
        self.num_classes = num_classes
        self.fc_p2t = nn.Linear(
            in_features=in_channel,
            out_features=num_classes * 2 * int(in_channel * expand_ratio),
            bias=False
        )
        self.fc_t2p = nn.Linear(
            in_features=int(in_channel * expand_ratio),
            out_features=num_classes * 2 * in_channel,
            bias=False
        )
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel, bias=True)
        self.act = act

    def forward(
        self,
        C_t_stack,
        C_stack,
        place_features
    ):
        tmp = self.fc_p2t(place_features).view(
            place_features.shape[0],
            place_features.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        tmp = self.act(torch.matmul(C_t_stack, tmp).sum(1))

        tmp = self.fc_t2p(tmp).view(
            tmp.shape[0],
            tmp.shape[1],
            self.num_classes * 2, -1
        ).permute(0, 2, 1, 3)
        place_features = self.act(torch.matmul(C_stack, tmp).sum(1) + place_features)
        return self.act(self.fc(place_features))


class PNCN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channel,
        num_pnc_layers,
        hidden_channel,
        expand_ratio,
        act=F.leaky_relu,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=768,
    ):
        super(PNCN, self).__init__()
        self.act = act
        self.num_classes = num_classes
        self.num_pnc_layers = num_pnc_layers

        self.fc = torch.nn.Linear(in_channel, hidden_channel)

        self.pnc_layers = nn.ModuleList([PNCLayer(num_classes=self.num_classes, in_channel=hidden_channel,
                                                  out_channel=hidden_channel, expand_ratio=expand_ratio,
                                                  act=act)] +
                                        [PNCLayer(num_classes=self.num_classes, in_channel=hidden_channel,
                                                  out_channel=hidden_channel, expand_ratio=expand_ratio,
                                                  act=act)
                                         for _ in range(num_pnc_layers - 1)])

        self.final_p2t = FinalP2TLayer(num_classes=self.num_classes,
                                       in_place_channel=hidden_channel)

        # 配置模型参数
        config = BertConfig(
            hidden_size=hidden_channel,  # 输入输出的隐藏层维度
            num_hidden_layers=num_transformer_layers,  # 只使用一个Transformer层
            num_attention_heads=num_attention_heads,  # 注意力头数（768/64=12）
            intermediate_size=transformer_intermediate_size,  # FFN中间层维度（默认值）
            position_embedding_type="none",  # 关键：禁用位置编码
        )
        self.final_transformer = BertEncoder(config)

        self.final_fc = nn.Linear(
            in_features=hidden_channel,
            out_features=1,
            bias=True
        )

    def forward(
        self,
        x,
        C_t_stack,
        C_stack,
        attention_mask,  # [B, |T| + |P|]
    ):
        x = self.act(self.fc(x))
        for pnc_layer in self.pnc_layers:
            x = F.instance_norm(x)  # TODO
            x = pnc_layer(C_t_stack=C_t_stack, C_stack=C_stack, place_features=x)
        x_t = self.final_p2t(C_t_stack=C_t_stack, place_features=F.instance_norm(x))  # TODO
        res = self.final_transformer(
            torch.cat([x_t, x], dim=1),
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
        )['last_hidden_state']
        logits = self.final_fc(res[:, :x_t.shape[1]])
        return logits


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def add_transition(self, state, due_time_of_order_, enable_transition_idxs, action, action_prob, advantage):
        transition = {
            'state': state,
            'due_time_of_order_': due_time_of_order_,
            'enable_transition_idxs': enable_transition_idxs,
            'action': action,
            'action_prob': action_prob,
            'advantage': advantage,
        }
        with self.lock:
            self.buffer.append(transition)

    def sample_batch(self, batch_size):
        """采样"""
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
        return batch


class Collector:
    def __init__(self, old_policy_net, instance_settings, group_size, greedy_guiding, replay_buffer_size):
        self.gamma = 0.99
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
                            reward,
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
            torch.save(self.old_policy_net.state_dict(), 'old_policy_net_0724.pth')
        print(accumulated_rewards)
        print("num_types", self.instance_settings[0]['num_types'])
        print("num_stages", self.instance_settings[0]['num_stages'])
        print("num_machines_per_stage", self.instance_settings[0]['num_machines_per_stage'])
        advantages = (accumulated_rewards - accumulated_rewards.mean()) / (accumulated_rewards.std() + 1e-6)

        for env_idx in range(len(self.pn_envs)):
            R = 0
            for transition in trajectories[env_idx][::-1]:
                reward = transition[5]
                R = reward + self.gamma * R
                self.replay_buffer.add_transition(
                    state=transition[0],
                    due_time_of_order_=transition[1],
                    enable_transition_idxs=transition[2],
                    action=transition[3],
                    action_prob=transition[4],
                    # advantage=R - len(transition[1]),
                    advantage=advantages[env_idx],
                )
        print("current buffer size", len(self.replay_buffer.buffer))


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=256, expand_ratio=0.25,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=256,
    ).to(device)
    model.device = device
    model.train()
    old_policy_net = PNCN(
        num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=256, expand_ratio=0.25,
        num_transformer_layers=3,
        num_attention_heads=4,
        transformer_intermediate_size=256,
    ).to(device)
    old_policy_net.device = device
    old_policy_net.eval()
    
    model.load_state_dict(torch.load('old_policy_net_0723.pth'))
    old_policy_net.load_state_dict(torch.load('old_policy_net_0723.pth'))

    instance_settings = [
        {
            "num_types": 5,
            "num_stages": 5,
            "num_machines_per_stage": 5,
            "lam": 0.005,
            "num_orders": 1,
            "process_time_lb": 20,
            "process_time_rb": 500,
            "due_lb": 200,
            "due_rb": 2500,
        },
    ]
    group_size = 3
    greedy_guiding = False
    replay_buffer_size = 500000

    collector = Collector(
        old_policy_net,
        instance_settings,
        group_size,
        greedy_guiding,
        replay_buffer_size
    )

    # warmup
    # collector.collect()

    # threading.Thread(target=collector.run, args=(1,)).start()

    # ========
    # train
    # ========
    batch_size = 64

    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    accumulate_steps = 20

    while True:
        collector.old_policy_net.load_state_dict(model.state_dict())
        collector.replay_buffer.buffer.clear()

        for episode in range(10):
            collector.collect()

        data = list(collector.replay_buffer.buffer)
        random.shuffle(data)
        for i in range(len(collector.replay_buffer.buffer) // batch_size):
            batch = data[i * batch_size: (i + 1) * batch_size]
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
            # old_log_prob = action_prob.log()
            # ratio = torch.exp(log_prob - old_log_prob)
            # policy_loss_1 = advantage * ratio
            # policy_loss_2 = advantage * torch.clamp(ratio, 0.9, 1.1)
            # loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            loss = - (log_prob * advantage).mean()
            print(loss)

            loss.backward()

        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    train()

