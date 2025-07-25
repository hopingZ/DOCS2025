from torch import nn, optim

from data.env_seed_generator import generate_process_time_info, generate_orders
from state_to_numpy import pnstate_to_vectors
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List
from copy import deepcopy
import numpy as np
import torch


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
        self.process_time_info = None
        self.num_stages = None

    def reset(
            self,
            process_time_info,
            num_types,
            num_stages,
            orders,
    ):
        self.num_stages = num_stages
        self.process_time_info = process_time_info
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

        while not self.cur_enable_transitions or self.is_all_overdue():
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
            post_place_names_of_transition_named[
                transition_name_to_copy].copy()

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

    def is_all_overdue(self):
        for cur_enable_transition in self.cur_enable_transitions:
            if cur_enable_transition.startswith('end'):
                return False

            begin_transition_name = cur_enable_transition
            s, t, m, o = begin_transition_name.split('_')[1:]
            next_stage_time = self.cur_pn_state.delay_of_place_named['_'.join([s, t, m])]
            estimated_left_time = next_stage_time
            t_idx = int(t[1:])
            for s_idx in range(int(s[1:]) + 1, self.num_stages):
                estimated_left_time += min([_[t_idx] for _ in self.process_time_info[s_idx].values()])

            if estimated_left_time <= self.due_time_of_order_[o[1:]] - self.cur_pn_state.cur_time:
                return False

            '''if self.cur_pn_state.cur_time + self.cur_pn_state.delay_of_place_named[self.cur_pn_state.post_place_names_of_transition_named[cur_enable_transition][0]] <= self.due_time_of_order_[cur_enable_transition.split('_O')[-1]]:
                return False'''

        return True

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

        # ==================
        # 过期结构拆除
        # ==================
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

        while not self.cur_enable_transitions or self.is_all_overdue():
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
