from collections import defaultdict
from copy import deepcopy
from .pn_state import PNState
from .pn_successor_fn import get_next_firing_begin_transition as _get_next_firing_begin_transition
from .pn_successor_fn import get_next_firing_end_transition as _get_next_firing_end_transition
from .pn_successor_fn import get_enable_transitions as _get_enable_transitions


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
        self.num_received_orders = None
        self.num_done_orders = None

    def set(self, cur_pn_state, process_time_info, num_stages, due_time_of_order_):
        self.orders = []
        self.cur_pn_state = cur_pn_state
        self.process_time_info = process_time_info
        self.num_stages = num_stages
        self.due_time_of_order_ = due_time_of_order_
        self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)
        self.num_received_orders = 0
        self.num_done_orders = 0

    def reset(
            self,
            process_time_info,
            num_types,
            num_stages,
            orders,
    ):
        self.num_done_orders = 0
        self.num_stages = num_stages
        self.process_time_info = process_time_info
        self.due_time_of_order_ = {
            order['order_id']: order['due_date']
            for order in orders
        }
        orders = orders[::-1]
        init_orders = []
        cur_time = orders[-1]['arrival_time']
        self.num_received_orders = 0
        while orders and orders[-1]['arrival_time'] == cur_time:
            init_orders.append(orders.pop())
            self.num_received_orders += 1
        self.cur_pn_state = get_init_pn_state(
            process_time_info, num_types, num_stages, init_orders,
        )
        self.orders = orders
        self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)

        while not self.cur_enable_transitions:
            if self.orders:
                new_orders = []
                new_time = self.orders[-1]['arrival_time']
                while self.orders and self.orders[-1]['arrival_time'] == new_time:
                    new_orders.append(self.orders.pop())
                    self.num_received_orders += 1

                self.cur_pn_state = self.get_pn_state_added_new_orders(
                    pn_state=self.cur_pn_state,
                    new_orders=new_orders,
                    timestep=new_time - self.cur_pn_state.cur_time,
                )
                self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)
            else:
                raise Exception("No enable transition and no new orders")

        return self.cur_pn_state

    @staticmethod
    def get_next_firing_end_transition(
            firing_transition_name,
            pn_state,
    ):
        return _get_next_firing_end_transition(
            firing_transition_name,
            pn_state,
        )

    @staticmethod
    def get_next_firing_begin_transition(
            firing_transition_name,
            pn_state,
    ):
        return _get_next_firing_begin_transition(
            firing_transition_name,
            pn_state,
        )

    def get_enable_transitions(  # 不只是 PN 定义的不可激发，还包括了 overdue
            self,
            pn_state
    ):
        return _get_enable_transitions(
            pn_state,
            self.num_stages,
            self.process_time_info,
            self.due_time_of_order_,
        )

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
                self.num_received_orders += 1

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
                self.num_done_orders += (self.cur_pn_state.cur_time <= self.due_time_of_order_[
                    self.cur_pn_state.order_of_place_named[place_name]])
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
                        self.num_done_orders += (self.cur_pn_state.cur_time <= self.due_time_of_order_[
                            self.cur_pn_state.order_of_place_named[place_name]])
                    self.cur_pn_state = self.get_next_firing_end_transition(transition_name, self.cur_pn_state)
                    break

        # ==================
        # 过期结构拆除
        # ==================
        transitions_to_remove = []
        places_to_remove = []
        for place_name, order_idx in self.cur_pn_state.order_of_place_named.items():
            due_time = self.due_time_of_order_[order_idx]
            # 把不可能达成的一起剔除
            if 'waiting' in place_name or 'pool' in place_name:
                remove = False

                if 'pool' in place_name:
                    t, _, o = place_name.split('_')
                    t_idx = int(t[1:])
                    s_idx = 0
                else:
                    t, ss, _, o = place_name.split('_')
                    t_idx = int(t[1:])
                    s_idx = int(ss.split('-')[-1][1:])

                estimated_left_time = 0
                for s_idx in range(s_idx, self.num_stages):
                    estimated_left_time += min([_[t_idx] for _ in self.process_time_info[s_idx].values()])
                if estimated_left_time > self.due_time_of_order_[o[1:]] - self.cur_pn_state.cur_time:
                    remove = True

                if not remove and due_time <= self.cur_pn_state.cur_time:
                    remove = True

                if remove:
                    places_to_remove.append(place_name)
                    overdue_transitions = [overdue_t for overdue_t in self.cur_pn_state.transition_names if
                                           overdue_t.endswith(f'_O{order_idx}')]
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

        reward -= len(places_to_remove)

        # ==================
        # 检查环境是否仍可推进
        # ==================
        self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)
        while not self.cur_enable_transitions:
            if self.orders:
                new_orders = []
                new_time = self.orders[-1]['arrival_time']
                while self.orders and self.orders[-1]['arrival_time'] == new_time:
                    new_orders.append(self.orders.pop())
                    self.num_received_orders += 1

                self.cur_pn_state = self.get_pn_state_added_new_orders(
                    pn_state=self.cur_pn_state,
                    new_orders=new_orders,
                    timestep=new_time - self.cur_pn_state.cur_time,
                )
                self.cur_enable_transitions = self.get_enable_transitions(self.cur_pn_state)

            else:
                done = True
                reward -= len(self.cur_pn_state.order_of_place_named)
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
