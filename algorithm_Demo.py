from collections import defaultdict
from copy import deepcopy
import pandas as pd
import numpy as np
import random


def helper(
        place_names,
        transition_names,
        delay_of_place_named,
        post_place_names_of_transition_named,
        pre_place_names_of_transition_named
):
    idx_of_place_named = {}
    idx_of_transition_named = {}

    for i, place_name in enumerate(place_names):
        idx_of_place_named[place_name] = i

    for i, transition_name in enumerate(transition_names):
        idx_of_transition_named[transition_name] = i

    delay = np.zeros(len(place_names), int)
    for place_name, delay_ in delay_of_place_named.items():
        delay[idx_of_place_named[place_name]] = delay_

    num_places = len(place_names)
    num_transitions = len(transition_names)
    C_post = np.zeros([num_places, num_transitions], int)
    for transition_name, post_place_names in post_place_names_of_transition_named.items():
        for post_place_name in post_place_names:
            C_post[idx_of_place_named[post_place_name], idx_of_transition_named[transition_name]] = 1

    C_pre = np.zeros([num_places, num_transitions], int)
    for transition_name, pre_place_names in pre_place_names_of_transition_named.items():
        for pre_place_name in pre_place_names:
            C_pre[idx_of_place_named[pre_place_name], idx_of_transition_named[transition_name]] = 1

    return idx_of_place_named, C_pre, C_post, delay


def get_pn_state(
        process_time_info, num_types, num_stages, orders, cur_time,
):
    place_names = []
    transition_names = []
    delay_of_place_named = {}
    pre_place_names_of_transition_named = defaultdict(list)
    post_place_names_of_transition_named = defaultdict(list)

    for stage_idx in range(num_stages):
        for machine_idx in process_time_info[stage_idx].keys():
            place_names.append(f"M{machine_idx}")

    for type_idx in range(num_types):
        place_names.append(f"T{type_idx}_pool")

    for stage_idx in range(1, num_stages):
        for type_idx in range(num_types):
            place_names.append(f"T{type_idx}_S{stage_idx - 1}-S{stage_idx}_waiting")

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

    for order in orders:
        if order['current_stage'] is None or cur_time >= order["end_time"]:
            waiting_place_name = f"T{order['product_type']}_pool" if order[
                                                                         'current_stage'] is None else f"T{order['product_type']}_S{order['current_stage']}-S{int(order['current_stage']) + 1}_waiting"
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

    idx_of_place_named, C_pre, C_post, delay = helper(
        place_names, transition_names, delay_of_place_named,
        post_place_names_of_transition_named, pre_place_names_of_transition_named
    )

    m_0 = np.zeros(len(idx_of_place_named), int)
    m_target = np.zeros(len(idx_of_place_named), int)
    x = np.zeros(len(idx_of_place_named), int)

    for stage_idx in range(num_stages):
        for machine_idx in process_time_info[stage_idx].keys():
            m_0[idx_of_place_named[f"M{machine_idx}"]] = 1
            m_target[idx_of_place_named[f"M{machine_idx}"]] = 1

    order_id_in_place_idx_ = {}
    for order in orders:
        if order['current_stage'] is None or cur_time >= order["end_time"]:
            waiting_place_name = f"T{order['product_type']}_pool" if order[
                                                                         'current_stage'] is None else f"T{order['product_type']}_S{order['current_stage']}-S{int(order['current_stage']) + 1}_waiting"
            place_name = waiting_place_name + f"_O{order['order_id']}"
            if m_0[idx_of_place_named[place_name]] == 0:
                m_0[idx_of_place_named[place_name]] = 1
                order_id_in_place_idx_[idx_of_place_named[place_name]] = order["order_id"]
        else:
            process_place_name = f"S{order['current_stage']}_T{order['product_type']}_M{order['assigned_machine']}"
            x[idx_of_place_named[process_place_name]] = cur_time - order['start_time']
            m_0[idx_of_place_named[process_place_name]] = 1
            m_0[idx_of_place_named[f"M{order['assigned_machine']}"]] = 0
            order_id_in_place_idx_[idx_of_place_named[process_place_name]] = order["order_id"]

    m_of_place_named = {place_name: int(m_0[idx_of_place_named[place_name]]) for place_name in place_names if
                        int(m_0[idx_of_place_named[place_name]])}

    x_of_place_named = {place_name: int(x[idx_of_place_named[place_name]]) for place_name in place_names if
                        int(x[idx_of_place_named[place_name]])}

    order_of_place_named = {place_names[place_id]: order_id_in_place_idx_[place_id] for place_id in
                            order_id_in_place_idx_}

    return (place_names, transition_names,
            post_place_names_of_transition_named, pre_place_names_of_transition_named,
            m_of_place_named,
            x_of_place_named,
            delay_of_place_named,
            order_of_place_named,
            cur_time)


def get_enable_transitions(
        m_of_place_named,
        pre_place_names_of_transition_named,
):
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


def get_next(
        firing_transition_name,
        place_names, transition_names,
        post_place_names_of_transition_named, pre_place_names_of_transition_named,
        m_of_place_named,
        x_of_place_named,
        delay_of_place_named,
        order_of_place_named,
        cur_time,
):
    if firing_transition_name.startswith("begin"):
        return get_next_firing_begin_transition(
            firing_transition_name,
            place_names, transition_names,
            post_place_names_of_transition_named, pre_place_names_of_transition_named,
            m_of_place_named,
            x_of_place_named,
            delay_of_place_named,
            order_of_place_named,
            cur_time,
        )
    else:
        (place_names, transition_names,
         post_place_names_of_transition_named, pre_place_names_of_transition_named,
         m_of_place_named,
         x_of_place_named,
         delay_of_place_named,
         order_of_place_named,
         cur_time,) = get_next_firing_end_transition(
            firing_transition_name,
            place_names, transition_names,
            post_place_names_of_transition_named, pre_place_names_of_transition_named,
            m_of_place_named,
            x_of_place_named,
            delay_of_place_named,
            order_of_place_named,
            cur_time,
        )
        while any(
                [x_of_place_named[place_name] == delay_of_place_named[place_name] for place_name in x_of_place_named]):
            for place_name in x_of_place_named:
                if x_of_place_named[place_name] == delay_of_place_named[place_name]:
                    (place_names, transition_names,
                     post_place_names_of_transition_named, pre_place_names_of_transition_named,
                     m_of_place_named,
                     x_of_place_named,
                     delay_of_place_named,
                     order_of_place_named,
                     cur_time,) = get_next_firing_end_transition(
                        f"end_{place_name}",
                        place_names, transition_names,
                        post_place_names_of_transition_named, pre_place_names_of_transition_named,
                        m_of_place_named,
                        x_of_place_named,
                        delay_of_place_named,
                        order_of_place_named,
                        cur_time,
                    )
                    break

        return (place_names, transition_names,
                post_place_names_of_transition_named, pre_place_names_of_transition_named,
                m_of_place_named,
                x_of_place_named,
                delay_of_place_named,
                order_of_place_named,
                cur_time,)


def get_next_firing_end_transition(
        firing_transition_name,
        place_names, transition_names,
        post_place_names_of_transition_named, pre_place_names_of_transition_named,
        m_of_place_named,
        x_of_place_named,
        delay_of_place_named,
        order_of_place_named,
        cur_time,
):
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
        return (new_place_names, new_transition_names,
                new_post_place_names_of_transition_named, new_pre_place_names_of_transition_named,
                new_m_of_place_named,
                new_x_of_place_named,
                new_delay_of_place_named,
                new_order_of_place_named,
                cur_time)

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
        new_post_place_names_of_transition_named[new_transition_name_to_copy] = post_place_names_of_transition_named[
            transition_name_to_copy].copy()

    new_m_of_place_named[new_tar_pool_place_name] = 1
    new_m_of_place_named.pop(tar_pool_place_name)

    return (new_place_names, new_transition_names,
            new_post_place_names_of_transition_named, new_pre_place_names_of_transition_named,
            new_m_of_place_named,
            new_x_of_place_named,
            new_delay_of_place_named,
            new_order_of_place_named,
            cur_time)


def get_next_firing_begin_transition(
        firing_transition_name,
        place_names, transition_names,
        post_place_names_of_transition_named, pre_place_names_of_transition_named,
        m_of_place_named,
        x_of_place_named,
        delay_of_place_named,
        order_of_place_named,
        cur_time,
):
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

    return (new_place_names, new_transition_names,
            new_post_place_names_of_transition_named, new_pre_place_names_of_transition_named,
            new_m_of_place_named,
            new_x_of_place_named,
            new_delay_of_place_named,
            new_order_of_place_named,
            cur_time)


# 参赛队伍算法（请封装成类）
class SchedulingAlgorithm:
    def __init__(self):
        self.type_of_order_ = {}
        self.due_time_of_order_ = {}

    @staticmethod
    def get_system_info(MBOM):
        '''MBOM_dict = {
            'product_type':    {0: '0', 1: '0', 2: '0', 3: '0', 4: '0', 5: '0', 6: '1', 7: '1', 8: '1', 9: '1', 10: '1', 11: '1'},
            'stage':           {0: '0', 1: '0', 2: '0', 3: '1', 4: '1', 5: '1', 6: '0', 7: '0', 8: '0', 9: '1', 10: '1', 11: '1'},
            'machine_id':      {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '0', 7: '1', 8: '2', 9: '3', 10: '4', 11: '5'},
            'process_time(s)': {0: 27, 1: 35, 2: 36, 3: 26, 4: 25, 5: 24, 6: 48, 7: 28, 8: 23, 9: 25, 10: 38, 11: 43}
        }'''
        # 如果输入是字典，转换为DataFrame
        if isinstance(MBOM, dict):
            df = pd.DataFrame(MBOM)
        else:
            df = MBOM

        # 获取唯一的产品类型和阶段，并按数值排序
        product_types = sorted(map(int, df['product_type'].unique()))
        stages = sorted(map(int, df['stage'].unique()))
        num_types = len(product_types)
        num_stages = len(stages)

        # 初始化结果结构：每个阶段一个字典
        result = [{} for _ in stages]

        # 为每个阶段构建机器字典
        for stage_idx in stages:
            stage_str = str(stage_idx)
            stage_data = df[df['stage'] == stage_str]

            for _, row in stage_data.iterrows():
                # machine_id = 'M' + str(row['machine_id'])
                machine_id = int(row['machine_id'])
                product_type = int(row['product_type'])
                time = row['process_time(s)']

                # 如果机器尚未在字典中，初始化全None列表
                if machine_id not in result[stage_idx]:
                    result[stage_idx][machine_id] = [None] * len(product_types)

                # 将处理时间放入对应位置
                result[stage_idx][machine_id][product_type] = time

        # [{'M0': [27, 48], 'M1': [35, 28], 'M2': [36, 23]}, {'M3': [26, 25], 'M4': [25, 38], 'M5': [24, 43]}]
        return result, num_types, num_stages

    def generate_schedule(self, platform) -> pd.DataFrame:
        """
        动态生成调度计划
        :param platform: 仿真测试平台
        :return: 调度计划DataFrame
                schedule_df: 调度计划DataFrame，包含以下列:
                    - task_id: 任务ID (格式: order_id-process_id)
                    - machine_id: 设备ID
                    - start_time: 计划开始时间
        """

        # 获取当前仿真状态
        # platform 提供的函数支持
        orders = platform.getOrders()
        """
        getOrders(self, only_unfinished=True) -> pd.DataFrame:
        获取订单数据及相关状态信息。

        该方法返回包含订单信息的DataFrame，包含订单基本信息、当前进度状态以及全系统订单完成率。

        Args:
            only_unfinished (bool, optional): 是否只返回未完成的订单。默认为True。
                - True: 仅返回尚未完成的订单
                - False: 返回所有订单（包括已完成的）

        Returns:
            pd.DataFrame: 包含订单信息的DataFrame，包含以下列：
                - order_id: 订单唯一标识
                - product_type: 产品类型
                - arrival_time: 订单到达时间
                - due_date: 订单交期
                - current_stage: 当前处理工序（如果当前没有进行中的工序则为上一个工序的结果。若首工序也未开始，则为None）
                - assigned_machine: 分配到的机器ID（如果当前没有分配则为为上一个工序的分配结果。若首工序也未开始，则为None）
                - start_time: 当前工序开始时间（如果未开始则为上一个工序的开始时间。若首工序也未开始，则为None）
                - end_time: 当前工序结束时间（如果未开始则为上一个工序的结束时间。若首工序也未开始，则为None）
                - fulfillment_rate: 当前系统订单达成率（已完成订单/所有已到达订单）
        """

        machines_status = platform.getCurrentMachineStatus()
        # print(machines_status)  # 哇

        """
        def getCurrentMachineStatus(self) -> pd.DataFrame:
        获取当前时刻所有机器的状态信息。

        该方法返回一个DataFrame，描述在当前时间点各机器的状态（空闲或正在执行的任务信息）。

        Returns:
            pd.DataFrame: 包含每台机器当前状态的DataFrame，包含以下列：
                - task_id: 当前执行的任务ID（如果机器空闲则为None）
                - start_time: 当前任务的开始时间（如果机器空闲则为None）
                - end_time: 当前任务的结束时间（如果机器空闲则为None）
        """

        MBOM = platform.getMBOM()
        """
        def getMBOM(self) -> pd.DataFrame:

        获取制造BOM(Bill of Materials)信息。

        该方法从仿真实例中提取产品的工艺路线信息，包括每个产品在各生产阶段可选用的设备和相应处理时间。

        Returns:
            pd.DataFrame: 包含制造BOM信息的DataFrame，包含以下列：
                - product_type: 产品类型标识（格式为"0"、"1"、"2"等）
                - stage: 生产阶段标识（格式为"0"、"1"、"2"等）
                - machine_id: 可用于该工序的设备ID
                - process_time(s): 在该设备上完成该工序所需的处理时间（秒）
        """
        process_time_info, num_types, num_stages = self.get_system_info(MBOM)
        '''MBOM_dict = {
            'product_type':    {0: '0', 1: '0', 2: '0', 3: '0', 4: '0', 5: '0', 6: '1', 7: '1', 8: '1', 9: '1', 10: '1', 11: '1'},
            'stage':           {0: '0', 1: '0', 2: '0', 3: '1', 4: '1', 5: '1', 6: '0', 7: '0', 8: '0', 9: '1', 10: '1', 11: '1'},
            'machine_id':      {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '0', 7: '1', 8: '2', 9: '3', 10: '4', 11: '5'},
            'process_time(s)': {0: 27, 1: 35, 2: 36, 3: 26, 4: 25, 5: 24, 6: 48, 7: 28, 8: 23, 9: 25, 10: 38, 11: 43}
        }'''

        # 获取当前时刻
        current_time = platform.getSimulationTime()
        # print(current_time)  # 哇
        """
        def getSimulationTime(self) -> float:
        获取当前仿真时间（从仿真开始起经过的时间）。

        该方法计算从仿真基础时间点（通常是仿真启动时间）到当前时刻经过的时间。

        Returns:
            float: 仿真经过的时间（以秒为单位）
        """

        orders = [order.to_dict() for _, order in orders.iterrows()]
        for order in orders:
            self.type_of_order_[order["order_id"]] = order["product_type"]
            self.due_time_of_order_[order["order_id"]] = order["due_date"]
        for machine_idx, info in machines_status.iterrows():

            # machines_status 有，但 orders 不存在，比如最后一个 stage 的
            if info.task_id is not None and \
                    not any([order["order_id"] == info.task_id.split('-')[1] for order in orders]):
                # print(info.task_id)  # 哇

                # if info.end_time <= current_time:
                if info.end_time == current_time:
                    next_task_id = '-'.join(info.task_id.split('-')[:2]) + f'-{int(info.task_id.split("-")[-1]) + 1}'
                    if next_task_id in machines_status['task_id'].values:
                        if int(machines_status[machines_status["task_id"] == next_task_id].start_time.values[
                                   0]) == info.end_time:
                            continue  # 同一个订单，两个工序无缝衔接的前一个工序，已经完成了，不需要再处理，直接 continue，不会加到 orders
                        else:
                            ll = 1
                    else:
                        if info.task_id.endswith(str(num_stages - 1)):
                            continue
                        else:
                            ll = 1
                elif info.start_time == current_time:
                    ll = 1  # 同一个订单，两个工序无缝衔接的后一个工序，不 continue，正常加入到 orders
                elif info.end_time < current_time:
                    ll = 1
                elif info.start_time > current_time:  # TODO 斟酌一下，这个只是暂时解决了，看下为什么会出现时间错误
                    ll = 1  # TODO:待定，需要问一下能不能重新分配
                    continue  # 如果可以重新分配，就忽略掉这个

                # 最后一个 stage 肯定会，但可能也存在其他的 stage 也出现 machines_status 有，但 orders 不存在的情况
                if info.task_id.endswith(str(num_stages - 1)):
                    ll = 1
                else:
                    next_task_id = '-'.join(info.task_id.split('-')[:2]) + f'-{int(info.task_id.split("-")[-1]) + 1}'
                    if next_task_id in machines_status['task_id'].values:
                        continue
                    else:
                        ll = 1

                _, order_id, stage_idx = info.task_id.split('-')
                type_id = self.type_of_order_[order_id]

                if info.end_time - info.start_time != process_time_info[int(stage_idx)][int(machine_idx)][int(type_id)]:
                    ll = 1  # TODO: to check
                # if self.due_time_of_order_[order_id] <= current_time
                orders.append({
                    "order_id": order_id,
                    "product_type": str(type_id),
                    "due_date": self.due_time_of_order_[order_id],
                    "current_stage": stage_idx,
                    "assigned_machine": str(machine_idx),
                    "start_time": info.start_time,
                    "end_time": info.end_time,
                })

        tmp = []
        for order in orders:
            if current_time >= order['due_date']:
                continue
            tmp.append(order)
        orders = tmp

        (place_names, transition_names,
         post_place_names_of_transition_named, pre_place_names_of_transition_named,
         m_of_place_named,
         x_of_place_named,
         delay_of_place_named,
         order_of_place_named,
         cur_time) = get_pn_state(
            process_time_info, num_types, num_stages, orders, current_time,
        )

        actions = []

        enable_transitions = get_enable_transitions(
            m_of_place_named,
            pre_place_names_of_transition_named,
        )
        # 选择一个可执行的转换
        while enable_transitions:
            # firing_transition_name = enable_transitions[0]
            if any([enable_transition.startswith('begin') for enable_transition in enable_transitions]):
                firing_transition_name = random.choice(
                    [enable_transition for enable_transition in enable_transitions if
                     enable_transition.startswith('begin')])
            else:
                firing_transition_name = random.choice(enable_transitions)

            if firing_transition_name.startswith("begin"):
                _, _, _, m_str, _ = firing_transition_name.split('_')
                if m_str not in m_of_place_named or m_of_place_named[m_str] == 0:
                    # print(firing_transition_name, m_of_place_named[m_str])  # 哇
                    pass

            actions.append([cur_time, firing_transition_name])
            # print(cur_time, firing_transition_name)
            (place_names, transition_names,
             post_place_names_of_transition_named, pre_place_names_of_transition_named,
             m_of_place_named,
             x_of_place_named,
             delay_of_place_named,
             order_of_place_named,
             cur_time) = get_next(
                firing_transition_name,
                place_names, transition_names,
                post_place_names_of_transition_named, pre_place_names_of_transition_named,
                m_of_place_named,
                x_of_place_named,
                delay_of_place_named,
                order_of_place_named,
                cur_time,
            )
            enable_transitions = get_enable_transitions(
                m_of_place_named,
                pre_place_names_of_transition_named,
            )

        # 在这里实现具体的调度算法
        # 调度规划算法 （随机选择）
        schedule_data = []
        order_of_order_id = {order["order_id"]: order for order in orders}
        for timestamp, action in actions:
            if action.startswith("begin"):
                # print(timestamp, action)
                _, s_str, t_str, m_str, o_str = action.split('_')
                if timestamp + delay_of_place_named[f'{s_str}_{t_str}_{m_str}'] > order_of_order_id[o_str[1:]][
                    'due_date']:
                    continue
                schedule_data.append(
                    {
                        'task_id': 'M-' + o_str[1:] + f'-{s_str[1:]}',
                        'machine_id': m_str[1:],
                        'start_time': timestamp
                    })
                if machines_status.iloc[int(m_str[1:])].task_id is not None and timestamp < machines_status.iloc[
                    int(m_str[1:])].end_time:
                    ll = 1  # 无视 machine status 的时间错误，使用原来的时间，并没有被系统阻止

        """
        输出DataFrame结果
        schedule_df: 调度计划DataFrame，包含以下列:
                    - task_id: 任务ID (格式: order_id-process_id)
                    - machine_id: 设备ID
                    - start_time: 计划开始时间
        """
        # print('=' * 50)
        # print(orders)  # 哇
        # print(schedule_data)  # 哇
        return pd.DataFrame(schedule_data)
