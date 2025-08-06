from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List
from copy import deepcopy
import pandas as pd
import numpy as np
from torch.distributions import Categorical
import torch
from torch import nn
from environment.pn_env import PNEnv
from models.pncn import PNCN


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


def get_pn_state(
        process_time_info, num_types, num_stages, orders, cur_time,
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

    # 闲置订单，从等待池模板拷贝存放
    for order in orders:
        if order['current_stage'] is None or cur_time >= order["end_time"]:
            waiting_place_name = f"T{order['product_type']}_pool" if order['current_stage'] is None \
                else f"T{order['product_type']}_S{order['current_stage']}-S{int(order['current_stage']) + 1}_waiting"
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

    for order in orders:
        # 闲置中的订单，放在订单专属的等待池
        if order['current_stage'] is None or cur_time >= order["end_time"]:
            waiting_place_name = f"T{order['product_type']}_pool" if order['current_stage'] is None \
                else f"T{order['product_type']}_S{order['current_stage']}-S{int(order['current_stage']) + 1}_waiting"
            place_name = waiting_place_name + f"_O{order['order_id']}"
            if m_of_place_named[place_name] == 0:
                m_of_place_named[place_name] = 1
                order_of_place_named[place_name] = order["order_id"]
        # 不闲置的订单，放在加工流程中，机器资源被占用
        else:
            process_place_name = f"S{order['current_stage']}_T{order['product_type']}_M{order['assigned_machine']}"
            x_of_place_named[process_place_name] = cur_time - order['start_time']
            m_of_place_named[process_place_name] = 1
            m_of_place_named[f"M{order['assigned_machine']}"] = 0
            order_of_place_named[process_place_name] = order["order_id"]

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
        rests[i][rests[i] == np.inf] = 0.

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


# 参赛队伍算法（请封装成类）
class SchedulingAlgorithm:
    def __init__(self):
        self.type_of_order_ = {}
        self.due_time_of_order_ = {}
        self.greedy_guiding = True
        self.group_size = 5 + int(self.greedy_guiding)
        self.pn_envs = [PNEnv() for _ in range(self.group_size)]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load('model_checkpoint_0803_ins1_best_metric--409.pth', map_location='cpu')
        model = PNCN(
            num_classes=2, in_channel=4, num_pnc_layers=5, hidden_channel=256, expand_ratio=0.25,
            num_transformer_layers=3,
            num_attention_heads=4,
            transformer_intermediate_size=256,
            dropout=0,
        ).to(device)
        model.device = device
        model.load_state_dict(state_dict)
        model.eval()

        self.model = model

    @staticmethod
    def get_system_info(MBOM):
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
        print(f"订单达成率: {orders['fulfillment_rate'].values[0]:.2%}")
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
        # print(machines_status)
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

        # 获取当前时刻
        current_time = platform.getSimulationTime()
        """
        def getSimulationTime(self) -> float:
        获取当前仿真时间（从仿真开始起经过的时间）。

        该方法计算从仿真基础时间点（通常是仿真启动时间）到当前时刻经过的时间。

        Returns:
            float: 仿真经过的时间（以秒为单位）
        """

        # =====================================
        # 从 machines_status 获取数据补全 orders
        # =====================================
        orders = [order.to_dict() for _, order in orders.iterrows()]
        for order in orders:
            self.type_of_order_[order["order_id"]] = order["product_type"]
            self.due_time_of_order_[order["order_id"]] = order["due_date"]
        for machine_idx, info in machines_status.iterrows():
            # machines_status 有，但 orders 不存在，应该只会是最后一个 stage 的订单
            if info.task_id is not None and \
                    not any([order["order_id"] == info.task_id.split('-')[1] for order in orders]):
                assert info.task_id.endswith(str(num_stages - 1))

                _, order_id, stage_idx = info.task_id.split('-')
                orders.append({
                    "order_id": order_id,
                    "product_type": str(self.type_of_order_[order_id]),
                    "due_date": self.due_time_of_order_[order_id],
                    "current_stage": stage_idx,
                    "assigned_machine": str(machine_idx),
                    "start_time": info.start_time,
                    "end_time": info.end_time,
                })

        orders = [order for order in orders if current_time < order['due_date']]
        # print(orders)

        schedule_datas = []

        # 将获取到的数据整合成当前 pn_state
        pn_state = get_pn_state(
            process_time_info, num_types, num_stages, orders, current_time,
        )

        for pn_env in self.pn_envs:
            pn_env.set(pn_state, process_time_info, num_stages, self.due_time_of_order_)
            schedule_datas.append([])

        accumulated_rewards = [0 for _ in range(self.group_size)]
        trajectories = [[] for _ in range(self.group_size)]
        done_flags = [False for _ in range(self.group_size)]
        while True:
            enable_transition_idxs_of_env_idx = [[] for _ in range(self.group_size)]

            for env_idx, pn_env in enumerate(self.pn_envs):
                begin_transition_names = [
                    enable_transition for enable_transition in pn_env.cur_enable_transitions if
                    enable_transition.startswith('begin')
                ]

                # 预估未来时间，剔除不可能达成的变迁
                begin_transition_names_to_remove = []
                estimated_left_time_of_transition_named_ = {}
                for begin_transition_name in begin_transition_names:
                    s, t, m, o = begin_transition_name.split('_')[1:]
                    next_stage_time = pn_env.cur_pn_state.delay_of_place_named['_'.join([s, t, m])]
                    estimated_left_time = next_stage_time
                    t_idx = int(t[1:])
                    for s_idx in range(int(s[1:]) + 1, num_stages):
                        estimated_left_time += min([_[t_idx] for _ in process_time_info[s_idx].values()])
                    estimated_left_time_of_transition_named_[begin_transition_name] = estimated_left_time

                    if estimated_left_time > pn_env.due_time_of_order_[o[1:]] - pn_env.cur_pn_state.cur_time:
                        begin_transition_names_to_remove.append(begin_transition_name)

                for transition_idx, transition_name in enumerate(pn_env.cur_pn_state.transition_names):
                    if transition_name in pn_env.cur_enable_transitions and \
                            transition_name not in begin_transition_names_to_remove:
                        enable_transition_idxs_of_env_idx[env_idx].append(transition_idx)

            for env_idx, enable_transition_idxs in enumerate(enable_transition_idxs_of_env_idx):
                if len(enable_transition_idxs) == 0:
                    done_flags[env_idx] = True

            # 使用当前模型采样动作
            with torch.no_grad():
                if self.greedy_guiding:
                    pn_envs = self.pn_envs[:-1]
                else:
                    pn_envs = self.pn_envs

                logits = get_logits_batch(
                    [pn_env.cur_pn_state for pn_env in pn_envs],
                    [pn_env.due_time_of_order_ for pn_env in pn_envs],
                    self.model
                )  # [10, 255, 1]
                mask = torch.full_like(logits, fill_value=-torch.inf)
                for env_idx in range(mask.shape[0]):
                    enable_transition_idxs = enable_transition_idxs_of_env_idx[env_idx]
                    mask[env_idx, enable_transition_idxs, 0] = 0.

                probs = nn.functional.softmax(logits + mask, dim=-2)
                probs[probs.isnan()] = 1 / probs.shape[1]
                dist = Categorical(probs.squeeze())
                actions = dist.sample()

            if self.greedy_guiding and not done_flags[-1]:
                pn_env = self.pn_envs[-1]
                # =========================
                # 贪婪策略
                # =========================
                enable_transitions = deepcopy(pn_env.cur_enable_transitions)
                for transition_idx, transition_name in enumerate(pn_env.cur_pn_state.transition_names):
                    if transition_name in pn_env.cur_enable_transitions:
                        enable_transition_idxs_of_env_idx[-1].append(transition_idx)

                if any([enable_transition.startswith('begin') for enable_transition in enable_transitions]):
                    begin_transition_names = [
                        enable_transition for enable_transition in enable_transitions if
                        enable_transition.startswith('begin')
                    ]

                    # 预估未来时间，剔除不可能达成的变迁
                    begin_transition_names_to_remove = []
                    estimated_left_time_of_transition_named_ = {}
                    for begin_transition_name in begin_transition_names:
                        s, t, m, o = begin_transition_name.split('_')[1:]
                        next_stage_time = pn_env.cur_pn_state.delay_of_place_named['_'.join([s, t, m])]
                        estimated_left_time = next_stage_time
                        t_idx = int(t[1:])
                        for s_idx in range(int(s[1:]) + 1, num_stages):
                            estimated_left_time += min([_[t_idx] for _ in process_time_info[s_idx].values()])
                        estimated_left_time_of_transition_named_[begin_transition_name] = estimated_left_time

                        if estimated_left_time > pn_env.due_time_of_order_[o[1:]] - pn_env.cur_pn_state.cur_time:
                            begin_transition_names_to_remove.append(begin_transition_name)
                            enable_transition_idxs_of_env_idx[-1].remove(
                                pn_env.cur_pn_state.transition_names.index(begin_transition_name)
                            )

                    begin_transition_names = [begin_transition_name for begin_transition_name in begin_transition_names
                                              if begin_transition_name not in begin_transition_names_to_remove]

                    if begin_transition_names:
                        cost_times = [
                            pn_env.cur_pn_state.delay_of_place_named['_'.join(begin_transition_name.split('_')[1:-1])]
                            for begin_transition_name in begin_transition_names]
                        firing_transition_name = begin_transition_names[np.argmin(cost_times).item()]
                        s, t, m, o = firing_transition_name.split('_')[1:]
                        transition_names_using_m = [begin_transition_name for begin_transition_name in
                                                    begin_transition_names if m in begin_transition_name]
                        if len(transition_names_using_m) > 1:  # 有竞争
                            firing_transition_name = transition_names_using_m[0]
                            shortest_estimated_left_time = estimated_left_time_of_transition_named_[
                                firing_transition_name]
                            for transition_name in transition_names_using_m[1:]:
                                if estimated_left_time_of_transition_named_[transition_name] < \
                                        shortest_estimated_left_time:
                                    shortest_estimated_left_time = \
                                        estimated_left_time_of_transition_named_[transition_name]
                                    firing_transition_name = transition_name

                    else:
                        enable_transitions = [enable_transition for enable_transition in enable_transitions if
                                              enable_transition.startswith('end')]
                        working_place_names = [enable_transition.replace('end_', '') for enable_transition in
                                               enable_transitions]
                        left_times = [pn_env.cur_pn_state.delay_of_place_named[place_name] -
                                      (pn_env.cur_pn_state.x_of_place_named[
                                           place_name] if place_name in pn_env.cur_pn_state.x_of_place_named else 0)
                                      for place_name in working_place_names]
                        firing_transition_name = enable_transitions[np.argmin(left_times).item()]

                else:
                    working_place_names = [enable_transition.replace('end_', '') for enable_transition in
                                           enable_transitions]
                    left_times = [pn_env.cur_pn_state.delay_of_place_named[place_name] -
                                  (pn_env.cur_pn_state.x_of_place_named[
                                       place_name] if place_name in pn_env.cur_pn_state.x_of_place_named else 0)
                                  for place_name in working_place_names]
                    firing_transition_name = enable_transitions[np.argmin(left_times).item()]

                action = pn_env.cur_pn_state.transition_names.index(firing_transition_name)
                nwe_probs = torch.zeros(probs.shape[0] + 1, max(action + 1, probs.shape[1]), 1)
                nwe_probs[-1, action] = 1.0
                nwe_probs[:-1, :probs.shape[1]] = probs.clone()
                actions = torch.cat([actions, torch.tensor([action]).to(actions)])
                probs = nwe_probs.to(actions.device)

            # 更新环境并存储
            for env_idx in range(len(self.pn_envs)):
                if done_flags[env_idx]:
                    continue

                action = actions[env_idx]
                firing_transition_name = self.pn_envs[env_idx].cur_pn_state.transition_names[action]

                last_pn_state = deepcopy(self.pn_envs[env_idx].cur_pn_state)

                if firing_transition_name.startswith('begin'):
                    _, s_str, t_str, m_str, o_str = firing_transition_name.split('_')
                    schedule_datas[env_idx].append(
                        {
                            'task_id': 'M-' + o_str[1:] + f'-{s_str[1:]}',
                            'machine_id': m_str[1:],
                            'start_time': self.pn_envs[env_idx].cur_pn_state.cur_time
                        }
                    )
                _, reward, done, info = self.pn_envs[env_idx].step(firing_transition_name)

                if not done_flags[env_idx] and "ignore" not in info:
                    trajectories[env_idx].append((
                        last_pn_state,
                        None,
                        enable_transition_idxs_of_env_idx[env_idx],
                        action,
                        reward,
                        probs[env_idx, action],
                    ))

                accumulated_rewards[env_idx] += reward

                if done:
                    done_flags[env_idx] = True

            if all(done_flags):
                break

        max_accumulated_reward = np.max(accumulated_rewards)
        times = [env.cur_pn_state.cur_time for env in self.pn_envs]
        for i in range(len(times)):
            if accumulated_rewards[i] != max_accumulated_reward:
                times[i] = np.inf
        env_idx = np.argmin(times)
        # print(times, env_idx)
        # print(accumulated_rewards)
        schedule_data = schedule_datas[env_idx]
        # print(schedule_data)
        """
        输出DataFrame结果
        schedule_df: 调度计划DataFrame，包含以下列:
                    - task_id: 任务ID (格式: order_id-process_id)
                    - machine_id: 设备ID
                    - start_time: 计划开始时间
        """
        return pd.DataFrame(schedule_data)
