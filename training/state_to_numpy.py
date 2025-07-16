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
