from .pn_state import PNState


def get_enable_transitions(
        pn_state, num_stages, process_time_info, due_time_of_order_
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

        # 剔除 overdue
        if enable and transition_name.startswith('begin'):
            s, t, m, o = transition_name.split('_')[1:]
            next_stage_time = pn_state.delay_of_place_named['_'.join([s, t, m])]
            estimated_left_time = next_stage_time
            t_idx = int(t[1:])
            for s_idx in range(int(s[1:]) + 1, num_stages):
                estimated_left_time += min([_[t_idx] for _ in process_time_info[s_idx].values()])

            if estimated_left_time > due_time_of_order_[o[1:]] - pn_state.cur_time:
                enable = False

        if enable:
            enable_transition_names.append(transition_name)

    return enable_transition_names


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


def get_next(
        pn_state, transition_name, due_time_of_order_, num_stages, process_time_info
):
    cost = 0
    done = False

    if transition_name.startswith('begin'):
        cur_pn_state = get_next_firing_begin_transition(transition_name, pn_state)
        done = False
        return cur_pn_state, cost, done

    # ==================
    # 处理 end transition
    # ==================
    cur_pn_state = get_next_firing_end_transition(transition_name, pn_state)

    # 会造成时间推移，所以顺便把已经完成的 process 弹出
    while True:
        # 寻找第一个已完成处理的 place
        completed_places = (
            place_name for place_name in cur_pn_state.x_of_place_named
            if cur_pn_state.x_of_place_named[place_name] == cur_pn_state.delay_of_place_named[place_name]
        )
        place_name = next(completed_places, None)
        if place_name is None:
            break

        transition_name = f"end_{place_name}"
        cur_pn_state = get_next_firing_end_transition(transition_name, cur_pn_state)

    # ==================
    # 过期结构拆除
    # ==================
    transitions_to_remove = []
    places_to_remove = []
    for place_name, order_idx in cur_pn_state.order_of_place_named.items():
        due_time = due_time_of_order_[order_idx]
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
            for s_idx in range(s_idx, num_stages):
                estimated_left_time += min([_[t_idx] for _ in process_time_info[s_idx].values()])
            if estimated_left_time > due_time_of_order_[o[1:]] - cur_pn_state.cur_time:
                remove = True

            if not remove and due_time <= cur_pn_state.cur_time:
                remove = True

            if remove:
                places_to_remove.append(place_name)
                overdue_transitions = [overdue_t for overdue_t in cur_pn_state.transition_names if
                                       overdue_t.endswith(f'_O{order_idx}')]
                transitions_to_remove.extend(overdue_transitions)

    for transition_to_remove in transitions_to_remove:
        cur_pn_state.post_place_names_of_transition_named.pop(transition_to_remove)
        cur_pn_state.pre_place_names_of_transition_named.pop(transition_to_remove)
        cur_pn_state.transition_names.remove(transition_to_remove)
    for place_to_remove in places_to_remove:
        cur_pn_state.order_of_place_named.pop(place_to_remove)
        cur_pn_state.place_names.remove(place_to_remove)
        cur_pn_state.m_of_place_named.pop(place_to_remove)
        if place_to_remove in cur_pn_state.x_of_place_named:
            cur_pn_state.x_of_place_named.pop(place_to_remove)
        if place_to_remove in cur_pn_state.delay_of_place_named:
            cur_pn_state.delay_of_place_named.pop(place_to_remove)

    cost += len(places_to_remove)

    enable_transitions = get_enable_transitions(
        cur_pn_state,
        num_stages,
        process_time_info,
        due_time_of_order_
    )
    if not enable_transitions:
        cost += len(cur_pn_state.order_of_place_named)
        done = True

    return cur_pn_state, cost, done
