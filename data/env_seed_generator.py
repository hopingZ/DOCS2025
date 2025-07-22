
import random

import numpy as np


def generate_process_time_info(
        num_types,
        num_stages,
        num_machines_per_stage,
        process_time_lb,
        process_time_rb
):
    process_time_info = []
    machine_idx = 0
    for stage_idx in range(num_stages):
        process_time_info.append({})
        for i in range(num_machines_per_stage):
            process_time_info[-1][machine_idx] = []
            for j in range(num_types):
                process_time_info[-1][machine_idx].append(random.randint(process_time_lb, process_time_rb))
            machine_idx += 1

    return process_time_info


def get_num_types_and_stages(process_time_info):
    num_types = len(process_time_info[0][0])
    num_stages = len(process_time_info)
    return num_types, num_stages


def generate_intervals(num_events, lambda_param):
    """
    生成指定数量的指数分布间隔时间
    :param num_events: 要生成的间隔数量
    :param lambda_param: 指数分布的λ参数
    :return: 间隔时间数组
    """
    return np.random.exponential(scale=1/lambda_param, size=num_events)


def generate_arrival_times(start_time, num_events, lambda_param):
    """
    从指定起始时间生成新的到达时间序列
    :param start_time: 起始时间
    :param num_events: 要生成的到达事件数量
    :param lambda_param: 指数分布的λ参数
    :return: 到达时间数组
    """
    intervals = generate_intervals(num_events, lambda_param)
    arrival_times = np.cumsum(intervals) + start_time
    return arrival_times


def generate_orders(num_orders, lam, due_lb, due_rb, num_types):
    order_types = []
    for i in range(num_orders):
        order_types.append(random.randint(0, num_types - 1))

    begin_times = generate_arrival_times(0, num_orders, lam).astype(int)
    end_times = begin_times.copy()
    for i in range(num_orders):
        end_times[i] += random.randint(due_lb, due_rb)

    orders = []
    for i in range(num_orders):
        orders.append({
            'arrival_time': int(begin_times[i]),
            'due_date': int(end_times[i]),
            'order_id': str(i + 1),
            'product_type': str(order_types[i]),
        })

    return orders


def env_seed_generator(
    lam,
    num_orders,
    process_time_lb,
    process_time_rb,
    due_lb,
    due_rb,
    num_types, num_stages,
    num_machines_per_stage,
):
    process_time_info = generate_process_time_info(
        num_types,
        num_stages,
        num_machines_per_stage,
        process_time_lb,
        process_time_rb
    )
    orders = generate_orders(num_orders, lam, due_lb, due_rb, num_types)
    return process_time_info, orders


def save_env_seed_as_txt(process_time_info, orders, fp):
    num_types, num_stages = get_num_types_and_stages(process_time_info)
    num_machines_per_stage = len(process_time_info[0])
    with open(fp, 'w') as fw:
        fw.write(f'{num_types} {num_stages} {num_machines_per_stage * num_stages}\n')
        fw.write(f'{" ".join([str(num_machines_per_stage) for _ in range(num_stages)])}\n')
        data = []
        for stage_data in process_time_info:
            for machine_idx, times in stage_data.items():
                data.append(times)
        for line in np.array(data).T:
            fw.write(' '.join([str(x) for x in line]) + '\n')

        fw.write(' '.join([orders[i]['order_id'] for i in range(len(orders))]) + '\n')
        fw.write(' '.join([orders[i]['product_type'] for i in range(len(orders))]) + '\n')
        fw.write(' '.join([str(orders[i]['arrival_time']) for i in range(len(orders))]) + '\n')
        fw.write(' '.join([str(orders[i]['due_date']) for i in range(len(orders))]) + '\n')


def load_env_seed_from_txt(file_path):
    with open(file_path, 'r') as f:
        # 读取第一行：类型数、阶段数、总机器数
        line = f.readline().split()
        num_types = int(line[0])
        num_stages = int(line[1])
        total_machines = int(line[2])

        # 读取第二行：每个阶段的机器数
        machines_per_stage = list(map(int, f.readline().split()))

        # 读取处理时间矩阵（类型 x 机器）
        process_times = []
        for _ in range(num_types):
            row = list(map(int, f.readline().split()))
            process_times.append(row)

        # 转置矩阵：机器 x 类型
        machine_times = []
        for j in range(total_machines):
            mt = []
            for i in range(num_types):
                mt.append(process_times[i][j])
            machine_times.append(mt)

        # 重建process_time_info数据结构
        process_time_info = []
        machine_idx = 0
        for stage in range(num_stages):
            stage_dict = {}
            for _ in range(machines_per_stage[stage]):
                stage_dict[machine_idx] = machine_times[machine_idx]
                machine_idx += 1
            process_time_info.append(stage_dict)

        # 读取订单数据
        order_ids = f.readline().split()
        product_types = f.readline().split()
        arrival_times = list(map(int, f.readline().split()))
        due_dates = list(map(int, f.readline().split()))

        # 构建orders列表
        orders = []
        for i in range(len(order_ids)):
            orders.append({
                'order_id': order_ids[i],
                'product_type': product_types[i],
                'arrival_time': arrival_times[i],
                'due_date': due_dates[i]
            })

    return process_time_info, orders


def unit_test():
    process_time_info, orders = env_seed_generator(
        lam=10,
        num_orders=100,
        process_time_lb=1,
        process_time_rb=10,
        due_lb=1,
        due_rb=10,
        num_types=3,
        num_stages=3,
        num_machines_per_stage=2,
    )


if __name__ == '__main__':
    unit_test()
