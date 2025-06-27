import pandas as pd


# 参赛队伍算法（请封装成类）
class SchedulingAlgorithm:
    def __init__(self):
        ...

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
        """
        def getSimulationTime(self) -> float:
        获取当前仿真时间（从仿真开始起经过的时间）。

        该方法计算从仿真基础时间点（通常是仿真启动时间）到当前时刻经过的时间。

        Returns:
            float: 仿真经过的时间（以秒为单位）
        """

        # 在这里实现具体的调度算法
        # 调度规划算法 （随机选择）
        schedule_data = []
        schedule_task_list = []
        for machine_id, machine in machines_status.iterrows():
            if machine['task_id'] is None:
                # 设备空闲，随机选择合法作业
                for _, order in orders.iterrows():
                    if order['order_id'] in schedule_task_list:
                        continue
                    if order['due_date'] - current_time <= 150:
                        continue

                    if order['current_stage'] is None:
                        # 订单未处理
                        if machine_id in [0, 1, 2, 3, 4]:
                            # 可以处理首工序
                            schedule_data.append(
                                {
                                    'task_id': 'M-' + order['order_id'] + '-0',
                                    'machine_id': str(machine_id),
                                    'start_time': current_time
                                })
                            schedule_task_list.append(order['order_id'])
                            break
                        else:
                            continue
                    if order['end_time'] <= current_time:
                        # 订单已经结束生产可以安排新的任务
                        stage = int(order['current_stage']) + 1
                        if (
                                (stage == 1 and (machine_id in [5, 6, 7, 8, 9]))
                                or (stage == 2 and (machine_id in [10, 11, 12, 13, 14]))
                                or (stage == 3 and (machine_id in [15, 16, 17, 18, 19]))
                                or (stage == 4 and (machine_id in [20, 21, 22, 23, 24]))
                        ):
                            # 设备符合生产要求
                            schedule_data.append(
                                {
                                    'task_id': 'M-' + order['order_id'] + '-' + str(stage),
                                    'machine_id': str(machine_id),
                                    'start_time': current_time
                                })
                            schedule_task_list.append(order['order_id'])
                            break

        """
        输出DataFrame结果
        schedule_df: 调度计划DataFrame，包含以下列:
                    - task_id: 任务ID (格式: order_id-process_id)
                    - machine_id: 设备ID
                    - start_time: 计划开始时间
        """

        return pd.DataFrame(schedule_data)