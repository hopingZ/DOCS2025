import os
from competitionPlatform import CompetitionPlatform
import pandas as pd

# 运行仿真
if __name__ == '__main__':
    # 0. 配置竞赛案例
    instance_name = 'num1000_lam0.03_change0__1.txt'
    instance_names = []
    path = os.path.join('schedulePlat', 'data', 'instance', 'competition', instance_name)

    # 打印当前工作目录和文件路径，以便调试
    print("当前工作目录是:", os.getcwd())
    print("尝试打开的文件路径是:", path)

    # 1. 创建仿真平台实例
    platform = CompetitionPlatform()

    # 2. 创建参赛队伍算法实例
    from algorithm_Demo import SchedulingAlgorithm

    team_algorithm = SchedulingAlgorithm()

    # 4. 运行仿真
    result = platform.run_simulation(path, team_algorithm, False)
    """
    def run_simulation(self, path, algorithm_module, isTimeout=True) -> dict:
    运行仿真案例
        :param path: 仿真案例的路径
        :param algorithm_module: 动态调度算法
        :param isTimeout:  是否开启30s实时调度限制
        :return: 返回仿真结果

    """
    # 5. 输出结果
    orders = platform.getOrders()
    print("\n仿真结果:")
    print(f"订单达成率: {orders['fulfillment_rate'].values[0]:.2%}")
    platform.getGantta(600, 900)
    """
    def getGantta(self, startTime, endTime)
    生成甘特图
    :param startTime: 开始时间点
    :param endTime: 结束时间点

    """
    machine_records = platform.getMachineRecord()
    """
    def getMachineRecord(self, hasGantta=False) -> Dict[str, pd.DataFrame]:
    获取所有机器的历史作业记录。

    该方法返回一个字典，包含每台机器已分配的所有任务信息。

    Args:
        hasGantta (bool, optional): 是否返回甘特图。默认为False。

    Returns:
        Dict[str, pd.DataFrame]: 以机器ID为键的字典，值为包含该机器所有作业记录的DataFrame，
            DataFrame包含以下列：
                - task_id: 任务ID
                - start_time: 任务开始时间
                - end_time: 任务结束时间
    """
    with pd.ExcelWriter('machine_records.xlsx') as writer:
        for sheet_name, df in machine_records.items():
            df.to_excel(writer, sheet_name=str(sheet_name), index=False)

    # 6 记录最终结果数据，保存在txt。
    print(f"{instance_name},{orders['fulfillment_rate'].values[0]},{result}", )
