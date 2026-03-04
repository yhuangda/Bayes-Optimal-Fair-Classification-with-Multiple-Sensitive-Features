import os
import numpy as np
import time

# 定义参数列表
base_command = "nohup python bayesfair.py --config-name base"
output_dir = "./logs"  # 保存输出日志的文件夹

# 确保日志文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 数据集列表 - 只包含用户指定的数据集
dataset_trial = ['adult', 'compas', 'adult_multi', 'compas_multi']

# 是否允许敏感特征
allow_sens_list = [False]

# 方法列表 - 只包含用户指定的方法
methods = ["Bayes_fair_logit", "Bayes_fair", "Bayes_fair_inprocess", "Bayes_fair_inprocess_logit"]

# 公平性概念列表 - 包含用户指定的所有概念
fair_notions = ['DP', 'EOpp', 'AP', 'PredEqual']

# 公平性类型列表
fair_type = ['MD', 'MR']

# 其他参数
c = 0.5
fair_tolerance = 0.3
commands = []
grid_interval = 0.1
max_lambda = 1

# 生成所有组合的命令
for method in methods:
    for allow_sens in allow_sens_list:
        for dataset in dataset_trial:
            for fair_notion in fair_notions:
                for type in fair_type:
                    command = (
                        f"{base_command} dataset={dataset}  method={method}  parameters.fair_notion={fair_notion} "
                        f"parameters.type={type}  parameters.fair_tolerance={fair_tolerance}  parameters.allow_sens={allow_sens}  parameters.max_lambda={max_lambda} parameters.grid_interval={grid_interval} parameters.c={c}"
                        f"> {output_dir}/{method}_set_{dataset}_{fair_notion}_{type}_{allow_sens}.log 2>&1 &"
                    )
                    commands.append(command)

# 执行所有命令
for cmd in commands:
    print(f"Running command: {cmd}")
    os.system(cmd)
    time.sleep(10)

print("All commands have been dispatched.")
