# coding=utf-8
import pandas as pd
from function_thick2lab import function_thick2lab
import matplotlib.pyplot as plt


def strip_a(a):
    a = a.strip("['] ")
    return a

aa = [i for i in range(81)]
cx_file = r'./1008_cx_data.csv'
cx_data = pd.read_csv(cx_file)
best_cx = [6.6524, 4.5071, 2.2922, 1.1451, 0.7214, 0.4108, 0.2415, 0.3834, 0.5492, 0.5964, 0.6647, 0.8838, 1.0936, 1.0963, 0.992, 0.972, 1.0551, 1.1061, 1.0686, 0.927, 0.8003, 0.7419, 0.7491, 0.7563, 0.7154, 0.6076, 0.4921, 0.4414, 0.4152, 0.4251, 0.4191, 0.3947, 0.3412, 0.2696, 0.2235, 0.1996, 0.1907, 0.1965, 0.1844, 0.1592, 0.1281, 0.0889, 0.0598, 0.0446, 0.031, 0.0375, 0.0398, 0.0386, 0.0287, 0.0257, 0.0249, 0.0326, 0.0593, 0.093, 0.1355, 0.1803, 0.2255, 0.268, 0.3089, 0.3568, 0.4025, 0.4695, 0.5404, 0.6328, 0.7406, 0.8652, 1.0015, 1.1388, 1.2783, 1.4115, 1.543, 1.6709, 1.8066, 1.9292, 2.0527, 2.1884, 2.3387, 2.5097, 2.6949, 2.8866, 3.1036]
lab = [float(strip_a(a)) for a in cx_data.iloc[33]['single_lab_curve'].split(',')]
thickness = [float(strip_a(a)) for a in cx_data.iloc[33]['Thickness'].split(',')]
function = function_thick2lab(thickness)

thickness.pop(6)
thickness.pop(1)
print("原始膜厚: {}".format(thickness))

diff_base_m_f = [function[i] - lab[i] for i in range(81)]
function2 = [best_cx[i] + diff_base_m_f[i] for i in range(81)]
plt.plot(aa, lab, label='original cx', color='pink')
# plt.plot(aa, function, label='f1', color='cornflowerblue')
plt.plot(aa, best_cx, label='best cx', color='yellow')
# plt.plot(aa, function2, label='f2', color='blue')
plt.legend()
plt.show()


# 开启膜厚搜索
from flaml import tune
import numpy as np
from sklearn.metrics import mean_squared_error


std_thinckness = thickness  # 待调节的基础膜厚
finetune_thre1 = [0.0, 0.0, 0.0, 0.5, 0.5] # 膜厚最多可相对于膜厚标准值调整比例-第一级（生产常用）
finetune_thre2 = [0.5, 0.5, 0.5, 1, 1] # 膜厚最多可相对于膜厚标准值调整比例-第二级别（权限较高）


def get_model_predict(thickness_list):
    thickness_list = [thickness_list[0]] + [4] + thickness_list[1:] + [35]
    # print(thickness_list)
    return function_thick2lab(thickness_list)


def evaluate_config(config):
    pred_lab_curve = get_model_predict([config['x1'],config['x2'],config['x3'],config['x4'],config['x5']])
    metric = mean_squared_error(function2, pred_lab_curve)
    # use tune.report to report the metric to optimize
    tune.report(metric=metric)

analysis = tune.run(
    evaluate_config,    # the function to evaluate a config
    config={
        'x1': tune.uniform(lower=std_thinckness[0]-finetune_thre2[0], upper=std_thinckness[0]+finetune_thre2[0]), # 取值范围
        'x2': tune.uniform(lower=std_thinckness[1]-finetune_thre2[1], upper=std_thinckness[1]+finetune_thre2[1]), # 取值范围
        'x3': tune.uniform(lower=std_thinckness[2]-finetune_thre2[2], upper=std_thinckness[2]+finetune_thre2[2]), # 取值范围
        'x4': tune.uniform(lower=std_thinckness[3]-finetune_thre2[3], upper=std_thinckness[3]+finetune_thre2[3]), # 取值范围
        'x5': tune.uniform(lower=std_thinckness[4]-finetune_thre2[4], upper=std_thinckness[4]+finetune_thre2[4]), # 取值范围
    }, # the search space
    low_cost_partial_config={'x1':std_thinckness[0],
                             'x2':std_thinckness[1],
                             'x3':std_thinckness[2],
                             'x4':std_thinckness[3],
                             'x5':std_thinckness[4]},    # a initial (partial) config with low cost
    metric='metric',    # the name of the metric used for optimization
    mode='min',         # the optimization mode, 'min' or 'max'
    num_samples=-1,    # the maximal number of configs to try, -1 means infinite
    time_budget_s=60,   # the time budget in seconds 优化时间限制, 60s停止优化
    local_dir='logs/',  # the local directory to store logs
    )

# print(analysis.best_trial.last_result)  # the best trial's result

# the best config
res = analysis.best_config
print("推荐膜厚: {}".format(res))
# 原膜厚:   [24.82, 4, 11.54, 36.32,  102.7,  94.6,  35]
# 推荐膜厚: [25.17,  4,  12,   35.82, 102.47,  93.6,  35]