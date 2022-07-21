# coding=utf-8
'''
调用[不同的]dataloader + filter, 得到不同数据清洗规则下的dataframe.

'''

import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import random
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import xgboost as xgb


def strip_a(a):
    a = a.strip("['] ")
    return a

def slim_thickness(a):
    inds = [0, 2, 3, 4, 5]
    res_a = [a[i] for i in inds]

    return res_a

import pandas as pd
cx_data1 = pd.read_csv(r'../resources/data/0910_cx_data.csv')
cx_data2 = pd.read_csv(r'../resources/data/0917_cx_data.csv')

cc_data1 = pd.read_csv(r'../resources/data/0910_cc_data.csv')
cc_data2 = pd.read_csv(r'../resources/data/0917_cc_data.csv')

# 合并910 917的数据
all_cx = pd.concat([cx_data1, cx_data2], join='inner')
all_cc = pd.concat([cc_data1, cc_data2], join='inner')

# 先测试cx
def generate_data(all_cx):
    X, Y = [], []
    lens = len(all_cx)
    for i in range(lens):
        # 5层可调膜厚
        thickness = [float(strip_a(a)) for a in all_cx.iloc[i]['Thickness'].split(',')]
        thickness = slim_thickness(thickness)
        # 本炉机器启动时间
        start_time = [all_cx.iloc[i]['Start_time']]
        # 本炉耗材情况
        Consumables = [float(strip_a(a)) for a in all_cx.iloc[i]['Consumables'].split(',')]
        # 本炉所属清洗周期
        cycle_index = [all_cx.iloc[i]['clean_cycle_index']]
        x = thickness + start_time + Consumables + cycle_index
        y = [float(strip_a(a)) for a in all_cx.iloc[i]['single_lab_curve'].split(',')]
        # 剔除异常数据
        if max(y) < 12 and y[9] < 2:
            X.append(x)
            Y.append(y)

    return X, Y


def report_best_scores(results, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            parameter = results['params'][candidate]
    tmp = dict()
    for k, v in parameter.items():
        tmp[k[11:]] = v
    data = json.dumps(tmp)
    with open(r'./parameter.json', 'w') as js_file:
        js_file.write(data)


def hyperparameter_searching(X_train, y_train):
    xgb_model = xgb.XGBRegressor()
    xgb_model = MultiOutputRegressor(xgb_model)

    params = {
        "estimator__colsample_bytree": uniform(0.7, 0.3),
        "estimator__gamma": uniform(0, 0.5),
        "estimator__learning_rate": uniform(0.01, 0.1),  # default 0.1
        "estimator__max_depth": randint(2, 5),           # default 3
        "estimator__n_estimators": randint(60, 100),     # default 100
        "estimator__subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=100, cv=5, verbose=1,
                                n_jobs=4, return_train_score=True)
    search.fit(X_train, y_train)

    report_best_scores(search.cv_results_, 5)


def overfiting(X, Y, param):
    dfull = xgb.DMatrix(X, Y)
    num_round = 200

    cvresult1 = xgb.cv(param, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()



def show_lab(gt, pred):
    a = [380+i*5 for i in range(81)]
    for i in range(len(gt)):
        if i == 0:
            plt.plot(a, gt[i], color='cornflowerblue', label='gt')
            plt.plot(a, pred[i], color='pink', label='pred')
        else:
            plt.plot(a, gt[i], color='cornflowerblue')
            plt.plot(a, pred[i], color='pink')
    plt.legend()
    plt.show()


def simple_run(X_train, X_test, y_train, y_test, params):
    model = xgb.XGBRegressor(objective="reg:linear", **params)
    model = MultiOutputRegressor(model)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    tmp = model.predict(X_train)
    print("train mse: {}".format(mean_squared_error(tmp, y_train)))
    print("test mse: {}".format(mean_squared_error(pred, y_test)))

    # 随机可视化10条样本看看
    # indexs = random.sample(range(0, len(y_test)), 10)
    # gt_indexs = [y_test[ind] for ind in indexs]
    # pred_indexs = [pred[ind] for ind in indexs]
    show_lab(y_test, pred)


X, Y = generate_data(all_cx)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# 参数搜索
# hyperparameter_searching(X_train, y_train)
params = json.load(open(r'./parameter.json', 'r'))
# overfiting(X, Y, params)
simple_run(X_train, X_test, y_train, y_test, params)
