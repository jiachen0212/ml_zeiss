# coding=utf-8
'''
1. model1: pre_sensor + cur_thick -> cur_sensor   this part  tree model
2. sensor -> lab_curve

# 0930 添加输入端参数搜索功能..


'''

def generate_data():

    file = r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab_thickness.json'
    data = json.load(open(file, 'r'))

    sensors = []
    thickness = []
    for k, v in data.items():
        snesor16 = [float(a) for a in v[0].split(',')[:-1][10:][::3]]
        thickness.append([float(a) for a in v[0].split(',')[:-1][:10]])
        sensors.append(snesor16)

    X, Y = [], []
    for i in range(len(sensors)-1):
        X.append(thickness[i+1] + sensors[i])
        Y.append(sensors[i+1])

    # train data
    # 暂时考虑跨越周期一起做, 模型学习的是机器前后状态互相影响的关系

    train_x, test_x = X[:105] + X[115:], X[105:115]
    train_y, test_y = Y[:105] + Y[115:], Y[105:115]

    return train_x, test_x, train_y, test_y


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
import pandas as pd



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
    with open(r'./parameter_cc.json', 'w') as js_file:
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


def show_lab(gt, pred, title=None):
    a = [i for i in range(16)]
    if title:
        plt.title(title)
    for i in range(len(gt)):
        if i == 0:
            plt.plot(a, gt[i], color='cornflowerblue', label='gt')
            plt.plot(a, pred[i], color='pink', label='pred2')
        else:
            plt.plot(a, gt[i], color='cornflowerblue')
            plt.plot(a, pred[i], color='pink')
    plt.legend()
    plt.show()



def save_model(model, model_path):

    model.save_model(model_path)

def simple_run(X_train, X_test, y_train, y_test, params, test_last_x, test_last_y):
    model = xgb.XGBRegressor(objective="reg:linear", **params)
    model = MultiOutputRegressor(model)
    model.fit(X_train, y_train)

    # save model
    save_model(model, r'./tree_thick_sensor2sensor.model')

    pred = model.predict(X_test)
    tmp = model.predict(X_train)
    print("train mse: {}".format(mean_squared_error(tmp, y_train)))
    print("test mse: {}".format(mean_squared_error(pred, y_test)))

    show_lab(tmp, y_train, 'train')
    show_lab(tmp, y_train, 'val')

    # show inference test
    for i, test_ in enumerate(test_last_x):
        inference_pred = model.predict([test_])[0]
        print("inference test mse: {}".format(mean_squared_error(inference_pred, test_last_y[i])))

    diffs = [0,0,0.5,1.1,1.1]

    single_test_x, single_test_y = test_last_x[6], test_last_y[6]
    new_test_x = [diffs[i]+single_test_x[:5][i] for i in range(5)] + single_test_x[5:]
    pred1 = model.predict([single_test_x])[0]
    pred2 = model.predict([new_test_x])[0]
    print("pred1: {}".format(pred1))
    print("pred2: {}".format(pred2))



def inference_test():

    train_x, test_x, train_y, test_y = generate_data()

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.3)

    # 全量数据上搜索参数
    # hyperparameter_searching(train_x, train_y)
    params = json.load(open(r'./parameter_cc.json', 'r'))
    simple_run(X_train, X_val, y_train, y_val, params, test_x, test_y)


if __name__ == "__main__":

    inference_test()




