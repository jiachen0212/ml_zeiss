# coding=utf-8
'''
hyperparameter_searching + xgboost model 通路

模型多输入单一输出 场景可套用 ~

'''
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
import json
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as TTS


def report_best_scores(results, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            # print("Model with rank: {0}".format(i))
            # print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            #       results['mean_test_score'][candidate],
            #       results['std_test_score'][candidate]))
            parameter = results['params'][candidate]
            # print("Parameters: {0}".format(parameter))

    # 超参落盘
    data = json.dumps(parameter)
    with open(r'./parameter.json', 'w') as js_file:
        js_file.write(data)


def hyperparameter_searching(X, Y):

    xgb_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.9, 0.1),
        "gamma": uniform(0, 0.),   # gamma越小, 模型越复杂..
        "learning_rate": uniform(0.01, 0.5),  # default 0.1
        "max_depth": randint(3, 10),  # default 3
        "n_estimators": randint(80, 150),  # default 100
        "subsample": uniform(0.6, 0.4)

    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, 5)


def save_model(model, model_path):
    model.save_model(model_path)

def cross_val(X_train, Y_train):

    parameters = json.load(open(r'./parameter.json', 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, Y_train)
    # save_model(xgb_model, r'./xgb.model')

    return xgb_model


def eval_model(parameters, x):
    model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    model.load_model(r'./xgb.model')
    res = model.predict(np.array([x]))

    return res.tolist()



def load_data():
    X, Y = np.array([]), np.array([])

    return X, Y

if __name__ == '__main__':

    # load data
    X, Y = load_data()
    X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=33)
    
    hyperparameter_searching(X, Y)
    Model = cross_val(X, Y)








