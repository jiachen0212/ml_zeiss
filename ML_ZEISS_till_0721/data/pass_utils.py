# coding=utf-8
import numpy as np
import json
from sklearn.model_selection import RandomizedSearchCV, train_test_split as TTS
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error
import os
from sklearn.decomposition import PCA


def is_thick_modified(thick2, thick1):
    deta_thick = [abs(thick2[r] - thick1[r]) for r in range(len(thick2))]
    for deta_ in deta_thick:
        if deta_ > 0:
            return True
    return False

def makedir(save_dir):
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        for file in files:
            os.remove(os.path.join(save_dir, file))
    else:
        os.makedirs(save_dir)


def json2npy_pca(js_data, pca_n=0):
    # 把feature_json转成数组, 然后方便后续做pca降维
    feature_npy = []
    ks = []
    for k in js_data:
        ks.append(k)
        feature_npy.append(js_data[k])
    feature_npy = np.array(feature_npy)
    # 把feature中的nan置0
    feature_npy[np.isnan(feature_npy)] = 0
    pca_1 = PCA(n_components=pca_n)
    feature_npy = pca_1.fit_transform(feature_npy).tolist()
    pca_js_data = dict()
    for ind, k in enumerate(ks):
        pca_js_data[k] = feature_npy[ind]

    return pca_js_data


def report_best_scores(param_file, results, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            parameter = results['params'][candidate]

    data = json.dumps(parameter)
    with open(param_file, 'w') as js_file:
        js_file.write(data)


def xgb_hyperparameter_searching(X, Y, param_file):
    ml_model = xgb.XGBRegressor()

    params = {
        "colsample_bytree": uniform(0.1, 0.9),
        "gamma": uniform(0, 1),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 9),
        "n_estimators": randint(10, 150),
        "subsample": uniform(0.4, 0.6),
        "reg_lambda": uniform(0.1, 5),
    }

    search = RandomizedSearchCV(ml_model, param_distributions=params, random_state=6, n_iter=200, cv=5, verbose=1,
                                n_jobs=-1, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(param_file, search.cv_results_, n_top=5)


def save_model(model, model_path):

    model.save_model(model_path)


def load_model1_model2(ll, seed, pca):
    model1s, model2s = [], []
    for ind in range(ll):
        params1 = json.load(open(r'./parameter_{}_index{}_seed{}_pca{}_minus.json'.format('model1', ind, seed, pca), 'r'))
        params2 = json.load(open(r'./parameter_{}_index{}_seed{}_pca{}_minus.json'.format('model2', ind, seed, pca), 'r'))
        model1 = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **params1)
        model1.load_model(r'./seed{}pca{}_{}_pass_minus_{}.model'.format(seed, pca, 'model1', ind))
        model1s.append(model1)
        model2 = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **params2)
        model2.load_model(r'./seed{}pca{}_{}_pass_minus_{}.model'.format(seed, pca, 'model2', ind))
        model2s.append(model2)

    return model1s, model2s


aa = [380+i*5 for i in range(81)]
important_inds = [0, 3, 6, 10, 15, 19, 24, 29, 34, 44, 54, 64, 74, 80]
def cross_val(X, y, ind, seed, pca, ff_train, param_file, model_flag=None):

    parameters = json.load(open(param_file, 'r'))

    ml_model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **parameters)
    ml_model.fit(X, y)
    save_model(ml_model, r'./seed{}pca{}_{}_pass_minus_{}.model'.format(seed, pca, model_flag, ind))

    # 交叉验证模型是否过拟合
    train_mse, test_mse = 0, 0
    for i in range(10):
        x_train, x_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=i*66)
        train_mse += mean_squared_error(ml_model.predict(x_train), y_train)
        test_mse += mean_squared_error(ml_model.predict(x_test), y_test)
    line = "seed{}, pca{}, 频段: {} 交叉验证, {}_train_mse: {}, test_mse: {}".format(seed, pca, aa[important_inds[ind]], model_flag, train_mse/10, test_mse/10)
    print(line)
    ff_train.write(line+'\n')

    return ml_model


def eval_metric(diff):
    weights = [1] * 14
    weights[2], weights[9], weights[4], weights[12], weights[13] = 2,2,3,3,3
    metric = [diff[i] * weights[i] for i in range(14)]

    return np.mean(metric)

