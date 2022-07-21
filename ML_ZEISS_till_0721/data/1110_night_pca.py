# coding=utf-8
'''
模拟蔡司日常生产使用模式, 训测数据无交集,
'''

import os
import matplotlib.pyplot as plt
from function_thick2lab import function_thick2lab
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import numpy as np
from scipy.stats import uniform, randint
from lstm_sensor_embedding import train_data_lstm_feature, test_data_lstm_feature
from flaml import tune
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


def slim_pre_sensor_feature(pre_sensor_feature, remove_inds_txt):
    # pre_sensor_feature = np.load('./data5_test_tf_lstm_feature.npy')
    # train test remove掉的冗余特征需要保持一致, 不然model参数维度对不上..
    mean_ = np.mean(pre_sensor_feature, axis=0)
    zero_inds = np.argwhere(mean_ == 0)
    zero_inds = [a[0] for a in zero_inds]
    nan_inds = np.argwhere(np.isnan(mean_))
    nan_inds = [a[0] for a in nan_inds]
    remove_inds = zero_inds + nan_inds
    f = open(remove_inds_txt, 'w')
    for remove_ind in remove_inds:
        f.write(str(remove_ind) + ',')
    f.close()
    pre_sensor_feature = pre_sensor_feature.tolist()
    ll = len(pre_sensor_feature[0])
    pre_sensor_feature = [[a[r] for r in range(ll) if r not in remove_inds] for a in pre_sensor_feature]
    # pre_sensor_feature = np.array(pre_sensor_feature)
    # print(pre_sensor_feature.shape)

    return pre_sensor_feature


def report_best_scores(results, ind, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            parameter = results['params'][candidate]

    data = json.dumps(parameter)
    with open(r'./parameter_{}_tf.json'.format(ind), 'w') as js_file:
        js_file.write(data)


def xgb_hyperparameter_searching(X, Y, ind):
    ml_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.8, 0.1),
        "gamma": uniform(0, 1),
        "learning_rate": uniform(0.001, 0.1),
        "max_depth": randint(3, 5),
        "n_estimators": randint(10, 30),
        "subsample": uniform(0.6, 0.4),
        "reg_lambda": uniform(1, 0.8),
    }

    search = RandomizedSearchCV(ml_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, ind, 5)


def save_model(model, model_path):
    model.save_model(model_path)


def cross_val(model_flag, X, y, ind, important_inds):
    if model_flag == 0:
        parameters = json.load(open(r'./parameter_{}_tf.json'.format(ind), 'r'))
        ml_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
        ml_model.fit(X, y)
        save_model(ml_model, r'./ml_{}.model'.format(ind))

    elif model_flag == 1:
        # lightgbm
        params = {'n_estimators': 110, 'learning_rate': 0.21550071029494922, 'num_leaves': 12, 'max_depth': 6,
                  'min_data_in_leaf': 50, 'max_bin': 181, 'lambda_l2': 100, 'min_gain_to_split': 10.627711503070213,
                  'bagging_fraction': 0.4, 'bagging_freq': 1, 'feature_fraction': 0.8}
        ml_model = lgb.LGBMRegressor(objective='regression', **params)
        ml_model.fit(X, y)
        joblib.dump(ml_model, r'./ml_{}.pkl'.format(ind))
    else:
        # catboost
        ml_model = []
        pass

    # train eval()
    ok, ng = 0, 0
    x_test = X.tolist()
    y_test = y.tolist()
    for iind in range(len(x_test)):
        pre_test_data = ml_model.predict(np.array([x_test[iind]]))
        if (y_test[iind][0] * pre_test_data[0]) > 0:
            ok += 1
        else:
            ng += 1
    print("train data, 频段{}, deta趋势一致: {}, 趋势不一致: {}, all data size: {}".format(important_inds[ind] * 5 + 380, ok, ng, ok + ng))

    return ml_model


def strip_a(a):
    a = a.strip("['] ")
    return a


def slim_thickness(thick):
    res = []
    inds = [0, 2, 3, 4, 5]
    for ind in inds:
        res.append(thick[ind])

    return res


def scale_deta_thickness(base_thick, cur_thick):

    inds = [0, 2, 3, 4, 5]
    base_cur_diff = [cur_thick[ind] - base_thick[ind] for ind in inds]

    base_cur_diff = [a * 1 for a in base_cur_diff]
    base_cur_diff = [base_cur_diff[0]] + [0] + base_cur_diff[1:] + [0]
    modified_thick = [base_cur_diff[k] + base_thick[k] for k in range(7)]

    return modified_thick


def samex_samey(X, Y):
    x_y = dict()
    for ind, x in enumerate(X):
        key = ''.join(str(np.round(a, 3)) for a in x)
        if key not in x_y:
            x_y[key] = []
        x_y[key].append(Y[ind])
    for k, v in x_y.items():
        if len(v) != 1:
            for v_ in v:
                print(v_)
            print('---')


def generate_data(datas, important_inds, x_data_file, y_data_file, Ovens_lstm_feature):
    # if os.path.exists(x_data_file):
    #     return np.load(x_data_file), np.load(y_data_file)
    # else:
    X = []
    Y = []
    ws = [1]*13
    # ws[10], ws[11], ws[12] = 1.1, 1.1, 1.1
    for data in datas:
        ll = len(data)
        for j in range(ll - 1):
            sensor_file_name1, sensor_file_name2 = data.iloc[j]['FileID'][3:], data.iloc[j + 1]['FileID'][3:]
            # 控制炉号连续, diff<=3
            if int(sensor_file_name2) - int(sensor_file_name1) <= 3:
                pre_oven = data.iloc[j]['OvenNo']
                pre_sensor = Ovens_lstm_feature[pre_oven]
                # pre_thickness
                pre_thickness = [float(strip_a(a)) for a in data.iloc[j]['Thickness'].split(',')]
                # cur_thickness
                cur_thickness = [float(strip_a(a)) for a in data.iloc[j + 1]['Thickness'].split(',')]
                scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness)
                cur_function = function_thick2lab(scaled_thickness)
                x = pre_sensor + slim_thickness(cur_thickness)
                assert len(x) == len(pre_sensor) + 5
                cur_lab = [float(strip_a(a)) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in
                                    important_inds]
                # 手动把700 750 780 处的deta放大,
                y_important_inds = [y_important_inds[k]*ws[k] for k in range(13)]
                X.append(x)
                Y.append(y_important_inds)

    # check x and y
    samex_samey(X, Y)

    # 统计x的均值方差
    mean_ = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)

    std_ = [a + 1e-6 for a in std_]
    ff = open('./X_mean_std_train.txt', 'w')
    ff.write(''.join(str(a) + ',' for a in mean_))
    ff.write('\n')
    ff.write(''.join(str(a) + ',' for a in std_))
    ff.close()
    print("base x: {}".format(X[0]))
    X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
    print("normalized x: {}".format(X[0]))

    X, Y = np.array(X), np.array(Y)
    np.save(x_data_file, X)
    np.save(y_data_file, Y)

    return X, Y


def slim_test_data(data):
    ll = len(data)
    bads = []
    for i in range(ll):
        path = data.iloc[i]['path_sensor_data']
        try:
            _ = pd.read_csv(path)
        except:
            bads.append(i)
            continue
    for bad in bads:
        data = data.drop([bad])

    return data

def generate_test_data(datas, important_inds, x_data_file, y_data_file, test_sensor_embedding_feature, test_Ovens):
    # if os.path.exists(x_data_file):
    #     return np.load(x_data_file), np.load(y_data_file), json.load(open(r'./lstm_and_thick_2_Oven.json', 'r')), json.load(open(r'./Oven_lab.json', 'r')), np.load(r'./scaled_thickness.npy')
    # else:
    lstm_and_thick_2_Oven = dict()
    Oven_lab = dict()
    X = []
    Y = []
    scaled_thicknesses = []
    for data in datas:
        data = slim_test_data(data)
        ll = len(data)
        for j in range(ll-1):
            pre_sensor = test_sensor_embedding_feature[j]
            # pre_thickness
            pre_thickness = [float(strip_a(a)) for a in data.iloc[j]['Thickness'].split(',')]
            # cur_thickness
            cur_thickness = [float(strip_a(a)) for a in data.iloc[j + 1]['Thickness'].split(',')]
            scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness)
            scaled_thicknesses.append(scaled_thickness)
            cur_function = function_thick2lab(scaled_thickness)
            x = pre_sensor + slim_thickness(cur_thickness)
            cur_lab = [float(strip_a(a)) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
            cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
            Oven_lab[str(data.iloc[j + 1]['OvenNo'])] = cur_lab
            y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in
                                important_inds]
            X.append(x)
            Y.append(y_important_inds)

    # 统计x的均值方差
    mean_ = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)

    #2. std_清零处理
    std_ = [a+1e-6 for a in std_]
    ff = open('./X_mean_std_test.txt', 'w')
    ff.write(''.join(str(a) + ',' for a in mean_))
    ff.write('\n')
    ff.write(''.join(str(a) + ',' for a in std_))
    ff.close()
    print("base x: {}".format(X[0]))
    X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
    X, Y = np.array(X), np.array(Y)
    X[np.isnan(X)] = 0
    print("normalized x: {}".format(X[0]))

    for j, x in enumerate(X):
        lstm_and_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in x)] = test_Ovens[j]
    data = json.dumps(lstm_and_thick_2_Oven)
    with open(r'./lstm_and_thick_2_Oven.json', 'w') as js_file:
        js_file.write(data)

    # Oven_lab
    data = json.dumps(Oven_lab)
    with open(r'./Oven_lab.json', 'w') as js_file:
        js_file.write(data)

    np.save(x_data_file, X)
    np.save(y_data_file, Y)
    scaled_thicknesses = np.array(scaled_thicknesses)
    np.save(r'./scaled_thickness.npy', np.save(x_data_file, X))

    return X, lstm_and_thick_2_Oven, Oven_lab, scaled_thicknesses


def overfiting(X, Y, ind):
    dfull = xgb.DMatrix(X, Y)

    param1 = json.load(open(r'./parameter_{}_tf.json'.format(ind), 'r'))
    num_round = 200

    cvresult1 = xgb.cv(param1, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()


def bad_Ovens(datas):
    bads = []
    c = 0
    for data in datas:
        ll = len(data)
        for i in range(ll - 1):
            OvenNo = data.iloc[i]['OvenNo']
            path = data.iloc[i]['path_sensor_data']
            c += 1
            try:
                sensor_data = pd.read_csv(path)
            except:
                bads.append(OvenNo)
                continue
    print(c)
    return bads


def search_thickness(model_flag, x_test, cx_lab_curve, slim_best_cx_curve, important_inds, save_dir):
    lstm_and_thick_2_Oven_data = json.load(open(r'./lstm_and_thick_2_Oven.json', 'r'))

    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ml_models = []
    for i in range(len(important_inds)):
        params = json.load(open(r'./parameter_{}_tf.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        if model_flag == 0:
            model.load_model(r'./ml_{}.model'.format(i))
        elif model_flag == 1:
            model = joblib.load(r'./ml_{}.pkl'.format(i))
        else:
            pass
        ml_models.append(model)

    x_test = x_test.tolist()
    ff = open('./X_mean_std_test.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test_no_normalize = [[test[r] * std_[r] + mean_[r] for r in range(len(mean_))] for test in x_test]

    for iindex, single_test_data in enumerate(x_test):
        oven = lstm_and_thick_2_Oven_data[''.join(str(np.round(a, 3)) + ',' for a in x_test[iindex])]
        try:
            real_curve_slim = [Oven_lab[str(oven)][r] for r in important_inds]
        except:
            continue
        preds_dim11 = []
        for ml_model in ml_models:
            preds_dim11.append(ml_model.predict(np.array([single_test_data]))[0])
        assert len(preds_dim11) == len(important_inds)

        embedding_sensor_feature_dim = len(std_) - 5
        pre_thickness = [x_test_no_normalize[iindex][embedding_sensor_feature_dim:][0]] + [4] + x_test_no_normalize[iindex][embedding_sensor_feature_dim + 1:] + [35]
        finetune_thre1 = [0.1, 0.1, 0.4, 0.8, 1]

        def get_y1(thickness_list):
            recommend_thickness = [thickness_list[0]] + [4] + thickness_list[1:] + [35]
            scaled_thickness = scale_deta_thickness(pre_thickness, recommend_thickness)
            cur_function = function_thick2lab(scaled_thickness)
            deta_function_and_best = [cur_function[important_ind] - cx_lab_curve[important_ind] for important_ind in
                                      important_inds]

            return deta_function_and_best

        def weighted_metric(a, b, weights):
            ll = len(a)
            diff = [(a[i] - b[i]) * (a[i] - b[i]) for i in range(ll)]
            diff = [diff[i] * weights[i] for i in range(ll)]

            return np.mean(diff)

        def evaluate_config(config):
            y1 = get_y1([config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
            # weights = [0.05, 0.6, 1.3, 1,1,1.5,1.3,1.5, 1,1,1,1.3,0.3]   # 1109_tf_li_gong_review_0
            # weights = [0.05, 0.6, 1.6, 1.8,1.5,1.9,1.5,1.5, 1,1.1,1.1,1.1,0.1]   # 1109_tf_li_gong_review_0_1
            # weights = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
            weights = [0.05]*13
            weights[3] = 1
            weights[11] = 1
            metric = weighted_metric(y1, preds_dim11, weights)
            # metric = mean_squared_error(y1, preds_dim11)

            # use tune.report to report the metric to optimize
            tune.report(metric=metric)

        analysis = tune.run(
            evaluate_config,  # the function to evaluate a config
            config={
                'x1': tune.uniform(lower=pre_thickness[0] - finetune_thre1[0],
                                   upper=pre_thickness[0] + finetune_thre1[0]),
                'x2': tune.uniform(lower=pre_thickness[2] - finetune_thre1[1],
                                   upper=pre_thickness[2] + finetune_thre1[1]),
                'x3': tune.uniform(lower=pre_thickness[3] - finetune_thre1[2],
                                   upper=pre_thickness[3] + finetune_thre1[2]),
                'x4': tune.uniform(lower=pre_thickness[4] - finetune_thre1[3],
                                   upper=pre_thickness[4] + finetune_thre1[3]),
                'x5': tune.uniform(lower=pre_thickness[5] - finetune_thre1[4],
                                   upper=pre_thickness[5] + finetune_thre1[4]),
            },  # the search space
            low_cost_partial_config={'x1': pre_thickness[0],
                                     'x2': pre_thickness[2],
                                     'x3': pre_thickness[3],
                                     'x4': pre_thickness[4],
                                     'x5': pre_thickness[5]},  # a initial (partial) config with low cost
            metric='metric',  # the name of the metric used for optimization
            mode='min',  # the optimization mode, 'min' or 'max'
            num_samples=-1,  # the maximal number of configs to try, -1 means infinite
            time_budget_s=60,  # the time budget in seconds 优化时间限制, 60s停止优化
            local_dir='logs/',  # the local directory to store logs
        )

        search_res = analysis.best_trial.last_result
        thickness_res = [np.round(search_res['config']['x1'], 2), 4, np.round(search_res['config']['x2'], 2),
                         np.round(search_res['config']['x3'], 2), np.round(search_res['config']['x4'], 2),
                         np.round(search_res['config']['x5'], 2), 35]
        plt.plot(aa, real_curve_slim, color='pink',
                 label='cur_curve: {}'.format(''.join(str(a) + ', ' for a in pre_thickness)))
        plt.plot(aa, slim_best_cx_curve, color='cornflowerblue',
                 label='to get best_curve: {}'.format(''.join(str(a) + ', ' for a in thickness_res)))
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(oven)))
        # plt.show()
        plt.close()


def compare_pred_realCurve(x_test, scaled_thickness, lstm_and_thick_2_Oven, Oven_lab, save_dir):
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ll = len(aa)
    xgb_models = []
    for i in range(ll):
        params = json.load(open(r'./parameter_{}_tf.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./ml_{}.model'.format(i))
        xgb_models.append(model)
    x_test = x_test.tolist()

    test_real = []
    test_preds = []
    for iindex, single_test_data in enumerate(x_test):
        single_preds = []
        oven = lstm_and_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in x_test[iindex])]
        try:
            real_curve_slim = [Oven_lab[str(oven)][r] for r in important_inds]
        except:
            continue

        for xgb_model in xgb_models:
            single_pred = xgb_model.predict(np.array([single_test_data]))
            single_preds.append(single_pred)
        single_f = function_thick2lab(scaled_thickness[iindex])
        single_f = [single_f[i] for i in important_inds]
        pred_m = [single_f[i] - single_preds[i] for i in range(len(important_inds))]
        test_real.append(real_curve_slim)
        test_preds.append(pred_m)
        plt.plot(aa, real_curve_slim, label='real_curve', color='pink')
        plt.plot(aa, pred_m, label='pred_curve', color='cornflowerblue')
        plt.legend()
        plt.savefig(os.path.join(save_dir, './{}.png'.format(oven)))
        plt.close()
    diff = [[abs(test_real[i][k] - test_preds[i][k]) for k in range(ll)] for i in range(len(test_real))]
    diff_13 = np.mean(diff, axis=0).tolist()
    print("data5, pred and real curve diff: {}".format(diff_13))


def curve_trend_check(x_test, lstm_and_thick_2_Oven, scaled_thickness):
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ll = len(aa)
    xgb_models = []
    ff = open('./X_mean_std_test.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    for i in range(ll):
        params = json.load(open(r'./parameter_{}_tf.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./ml_{}.model'.format(i))
        xgb_models.append(model)
    x_test = x_test.tolist()
    x_test_no_normalize = [[test[r] * std_[r] + mean_[r] for r in range(len(mean_))] for test in x_test]
    x_lens = len(x_test_no_normalize[0])

    # modified = [0, 0, 0, 0, 0.6, 1, 0]
    # all_modified = [0]*(x_lens-5) + [0, 0, 0, 0.6, 1]
    modified = [0, 0, 0, 0, 1, -1, 0]
    all_modified = [0] * (x_lens - 5) + [0, 0, 0, 1, -1]

    # x: pre_sensor_feature + cur_thickness -> deta(f, m)
    for iindex, single_test_data in enumerate(x_test):
        # de_normalized_x:
        de_single_test_data = x_test_no_normalize[iindex]
        # print(de_single_test_data[-10:], '1')
        de_single_test_data = [de_single_test_data[r]+all_modified[r] for r in range(x_lens)]
        # print(de_single_test_data[-10:], '2')
        de_single_test_data = [(de_single_test_data[k]-mean_[k])/std_[k] for k in range(x_lens)]
        single_preds = []
        modified_single_preds = []
        oven = lstm_and_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in x_test[iindex])]
        for xgb_model in xgb_models:
            single_pred = xgb_model.predict(np.array([single_test_data]))
            single_preds.append(single_pred)
            modified_single_pred = xgb_model.predict(np.array([de_single_test_data]))
            modified_single_preds.append(modified_single_pred)
        single_f = function_thick2lab(scaled_thickness[iindex])
        single_f = [single_f[i] for i in important_inds]
        pred_m = [single_f[i] - single_preds[i] for i in range(len(important_inds))]
        # 手动调整膜厚后的模型预测曲线
        modified_thickness = [scaled_thickness[iindex][r] + modified[r] for r in range(7)]
        single_f_modified = function_thick2lab(modified_thickness)
        single_f_modified = [single_f_modified[i] for i in important_inds]
        pred_m_modified = [single_f_modified[i] - modified_single_preds[i] for i in range(len(important_inds))]
        # show curve
        plt.plot(aa, pred_m, label='model_predicate', color='cornflowerblue')
        plt.plot(aa, pred_m_modified, label='layer5 +1, layer6 -1 model_predicate', color='pink')
        plt.legend()
        plt.savefig(os.path.join(save_dir, './{}.png'.format(oven)))
        plt.close()


# if __name__ == '__main__':

def fun():
    sensor_cols = ["ACT_V1_THERMOVAC_PREVLINE", "ACT_V1_THERMOVAC_CH", "ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH",
                   "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP",
                   "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE","ACT_Q10_CURRENT_ANODE",
                   "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL",
                   "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "ACT_O1_QCMS_RATE", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'

    data5 = r'./1105_cx_data.csv'
    train_data = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4)]
    test_data = [pd.read_csv(data5)]

    important_inds = [0, 3, 6, 14, 19, 24, 29, 34, 44, 54, 64, 74, 80]
    x_data_file = r'./X_tmp.npy'
    y_data_file = r'./Y_tmp.npy'
    # remove_inds_txt = r'./pre_sensor_feature_remove_inds.txt'

    #1. train model
    # load train lstm_sensor_feature
    # sensor_embedding_feature, train_Ovens = train_data_lstm_feature(train_data, sensor_cols)
    sensor_embedding_feature, train_Ovens = np.load(r'./all_tf_sensor_features.npy'), np.load(r"./Ovens.npy").tolist()
    print("before PCA, train data sensor feature dim: {}".format(sensor_embedding_feature.shape[1]))
    # sensor_embedding_feature = slim_pre_sensor_feature(sensor_embedding_feature, remove_inds_txt)
    sensor_embedding_feature[np.isnan(sensor_embedding_feature)] = 0
    pca_1 = PCA(n_components=50)
    sensor_embedding_feature = pca_1.fit_transform(sensor_embedding_feature).tolist()
    print("after PCA, train data sensor feature dim: {}".format(len(sensor_embedding_feature[0])))
    data_size = len(sensor_embedding_feature)
    assert data_size == len(train_Ovens)
    print("train data size: {}".format(data_size))
    Ovens_lstm_feature = dict()
    for ind, oven in enumerate(train_Ovens):
        Ovens_lstm_feature[oven] = sensor_embedding_feature[ind]
    X, Y = generate_data(train_data, important_inds, x_data_file, y_data_file, Ovens_lstm_feature)
    Y = Y.tolist()
    # ['xgboost', 'lightgbm', 'catboost']
    model_flag = 0

    ml_models = []
    y_tests = []
    y_trains = []
    for ind, important_ind in enumerate(important_inds):
        y = np.array([[Y_[ind]] for Y_ in Y])
        # 'xgboost', 'lightgbm', 'catboost' 的参数搜索
        if model_flag == 0:
            # xgb_hyperparameter_searching(X, y, ind)
            # overfiting(X, Y, ind)
            pass
        elif model_flag == 1:
            # lightgbm
            # lightgbm_hyperparameter_searching(X, y, ind)
            pass
        else:
            # catboost
            pass

        ml_model = cross_val(model_flag, X, y, ind, important_inds)
        ml_models.append(ml_model)


    # 2. model_pred and real_cuvre check
    # load test lstm_sensor_feature
    # test_sensor_embedding_feature, test_Ovens = test_data_lstm_feature(test_data, sensor_cols)
    test_sensor_embedding_feature, test_Ovens = np.load('./data5_test_tf_lstm_feature.npy'), np.load('./test_Ovens.npy').tolist()
    # remove_inds = [int(a) for a in open(remove_inds_txt, 'r').readlines()[0].split(',')[:-1]]
    ll_tmp = test_sensor_embedding_feature.shape[1]
    # test_sensor_embedding_feature = [[a[r] for r in range(ll_tmp) if r not in remove_inds] for a in test_sensor_embedding_feature]
    test_sensor_embedding_feature[np.isnan(test_sensor_embedding_feature)] = 0
    test_sensor_embedding_feature = pca_1.fit_transform(test_sensor_embedding_feature).tolist()
    print(ll_tmp, len(test_sensor_embedding_feature[0]))

    save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\1110\data5_pca'
    # generate test_data
    x_data_file_test, y_data_file_test = r'./X_tmp_test.npy', r'./X_tmp_test.npy'
    x_test, lstm_and_thick_2_Oven, Oven_lab, scaled_thickness = generate_test_data(test_data, important_inds, x_data_file_test, y_data_file_test, test_sensor_embedding_feature, test_Ovens)
    compare_pred_realCurve(x_test, scaled_thickness, lstm_and_thick_2_Oven, Oven_lab, save_dir)

# 20220517
# tsf_test_feature.json, tsf_train_feature.json





