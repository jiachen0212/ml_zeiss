# coding=utf-8
import os
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import json
import numpy as np
from sklearn.model_selection import train_test_split as TTS
from scipy.stats import uniform, randint
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from flaml import tune
from function_thick2lab import function_thick2lab


def pre_data_process(all_data):
    yichang_ovens = ["33121090910", "33121110302", "33321091006", "33321090806", "33121090508"]
    ok_ovens = []
    for data in all_data:
        ll = len(data)
        for i in range(ll):
            OvenNo = str(data.iloc[i]['OvenNo'])
            path = data.iloc[i]['path_sensor_data']
            try:
                _ = pd.read_csv(path)
                if OvenNo not in yichang_ovens:
                    ok_ovens.append(OvenNo)
            except:
                continue
    np.save("./ok_ovens.npy", np.array(ok_ovens))

    return ok_ovens


def scale_deta_thickness(base_thick, cur_thick):
    inds = [0, 2, 3, 4, 5]
    base_cur_diff = [cur_thick[ind] - base_thick[ind] for ind in inds]

    base_cur_diff = [a * 1 for a in base_cur_diff]
    base_cur_diff = [base_cur_diff[0]] + [0] + base_cur_diff[1:] + [0]
    modified_thick = [base_cur_diff[k] + base_thick[k] for k in range(7)]

    return modified_thick


def slim_thickness(thick):
    res = []
    inds = [0, 2, 3, 4, 5]
    for ind in inds:
        res.append(thick[ind])

    return res


def generate_data(datas, important_inds, x_data_file, y_data_file, ovens_lstm_tsf_ender_feature):
    if os.path.exists(x_data_file):
        return np.load(x_data_file), np.load(y_data_file)
    else:
        print("generate train .npy ~ ")
        X = []
        Y = []
        ws = [1] * len(important_inds)
        for data in datas:
            ll = len(data)
            for j in range(ll - 1):
                sensor_file_name1, sensor_file_name2 = data.iloc[j]['FileID'][3:], data.iloc[j + 1]['FileID'][3:]
                tmp2 = int(sensor_file_name2[-3:])
                tmp1 = int(sensor_file_name1[-3:])
                if tmp2 - tmp1 <= 3:
                    pre_oven = str(data.iloc[j]['OvenNo'])
                    try:
                        pre_sensor = ovens_lstm_tsf_ender_feature[pre_oven]
                    except:
                        continue
                    pre_thickness = [float(a.strip("['] ")) for a in data.iloc[j]['Thickness'].split(',')]
                    cur_thickness = [float(a.strip("['] ")) for a in data.iloc[j + 1]['Thickness'].split(',')]
                    scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness)
                    cur_function = function_thick2lab(scaled_thickness)
                    x = pre_sensor + slim_thickness(cur_thickness)
                    assert len(x) == len(pre_sensor) + 5
                    cur_lab = [float(a.strip("['] ")) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                    cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                    # lab曲线可能因为误统计出现负值, 置零处理..
                    cur_lab = [max(0, a) for a in cur_lab]
                    y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in
                                        important_inds]
                    X.append(x)
                    y_important_inds = [y_important_inds[k] * ws[k] for k in range(len(important_inds))]
                    Y.append(y_important_inds)

        mean_ = np.mean(X, axis=0)
        std_ = np.std(X, axis=0)
        std_ = [a + 1e-6 for a in std_]
        ff = open('./X_mean_std_train.txt', 'w')
        ff.write(''.join(str(a) + ',' for a in mean_))
        ff.write('\n')
        ff.write(''.join(str(a) + ',' for a in std_))
        ff.close()
        X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]

        X, Y = np.array(X), np.array(Y)
        np.save(x_data_file, X)
        np.save(y_data_file, Y)

        return X, Y


def tmp_generate_test_data(pca, seed, datas, important_inds, x_data_file, y_data_file, ovens_lstm_tsf_ender_feature):
    if os.path.exists(x_data_file):
        return np.load(x_data_file), np.load(y_data_file), np.load(r'./tmp{}{}.npy'.format(seed, pca))
    else:
        X = []
        Y = []
        pre_ovens = []
        ws = [1] * len(important_inds)
        for data in datas:
            # 这里其实是data6
            ll = len(data)
            for j in range(ll - 1):
                sensor_file_name1, sensor_file_name2 = data.iloc[j]['FileID'][3:], data.iloc[j + 1]['FileID'][3:]
                tmp2 = int(sensor_file_name2[-3:])
                tmp1 = int(sensor_file_name1[-3:])
                pre_oven = data.iloc[j]['OvenNo']
                if tmp2 - tmp1 <= 3:
                    try:
                        pre_sensor = ovens_lstm_tsf_ender_feature[str(pre_oven)]
                        pre_ovens.append(str(pre_oven))
                    except:
                        continue
                    pre_thickness = [float(a.strip("['] ")) for a in data.iloc[j]['Thickness'].split(',')]
                    cur_thickness = [float(a.strip("['] ")) for a in data.iloc[j + 1]['Thickness'].split(',')]
                    scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness)
                    cur_function = function_thick2lab(scaled_thickness)
                    x = pre_sensor + slim_thickness(cur_thickness)
                    # assert len(x) == len(pre_sensor) + 5
                    cur_lab = [float(a.strip("['] ")) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                    cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                    cur_lab = [max(0, a) for a in cur_lab]
                    y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in
                                        important_inds]
                    X.append(x)
                    y_important_inds = [y_important_inds[k] * ws[k] for k in range(len(important_inds))]
                    Y.append(y_important_inds)

        mean_, std_ = [],[]
        ff = open('./X_mean_std_train.txt', 'r').readlines()
        for ind, line in enumerate(ff):
            if ind == 0:
                mean_ = line.split(',')[:-1]
                print(len(mean_))
                mean_ = [float(a) for a in mean_]
            if ind == 1:
                std_ = [float(a) for a in line.split(',')[:-1]]

        X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
        X, Y = np.array(X), np.array(Y)
        np.save(x_data_file, X)
        np.save(y_data_file, Y)
        np.save('./tmp{}{}.npy'.format(seed, pca), np.array(pre_ovens))
        
        return X, Y, pre_ovens


def pca_tsf_feature(tsf_feature_js, pca_1, flag=None):
    vs = []
    for k, v in tsf_feature_js.items():
        vs.append(v)
    vs = np.array(vs)
    vs[np.isnan(vs)] = 0
    if not flag:
        return vs.tolist()
    else:
        pca_tsf_feature = pca_1.fit_transform(vs).tolist()
        return pca_tsf_feature


def report_best_scores(results, ind, seed, pca, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            parameter = results['params'][candidate]

    data = json.dumps(parameter)
    with open(r'./parameter_{}_seed{}_pca{}.json'.format(ind, seed, pca), 'w') as js_file:
        js_file.write(data)


def xgb_hyperparameter_searching(X, Y, ind, seed, pca):
    ml_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.1, 0.9),
        "gamma": uniform(0, 1),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 9),
        "n_estimators": randint(10, 120),
        "subsample": uniform(0.4, 0.6),
        "reg_lambda": uniform(0.1, 5),
    }
    search = RandomizedSearchCV(ml_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=-1, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, ind, seed, pca, 5)


def save_model(model, model_path):
    model.save_model(model_path)


def show_mse(ind_info, x, y, ml_model, dim_mse, rate, flag=None):
    ok, ng = 0, 0
    x_test = x.tolist()
    # pred_ok, pred_ng = [], []
    # gt_ok, gt_ng = [], []
    mse = mean_squared_error(y, ml_model.predict(x))
    # all data mse
    # if mse >= 1:
    dim_mse[ind_info] = mse
    y_test = y.tolist()
    for iind, x in enumerate(x_test):
        pre_test_data = ml_model.predict(np.array([x]))
        if (y_test[iind][0] * pre_test_data[0]) > 0:
            ok += 1
        else:
            ng += 1
    print("{} data, 频段{}, rate_value: {}, mse: {}, 趋势不一致率: {}".format(flag, ind_info, rate, mse, ng / (ok + ng)))


def cross_val(model_flag, X, y, X_test, test_y, ind, important_inds, dim_mse, dim_mse_):
    ind_info = important_inds[ind] * 5 + 380
    slim_best_curve = [6.767054736842118, 1.3183033198380567, 0.26796543441079645, 0.7729350602956238,
                       1.0927496373593792,
                       0.9318485721707224, 0.6553284981271605, 0.40983069562010727, 0.24249118621313095,
                       0.037322223906119174,
                       0.13442843534172752, 0.7438048388143235, 2.0543106629825285, 3.1030965247677997]
    rate = np.round(slim_best_curve[ind], 2)

    if model_flag == 0:
        parameters = json.load(open(r'./parameter_{}_seed{}_pca{}.json'.format(ind, seed, pca), 'r'))
        ml_model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **parameters)
        ml_model.fit(X, y)
        save_model(ml_model, r'./qyicktest_ml_{}.model'.format(ind))

    # show feature importance
    # fs = ml_model.feature_importances_
    # ll = len(fs)
    # no_zero_f = [ind for ind in range(ll) if fs[ind] > 0]
    # if ind in bad_inds:
    #     print(no_zero_f)
    #     dims = [i*5+380 for i in range(81)]
    #     plt.title('feature importances: {}'.format(dims[important_inds[ind]]))
    #     plt.bar(range(len(fs)), fs)
    #     plt.show()
    show_mse(ind_info, X, y, ml_model, dim_mse, rate, flag='train')
    show_mse(ind_info, X_test, test_y, ml_model, dim_mse_, rate, flag='test')

    return ml_model


def train(seed, pca, model_flag, x_train, y_train, x_test, y_test, important_inds, all_data,
          train_feature, test_feature):
    '''
    train model

    '''

    X, Y = generate_data(all_data, important_inds, x_train, y_train, train_feature)

    X_test, Y_test, test_oven = tmp_generate_test_data(pca, seed, [all_data[-1]], important_inds, x_test, y_test, test_feature)

    print("train X size: {}".format(X.shape))
    print("test X size: {}".format(X_test.shape))
    Y = Y.tolist()
    Y_test = Y_test.tolist()
    ml_models = []
    dim_mse = dict()
    dim_mse_ = dict()

    for ind, important_ind in enumerate(important_inds):
        y = np.array([[Y_[ind]] for Y_ in Y])
        test_y = np.array([[Y_[ind]] for Y_ in Y_test])
        if model_flag == 0:
            if ind != 12:
                # xgb_hyperparameter_searching(X, y, ind, seed, pca)
                # overfiting(X, Y, ind)
                pass
        elif model_flag == 1:
            # lightgbm
            # lightgbm_hyperparameter_searching(X, y, ind)
            pass
        else:
            # catboost
            pass
        ml_model = cross_val(model_flag, X, y, X_test, test_y, ind, important_inds, dim_mse, dim_mse_)
        # print("train mse: {}".format(dim_mse))
        # print("test mse: {}".format(dim_mse_))
        ml_models.append(ml_model)

    return ml_models, test_oven


def generate_test_data(datas, test_oven, important_inds, test_oven_allfeature):
    print("generate test data ~ ")
    pre_oven_x = dict()
    pre_cur_oven = dict()
    lstm_tsf_ender_thick_2_Oven = dict()
    Oven_lab = dict()
    X = []
    Y = []
    cur_oven_thick = dict()
    test_oven_normalize_feature = dict()
    scaled_thicknesses = []
    for i, data in enumerate(datas):
        ll = len(data)
        for j in range(ll - 1):
            pre_oven = str(data.iloc[j]['OvenNo'])
            if pre_oven in test_oven:
                try:
                    pre_sensor = test_oven_allfeature[pre_oven]
                except:
                    continue
                pre_thickness = [float(a.strip("['] ")) for a in data.iloc[j]['Thickness'].split(',')]
                cur_thickness = [float(a.strip("['] ")) for a in data.iloc[j + 1]['Thickness'].split(',')]
                scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness)
                scaled_thicknesses.append(scaled_thickness)
                cur_function = function_thick2lab(scaled_thickness)
                x = pre_sensor + slim_thickness(cur_thickness)
                cur_lab = [float(a.strip("['] ")) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                cur_lab = [max(a, 0) for a in cur_lab]
                cur_oven = str(data.iloc[j + 1]['OvenNo'])
                cur_oven_thick[cur_oven] = cur_thickness
                Oven_lab[cur_oven] = cur_lab
                y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in
                                    important_inds]
                X.append(x)
                Y.append(y_important_inds)
                pre_oven_x[pre_oven] = x
                pre_cur_oven[pre_oven] = cur_oven

    ff = open('./X_mean_std_train.txt', 'r').readlines()
    for ind, line in enumerate(ff):
        if ind == 0:
            mean_ = [float(a) for a in line.split(',')[:-1]]
        if ind == 1:
            std_ = [float(a) for a in line.split(',')[:-1]]

    X_ = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
    X_, Y = np.array(X_), np.array(Y)
    X_[np.isnan(X_)] = 0
    tmp_X = X_.tolist()

    for oven in test_oven:
        try:
            x = pre_oven_x[oven]
            x_ind = X.index(x)
            x_nor = tmp_X[x_ind]
        except:
            continue
        lstm_tsf_ender_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in x_nor)] = oven

    assert len(lstm_tsf_ender_thick_2_Oven) == len(X)


    data = json.dumps(cur_oven_thick)
    with open(r'./cur_oven_thick{}{}.json'.format(seed, pca), 'w') as js_file:
        js_file.write(data)

    ks, vs = [], []
    for k, v in test_oven_allfeature.items():
        ks.append(k)
        vs.append(v)
    me_, st_ = mean_[:-5], std_[:-5]
    assert len(me_) == 300
    vs = [[(v[i]-me_[i])/st_[i] for i in range(300)] for v in vs]
    for ind, k in enumerate(ks):
        test_oven_normalize_feature[k] = vs[ind]
    data = json.dumps(test_oven_normalize_feature)
    with open(r'./test_normalize_feature{}{}.json'.format(seed, pca), 'w') as js_file:
        js_file.write(data)

    return X_, lstm_tsf_ender_thick_2_Oven, Oven_lab, scaled_thicknesses, pre_cur_oven, cur_oven_thick


def compare_pred_realCurve(seed, pca, pre_cur_oven, x_test, scaled_thickness, lstm_tsf_ender_thick_2_Oven, Oven_lab, save_dir):
    all_weighted_metric = []
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ll = len(aa)
    xgb_models = []
    for i in range(ll):
        params = json.load(open(r'./parameter_{}_seed{}_pca{}.json'.format(i, seed, pca), 'r'))
        model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **params)
        model.load_model(r'./qyicktest_ml_{}.model'.format(i))
        xgb_models.append(model)
    x_test = x_test.tolist()

    test_real = []
    test_preds = []
    Oven_pred_lab = dict()
    for iindex, single_test_data in enumerate(x_test):
        single_preds = []
        try:
            pre_oven = lstm_tsf_ender_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in single_test_data)]
        except:
            print("compare model and real: bad lstm_tsf_ender_thick_2_Oven")
            continue
        try:
            cur_oven = pre_cur_oven[pre_oven]
            real_curve_slim = [Oven_lab[cur_oven][r] for r in important_inds]
        except:
            print("compare model and real: bad pre_oven: {}".format(pre_oven))
            continue
        for xgb_model in xgb_models:
            single_pred = xgb_model.predict(np.array([single_test_data]))
            single_preds.append(single_pred)
        single_f = function_thick2lab(scaled_thickness[iindex])
        single_f = [single_f[i] for i in important_inds]
        pred_m = [single_f[i] - single_preds[i] for i in range(len(important_inds))]
        test_real.append(real_curve_slim)
        pred_m = [a.tolist()[0] for a in pred_m]
        diff = [pred_m[i] - real_curve_slim[i] for i in range(ll)]
        weighted_metric = np.round(eval_metric(diff), 2)
        all_weighted_metric.append(weighted_metric)
        test_preds.append(pred_m)
        Oven_pred_lab[cur_oven] = pred_m
        x_ticks = np.arange(380, 800, 50)
        y_ticks = np.arange(0, 10, 1)
        plt.plot(aa, real_curve_slim, label='real_curve', color='pink')
        plt.plot(aa, pred_m, label='pred_curve, weighted_mse: {}'.format(weighted_metric), color='cornflowerblue')
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, './{}.png'.format(cur_oven)))
        plt.close()
    diff = [[abs(test_real[i][k] - test_preds[i][k]) for k in range(ll)] for i in range(len(test_real))]
    diff = np.mean(diff, axis=0).tolist()
    print("pred and real diff: {}".format(diff))

    data = json.dumps(Oven_pred_lab)
    with open(r'./oven_pred_lab_seed{}_pca{}.json'.format(seed, pca), 'w') as js_file:
        js_file.write(data)
    abs_diff_list = [np.round(diff[i], 3) for i in range(14)]
    # return Oven_pred_lab, np.round(np.mean(all_weighted_metric), 3)
    return Oven_pred_lab, abs_diff_list


def forward_model(ff, seed, pca, test_oven, save_dir, all_data, test_feature_dict):
    '''
    模型前向 test

    '''

    X_test_data, lstm_tsf_ender_thick_2_Oven, Oven_real_lab, scaled_thickness, pre_cur_oven, cur_oven_thick = generate_test_data([all_data[-1]], test_oven,
                                                                                                   important_inds,
                                                                                                   test_feature_dict)
    oven_pred_lab, eval_metric = compare_pred_realCurve(seed, pca, pre_cur_oven, X_test_data, scaled_thickness, lstm_tsf_ender_thick_2_Oven,
                                           Oven_real_lab, save_dir)
    line = ''.join(str(a) + ', ' for a in eval_metric)
    print("test data 14dims_abs_diff: {}".format(line))

    ff.write("seed_{}_pca{}, 14dims_abs_diff: {}".format(seed, pca, line) + '\n')

    return X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness, oven_pred_lab, Oven_real_lab, pre_cur_oven


def search_thickness(pca, seed, model_flag, x_test, cx_lab_curve, slim_best_cx_curve, important_inds, save_dir,
                     lstm_tsf_ender_thick_2_Oven, pre_cur_oven, oven_real_lab, oven_pred_lab):
    '''
    search thickness

    '''

    cur_oven_thick = json.load(open(r'./cur_oven_thick{}{}.json'.format(seed, pca), 'r'))
    test_normalize_feature = json.load(open(r'./test_normalize_feature{}{}.json'.format(seed, pca), 'r'))

    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ml_models = []
    for i in range(len(important_inds)):
        params = json.load(open(r'./parameter_{}_seed{}_pca{}.json'.format(i, seed, pca), 'r'))
        model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **params)
        if model_flag == 0:
            model.load_model(r'./qyicktest_ml_{}.model'.format(i))
        elif model_flag == 1:
            pass
        else:
            pass
        ml_models.append(model)

    x_test = x_test.tolist()
    ff = open('./X_mean_std_train.txt', 'r').readlines()
    for ind, line in enumerate(ff):
        if ind == 0:
            mean_ = [float(a) for a in line.split(',')[:-1]]
        if ind == 1:
            std_ = [float(a) for a in line.split(',')[:-1]]

    for iindex, single_test_data in enumerate(x_test):
        try:
            pre_oven = lstm_tsf_ender_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in x_test[iindex])]
        except:
            print("search thickness, bad lstm_tsf_ender_thick_2_Oven")
            continue
        try:
            cur_oven = pre_cur_oven[pre_oven]
            real_curve_slim = [oven_real_lab[cur_oven][r] for r in important_inds]
            cur_oven_normalize_sensor_feature = test_normalize_feature[cur_oven]
            cur_thickness = cur_oven_thick[cur_oven]
        except:
            print("search thickness, bad pre_oven: {}".format(pre_oven))
            continue
        # if cur_oven in ["33321110507", "33321110606", "33321110607", "33321110704", "33321110705", "33321111504"]:
        pred_curve_slim = oven_pred_lab[cur_oven]
        thick_mean = mean_[-5:]
        thick_std = std_[-5:]

        finetune_thre1 = [0.01, 0.01, 0.3, 0.5, 0.8]
        return_scale = [1]*7
        return_scale[4], return_scale[5] = 0.5, 0.8

        def get_deta_function_and_best(thickness_list):
            preds_dims = []
            for ml_model in ml_models:
                nor_thickness = [(thickness_list[i] - thick_mean[i]) / thick_std[i] for i in range(5)]
                test = cur_oven_normalize_sensor_feature + nor_thickness
                preds_dims.append(ml_model.predict(np.array([test]))[0])
            recommend_thickness = [thickness_list[0]] + [4] + thickness_list[1:] + [35]
            scaled_thickness = scale_deta_thickness(cur_thickness, recommend_thickness)
            cur_function = function_thick2lab(scaled_thickness)
            deta_function_and_best = [cur_function[important_ind] - cx_lab_curve[important_ind] for important_ind in
                                      important_inds]
            slim_function = [cur_function[ind] for ind in important_inds]

            return deta_function_and_best, slim_function, preds_dims


        def weighted_metric(a, b, weights, model_pred_m):
            ll = len(a)
            diff = [np.power((a[i] - b[i]), 2) for i in range(ll)]
            # diff = [abs(a[i] - b[i]) for i in range(ll)]

            if model_pred_m[2] > 0.26:
                weights[2] *= 2
            if model_pred_m[9] > 0.2:
                weights[9] *= 2
            if model_pred_m[4] > 2:
                weights[4] *= 2
            if model_pred_m[12] > 3:
                weights[12] *= 2

            diff = [diff[i]*weights[i] for i in range(14)]
            res = np.mean(diff)
            tmp_weights = [1] * ll
            tmp_weights[2], tmp_weights[4], tmp_weights[9], tmp_weights[12] = 0, 0, 0, 0
            tmp_diff = [diff[i] * tmp_weights[i] for i in range(ll)]
            a = np.mean(tmp_diff)
            b = res - a
            print("总mse: {}, 重要频段mse: {}, 其他mse: {}".format(res, b, a))

            return np.mean(diff)

        def LAB_curve_metric(gt, pred):
            pred = np.array(pred)
            gt = np.array(gt)
            pred /= gt
            gt /= gt
            return mean_squared_error(pred, gt)

        def evaluate_config(config):
            ll = len(important_inds)
            y1, slim_f, preds_dims = get_deta_function_and_best([config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
            weights = [0] * ll
            weights[2], weights[9], weights[4], weights[12], weights[13] = 1,1,2,2,2
            model_pred_m = [slim_f[ind] - preds_dims[ind] for ind in range(ll)]
            # plt.plot([i for i in range(14)], model_pred_m, label='model_pred')
            # plt.legend()
            # plt.show()
            metric = weighted_metric(y1, preds_dims, weights, model_pred_m)

            # use tune.report to report the metric to optimize
            tune.report(metric=metric)

        analysis = tune.run(
            evaluate_config,  # the function to evaluate a config
            config={
                'x1': tune.uniform(lower=cur_thickness[0] - finetune_thre1[0],
                                   upper=cur_thickness[0] + finetune_thre1[0]),
                'x2': tune.uniform(lower=cur_thickness[2] - finetune_thre1[1],
                                   upper=cur_thickness[2] + finetune_thre1[1]),
                'x3': tune.uniform(lower=cur_thickness[3] - finetune_thre1[2],
                                   upper=cur_thickness[3] + finetune_thre1[2]),
                'x4': tune.uniform(lower=cur_thickness[4] - finetune_thre1[3],
                                   upper=cur_thickness[4] + finetune_thre1[3]),
                'x5': tune.uniform(lower=cur_thickness[5] - finetune_thre1[4],
                                   upper=cur_thickness[5] + finetune_thre1[4]),
            },  # the search space
            low_cost_partial_config={'x1': cur_thickness[0],
                                     'x2': cur_thickness[2],
                                     'x3': cur_thickness[3],
                                     'x4': cur_thickness[4],
                                     'x5': cur_thickness[5]},  # a initial (partial) config with low cost
            metric='metric',  # the name of the metric used for optimization
            mode='min',  # the optimization mode, 'min' or 'max'
            num_samples=-1,  # the maximal number of configs to try, -1 means infinite
            time_budget_s=10,  # the time budget in seconds 优化时间限制, 10s停止优化
            local_dir='logs/',  # the local directory to store logs
        )

        search_res = analysis.best_trial.last_result
        thickness_res = [np.round(search_res['config']['x1'], 2), 4, np.round(search_res['config']['x2'], 2),
                         np.round(search_res['config']['x3'], 2), np.round(search_res['config']['x4'], 2),
                         np.round(search_res['config']['x5'], 2), 35]

        thickness_diff = [thickness_res[i] - cur_thickness[i] for i in range(7)]

        # scale result
        thickness_diff = [thickness_diff[i] * return_scale[i] for i in range(7)]
        thickness_diff[1], thickness_diff[6] = 0, 0

        plt.plot(aa, real_curve_slim, color='pink',
                 label='real_curve: {}'.format(''.join(str(np.round(a, 3)) + ', ' for a in cur_thickness)))
        plt.plot(aa, pred_curve_slim, color='dimgray', linewidth=1.0,
                 label='pred_curve, thickness diff: {}'.format(''.join(str(np.round(a, 2)) + ', ' for a in thickness_diff)))
        plt.plot(aa, slim_best_cx_curve, color='cornflowerblue', label='best_curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(cur_oven)))
        # plt.show()
        plt.close()


def run_train(seed, pca, train_feature, test_feature, model_flag, important_inds, all_data):
    x_train = r'./X_train_all8_sensors_seed{}pca{}.npy'.format(seed, pca)
    y_train = r'./Y_train_all8_sensors_seed{}pca{}.npy'.format(seed, pca)
    x_test = r'./X_test_all8_sensors_seed{}pca{}.npy'.format(seed, pca)
    y_test = r'./Y_test_all8_sensors_seed{}pca{}.npy'.format(seed, pca)

    print("train scale model, ssed: {}, pca: {}".format(seed, pca))

    # 1.train model
    print("train model, ssed: {}, pca: {}".format(seed, pca))
    ml_models, test_oven = train(seed, pca, model_flag, x_train, y_train, x_test, y_test,
                                         important_inds, all_data, train_feature, test_feature)

    return ml_models, test_oven



def run_forward(ff, seed, pca, test_oven, all_data, test_feature_dict):
    save_dir = r'./1117/seed{}_forward'.format(seed)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness, oven_pred_lab, Oven_real_lab, pre_cur_oven = \
        forward_model(ff, seed, pca, test_oven, save_dir, all_data, test_feature_dict)

    # np.save('./test_seed{}_pca{}.npy'.format(seed, pca), X_test_data)
    # np.save('./scaled_thickness_seed{}_pca{}.npy'.format(seed, pca), np.array(scaled_thickness))
    # data = json.dumps(lstm_tsf_ender_thick_2_Oven)
    # with open(r'./feature_oven_seed{}_pca{}.json'.format(seed, pca), 'w') as js_file:
    #     js_file.write(data)
    # data = json.dumps(pre_cur_oven)
    # with open(r'./pre_cur_oven_seed{}_pca{}.json'.format(seed, pca), 'w') as js_file:
    #     js_file.write(data)
    # data = json.dumps(Oven_real_lab)
    # with open(r'./Oven_real_lab_seed{}_pca{}.json'.format(seed, pca), 'w') as js_file:
    #     js_file.write(data)
    # data = json.dumps(oven_pred_lab)
    # with open(r'./Oven_pred_lab_seed{}_pca{}.json'.format(seed, pca), 'w') as js_file:
    #     js_file.write(data)

    return X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness, oven_pred_lab, Oven_real_lab, pre_cur_oven


def quick_search_thckness(pca, seed, model_flag, best_cx_curve, slim_best_cx_curve, important_inds):
    save_dir_ = r'./1117/seed{}_search_thickness'.format(seed)
    if not os.path.exists(save_dir_):
        os.mkdir(save_dir_)
    X_test_data = np.load('./test_seed{}_pca{}.npy'.format(seed, pca))
    lstm_tsf_ender_thick_2_Oven = json.load(open(r'./feature_oven_seed{}_pca{}.json'.format(seed, pca), 'r'))
    pre_cur_oven = json.load(open(r'./pre_cur_oven_seed{}_pca{}.json'.format(seed, pca), 'r'))
    Oven_real_lab = json.load(open(r'./Oven_real_lab_seed{}_pca{}.json'.format(seed, pca), 'r'))
    Oven_pred_lab = json.load(open(r'./Oven_pred_lab_seed{}_pca{}.json'.format(seed, pca), 'r'))
    search_thickness(pca, seed, model_flag, X_test_data, best_cx_curve, slim_best_cx_curve, important_inds,
                     save_dir_, lstm_tsf_ender_thick_2_Oven, pre_cur_oven, Oven_real_lab, Oven_pred_lab)


def eval_metric(diff):
    weights = [1] * 14
    weights[2], weights[9], weights[4], weights[12] = 2, 2, 3, 5
    eval_metric = [np.power(diff[i] / slim_best_cx_curve[i], 2) * weights[i] for i in range(14)]

    return np.mean(eval_metric)



def curve_trend_check(pca, save_dir, x_test, lstm_tsf_ender_thick_2_Oven, scaled_thickness):
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ll = len(aa)
    xgb_models = []
    ff = open('./X_mean_std_train.txt', 'r').readlines()
    for ind, line in enumerate(ff):
        if ind == 0:
            mean_ = [float(a) for a in line.split(',')[:-1]]
        if ind == 1:
            std_ = [float(a) for a in line.split(',')[:-1]]
    for i in range(ll):
        params = json.load(open(r'./parameter_{}_seed{}_pca{}.json'.format(i, seed, pca), 'r'))
        model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', **params)
        model.load_model(r'./qyicktest_ml_{}.model'.format(i))
        xgb_models.append(model)
    x_test = x_test.tolist()
    x_test_no_normalize = [[test[r] * std_[r] + mean_[r] for r in range(len(mean_))] for test in x_test]
    x_lens = len(x_test_no_normalize[0])

    modified = [0, 0, 0, 0, -1, 0, 0]
    all_modified = [0] * (x_lens - 5) + [0, 0, 0, -1, 0]

    # x: pre_sensor_feature + cur_thickness -> deta(f, m)
    for iindex, single_test_data in enumerate(x_test):
        de_single_test_data = x_test_no_normalize[iindex]
        de_single_test_data = [de_single_test_data[r]+all_modified[r] for r in range(x_lens)]
        de_single_test_data = [(de_single_test_data[k]-mean_[k])/std_[k] for k in range(x_lens)]
        single_preds = []
        modified_single_preds = []
        oven = lstm_tsf_ender_thick_2_Oven[''.join(str(np.round(a, 3)) + ',' for a in x_test[iindex])]
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
        plt.plot(aa, pred_m_modified, label='layer6-1, model_predicate', color='pink')
        plt.legend()
        plt.savefig(os.path.join(save_dir, './{}.png'.format(oven)))
        plt.close()


def run_trend_check(pca, X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness):
    save_dir = r'./1117/seed{}_human_modify'.format(seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    curve_trend_check(pca, save_dir, X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness)


if __name__ == '__main__':
    sensor_cols = ["ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH", "ACT_V1_THERMOVAC_PREVLINE",
                   "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP",
                   "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE",
                   "ACT_Q10_CURRENT_ANODE", "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE",
                   "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL", "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2",
                   "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]
    best_cx_curve = [6.6524, 4.5071, 2.2922, 1.1451, 0.7214, 0.4108, 0.2415, 0.3834, 0.5492, 0.5964, 0.6647, 0.8838,
                     1.0936, 1.0963, 0.992, 0.972, 1.0551, 1.1061, 1.0686, 0.927, 0.8003, 0.7419, 0.7491, 0.7563,
                     0.7154, 0.6076, 0.4921, 0.4414, 0.4152, 0.4251, 0.4191, 0.3947, 0.3412, 0.2696, 0.2235, 0.1996,
                     0.1907, 0.1965, 0.1844, 0.1592, 0.1281, 0.0889, 0.0598, 0.0446, 0.031, 0.0375, 0.0398, 0.0386,
                     0.0287, 0.0257, 0.0249, 0.0326, 0.0593, 0.093, 0.1355, 0.1803, 0.2255, 0.268, 0.3089, 0.3568,
                     0.4025, 0.4695, 0.5404, 0.6328, 0.7406, 0.8652, 1.0015, 1.1388, 1.2783, 1.4115, 1.543, 1.6709,
                     1.8066, 1.9292, 2.0527, 2.1884, 2.3387, 2.5097, 2.6949, 2.8866, 3.1036]
    # smooth best_cx_curve
    best_cx_curve = savgol_filter(best_cx_curve, 15, 5).tolist()
    important_inds = [0, 3, 6, 10, 15, 19, 24, 29, 34, 44, 54, 64, 74, 80]

    slim_best_cx_curve = [best_cx_curve[i] for i in important_inds]

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    data5 = r'./1105_cx_data.csv'
    data6 = r'./1121_cx_data.csv'
    all_data = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4), pd.read_csv(data5), pd.read_csv(data6)]
    ok_ovens = pre_data_process(all_data)

    model_flag = 0
    flags = ['train and forward', 'trend_check', 'quick_search_thickness']
    flag = flags[2]
    pcas = [300]

    # seeds = [i + 33 for i in range(5)]
    seeds = [33]
    ff = open('./11111.txt', 'w')
    for pca in pcas:
        for seed in seeds:
            train_feature = json.load(open(r'./no_thick_kpca_train_seed{}pca{}.json'.format(seed, pca), 'r'))
            test_feature = json.load(open(r'./no_thick_kpca_test_seed{}pca{}.json'.format(seed, pca), 'r'))

            X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness = [],[],[]
            if flag == "train and forward":
                #1.train model
                ml_models, test_oven = run_train(seed, pca, train_feature, test_feature, model_flag, important_inds, all_data)
                X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness, oven_pred_lab, Oven_real_lab, pre_cur_oven = \
                    run_forward(ff, seed, pca, test_oven, all_data, test_feature)

                run_trend_check(pca, X_test_data, lstm_tsf_ender_thick_2_Oven, scaled_thickness)

            # 4. qucik search thickness
            if flag == "quick_search_thickness":
                quick_search_thckness(pca, seed, model_flag, best_cx_curve, slim_best_cx_curve, important_inds, )
