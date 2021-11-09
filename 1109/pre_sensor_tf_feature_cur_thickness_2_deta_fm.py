# coding=utf-8

'''
pre_sensor feature[可暂定ACT_O1_QCMS_THICKNESS_CH1这列, 8个step分段, 但是提取什么特征好需要想想[后续考虑搭建一个encode来做] + 下一炉thickness设置值 -> deta(f-m)
考虑设计一个函数, scale 一下deta_thick, 让function里膜厚调整在lab曲线上显著点..  现在是简单的做了一个 *3 放大
'''

import os
import matplotlib.pyplot as plt
import tsfresh as tsf
from function_thick2lab import function_thick2lab
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error
from flaml import tune
from scipy.signal import savgol_filter


def report_best_scores(results, ind, n_top=3):
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
    with open(r'./parameter_{}_tf.json'.format(ind), 'w') as js_file:
        js_file.write(data)


def xgb_hyperparameter_searching(X, Y, ind):

    ml_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.8, 0.1),  # 特征采集比例
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

def cross_val(model_flag, X_train, Y_train, x_test, y_test, ind, important_inds):
    
    if model_flag == 0:
        parameters = json.load(open(r'./parameter_{}_tf.json'.format(ind), 'r'))
        ml_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
        ml_model.fit(X_train, Y_train)
        save_model(ml_model, r'./ml_{}.model'.format(ind))

    elif model_flag == 1:
        # lightgbm
        params = {'n_estimators': 110, 'learning_rate': 0.21550071029494922, 'num_leaves': 12, 'max_depth': 6, 'min_data_in_leaf': 50, 'max_bin': 181, 'lambda_l2': 100, 'min_gain_to_split': 10.627711503070213, 'bagging_fraction': 0.4, 'bagging_freq': 1, 'feature_fraction': 0.8}
        ml_model = lgb.LGBMRegressor(objective='regression', **params)
        ml_model.fit(X_train, Y_train)
        joblib.dump(ml_model, r'./ml_{}.pkl'.format(ind))
    else:
        # catboost
        ml_model = []
        pass
    
    # eval()
    ok, ng = 0, 0
    x_test = x_test.tolist()
    y_test = y_test.tolist()
    for iind in range(len(x_test)):
        pre_test_data = ml_model.predict(np.array([x_test[iind]]))
        # print("important ind {}, best_curve value: {}, deta_real : {}, deta_pred: {}, diff: {}".format(important_inds[ind]*5+380, cx_lab_curve[ind], y_test[iind][0], pre_test_data[0], y_test[iind][0]-pre_test_data[0]))
        # print("重要频段点预测和真实是否趋势一致: {}".format((y_test[iind][0] * pre_test_data[0]) > 0))
        if (y_test[iind][0] * pre_test_data[0]) > 0:
            ok += 1
        else:
            ng += 1
        # loss = mean_squared_error(np.array(pre_test_data), np.array(y_test[ind]) )
    print("频段{}, deta趋势一致: {}, 趋势不一致: {}, all data size: {}".format(important_inds[ind]*5+380, ok, ng, ok+ng))

    return ml_model


def eval_model(parameters, x, ind):
    model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    model.load_model(r'./ml_{}.model'.format(ind))
    res = model.predict(np.array([x]))

    return res.tolist()


def strip_a(a):
    a = a.strip("['] ")
    return a

def slim_thickness(thick):
    res = []
    inds = [0, 2, 3, 4, 5]
    for ind in inds:
        res.append(thick[ind])

    return res


def generate_single_step_feature(col):
    # 后续考虑搭建一个encode来做这一块
    # 提取数据的平稳性特征
    ts = pd.Series(col)
    ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])

    return ae2[0][1]


def get_sensor_feature(col):
    # col, sensor.csv中的'ACT_O1_QCMS_THICKNESS_CH1'列
    step_features = []
    thick_inds = []
    ll = len(col)-1
    # 数值截断取每个step段
    for i in range(ll):
        if col[i] > 0 and col[i+1] == 0:
            thick_inds.append(i)
    ll_ = len(thick_inds)

    if ll_ == 6:
        # 有的sensor文件, 镀膜记录值截至在: 24.29864, 后面没有0了.. 导致最后一个ind获取不到..
        thick_inds.append(ll)

    for i in range(len(thick_inds)):
        if i == 0:
            tmp = [a for a in col[:thick_inds[i]+1]]
        else:
            tmp = col[thick_inds[i-1]+1: thick_inds[i]+1]
        step_features.append(generate_single_step_feature(tmp))
    assert len(step_features) == 7

    return step_features


def scale_deta_thickness(base_thick, cur_thick):
    # print("base thickness: {}".format(base_thick))
    # print("cur thickness: {}".format(cur_thick))
    # zhige: scale 一下 deta_thick, 让function里膜厚调整在lab曲线上显著点
    inds = [0, 2, 3, 4, 5]
    base_cur_diff = [cur_thick[ind] - base_thick[ind] for ind in inds]

    # 这个函数应该还需要精细设计, 现在是简单的做x3处理.
    base_cur_diff = [a*1 for a in base_cur_diff]
    base_cur_diff = [base_cur_diff[0]] + [0] + base_cur_diff[1:] + [0]
    # print("base_cur_diff: ", base_cur_diff)
    modified_thick = [base_cur_diff[k] + base_thick[k] for k in range(7)]
    # print("modified_thick: ", modified_thick)

    return modified_thick



def samex_samey(X, Y):
    x_y = dict()
    for ind, x in enumerate(X):
        key = ''.join(str(np.round(a, 3)) for a in x)
        if key not in x_y:
            x_y[key] = []
        x_y[key].append(Y[ind])
    assert len(x_y) == 278
    for k, v in x_y.items():
        if len(v) != 1:
            for v_ in v:
                print(v_)
            print('---')


def generate_data(datas, important_inds, x_data_file, y_data_file, sensor_thick_2_OvenNo, OvenNoPair_Labs, Ovens_embedding_sensor_feature):
    if os.path.exists(x_data_file):
        return np.load(x_data_file), np.load(y_data_file)
    else:
        sensor_and_thick2OvenNo = dict()
        OvenNos_pairs = dict()
        X = []
        Y = []
        Y_ = []
        scaled_thicknesss = []
        oven12 = []
        for data in datas:
            ll = len(data)
            for j in range(ll-1):
                sensor_file_name1, sensor_file_name2 = data.iloc[j]['FileID'][3:], data.iloc[j+1]['FileID'][3:]
                # 控制炉号连续, diff<=3
                if int(sensor_file_name2) - int(sensor_file_name1) <= 3:
                    # load embedding sensor_feature
                    pre_oven = data.iloc[j]['OvenNo']
                    pre_sensor = Ovens_embedding_sensor_feature[pre_oven]
                    # pre_thickness
                    pre_thickness = [float(strip_a(a)) for a in data.iloc[j]['Thickness'].split(',')]
                    # cur_thickness
                    cur_thickness = [float(strip_a(a)) for a in data.iloc[j+1]['Thickness'].split(',')]
                    scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness)
                    cur_function = function_thick2lab(scaled_thickness)
                    x = pre_sensor + slim_thickness(cur_thickness)
                    assert len(x) == len(pre_sensor)+5
                    # 滤波处理下~
                    cur_lab = [float(strip_a(a)) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                    cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                    pre_lab = [float(strip_a(a)) for a in data.iloc[j]['single_lab_curve'].split(',')]
                    pre_lab = savgol_filter(pre_lab, 15, 5).tolist()
                    # plt.plot([i*5+380 for i in range(81)], pre_lab)
                    # plt.show()
                    Y_.append([cur_function[i] - cur_lab[i] for i in range(81)])
                    # Y: 重要频段的, deta(f, m)
                    y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in important_inds]
                    X.append(x)
                    scaled_thicknesss.append(scaled_thickness)
                    OvenNos_pairs["{}_{}".format(data.iloc[j]['OvenNo'], data.iloc[j+1]['OvenNo'])] = [pre_lab, cur_lab]
                    Y.append(y_important_inds)
                    oven12.append("{}_{}".format(data.iloc[j]['OvenNo'], data.iloc[j+1]['OvenNo']))

        # check x and y
        samex_samey(X, Y)

        # 落盘连续的炉号和对应的曲线
        data = json.dumps(OvenNos_pairs)
        with open(OvenNoPair_Labs, 'w') as js_file:
            js_file.write(data)

        # 统计x的均值方差
        mean_ = np.mean(X, axis=0)
        std_ = np.std(X, axis=0)

        # std_清零处理
        # std_ = [a+0.000001 for a in std_]
        slim = [a for a in range(len(std_)) if std_[a] != 0]
        mean_ = [mean_[k] for k in slim]
        std_ = [std_[k] for k in slim]

        ff = open('./X_mean_std.txt', 'w')
        ff.write(''.join(str(a) + ',' for a in mean_))
        ff.write('\n')
        ff.write(''.join(str(a) + ',' for a in std_))
        ff.close()

        ff = open('./X_mean_std.txt', 'r').readlines()
        mean_ = [float(a) for a in ff[0].split(',')[:-1]]
        std_ = [float(b) for b in ff[1].split(',')[:-1]]
        X = [[x[ind] for ind in slim] for x in X]

        print("base x: {}".format(X[0]))
        X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
        print("normalized x: {}".format(X[0]))

        for ind, x in enumerate(X):
            sensor_and_thick2OvenNo[''.join(str(np.round(a, 3)) + ',' for a in x)] = oven12[ind]
        data = json.dumps(sensor_and_thick2OvenNo)
        with open(sensor_thick_2_OvenNo, 'w') as js_file:
            js_file.write(data)

        X, Y = np.array(X), np.array(Y)

        # check x key 没有重复
        # assert len(sensor_and_thick2OvenNo) == X.shape[0]

        np.save(x_data_file, X)
        np.save(y_data_file, Y)
        np.save('./scaled_thickness.npy', np.array(scaled_thicknesss))

        return X, Y


def double_curve_check(datas):
    names = ['0910data', '0917data', '1008data', '1021data']
    aa = [380+5*i for i in range(81)]
    for ind, data in enumerate(datas):
        OvenNos = data['OvenNo']
        print(names[ind], ':')
        plt.title(names[ind])
        doubles = []
        for iind, tmp in enumerate(data['double_lab_curve']):
            try:
                _ = len(tmp)
                tmp = [float(strip_a(a)) for a in tmp.split(',')]
                if tmp[0] < 40 and tmp[4] < 5:
                    doubles.append(tmp)
                elif tmp[4] > 5:
                    print("{}: 400波长处, 反射率值大于5的样本 LAB值: {}".format(OvenNos[iind], data['double_lab_value'][iind]))
                    plt.plot(aa, tmp, color='pink')
            except:
                continue
        for curve in doubles:
            plt.plot(aa, curve)
        plt.show()


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


def eval_test_data(x_test, important_inds, y_tests, title='train data'):
    ll = len(important_inds)
    ml_models = []
    for i in range(ll):
        params = json.load(open(r'./parameter_{}_tf.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./ml_{}.model'.format(i))
        ml_models.append(model)
    preds = []
    for ml_model in ml_models:
        pred = ml_model.predict(x_test)
        preds.append(pred)
    preds = [[preds[k][j] for k in range(ll)] for j in range(x_test.shape[0])]
    gt = [[y_tests[k][j] for k in range(ll)] for j in range(x_test.shape[0])]
    x_test = x_test.tolist()
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test = [[x[r]*std_[r] + mean_[r] for r in range(len(mean_)-5, len(mean_))] for x in x_test]
    functions = [function_thick2lab([x[0]] + [4] + x[1:] + [35]) for x in x_test]
    functions = [[tmp[r] for r in important_inds] for tmp in functions]
    res = [[functions[j][k] - preds[j][k] for k in range(11)] for j in range(len(x_test))]
    gt = [[functions[j][k] - gt[j][k] for k in range(11)] for j in range(len(x_test))]
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    print("{} mse: {}".format(title, mean_squared_error(res, gt)))
    plt.title(title)
    for ind, single_res in enumerate(res):
        if ind == 0:
            plt.plot(aa, single_res, color='pink', label='pred')
            plt.plot(aa, gt[ind], color='blue', label='gt')
        else:
            if gt[ind][5] > 1.1:
                plt.plot(aa, single_res, color='red')
                plt.plot(aa, gt[ind], color='yellow')
            else:
                plt.plot(aa, single_res, color='pink')
                plt.plot(aa, gt[ind], color='blue')
    plt.legend()
    plt.show()


def bad_Ovens(datas):
    bads = []
    c = 0
    for data in datas:
        ll = len(data)
        for i in range(ll-1):
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


def Test(model_flag, x_test, cx_lab_curve, slim_best_cx_curve, important_inds, sensor_thick_2_OvenNo, OvenNoPair_Labs, save_dir):
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
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]

    x_test_no_normalize = [[test[r] * std_[r] + mean_[r] for r in range(len(mean_))] for test in x_test]
    sensor_thick_2_OvenNo_data = json.load(open(sensor_thick_2_OvenNo, 'r'))
    OvenNoPair_Lab_data = json.load(open(OvenNoPair_Labs, 'r'))

    for iindex, single_test_data in enumerate(x_test):
        oven1_oven2 = sensor_thick_2_OvenNo_data[''.join(str(np.round(a, 3))+',' for a in x_test[iindex])]
        real_curve_slim = [OvenNoPair_Lab_data[oven1_oven2][1][r] for r in important_inds]
        preds_dim11 = []
        for ml_model in ml_models:
            preds_dim11.append(ml_model.predict(np.array([single_test_data]))[0])
        assert len(preds_dim11) == len(important_inds)

        embedding_sensor_feature_dim = len(std_) - 5
        pre_thickness = [x_test_no_normalize[iindex][embedding_sensor_feature_dim:][0]] + [4] + x_test_no_normalize[iindex][embedding_sensor_feature_dim+1:] + [35]
        finetune_thre1 = [0.1, 0.1, 0.6, 1, 1.5]

        def get_y1(thickness_list):
            recommend_thickness = [thickness_list[0]] + [4] + thickness_list[1:] + [35]
            scaled_thickness = scale_deta_thickness(pre_thickness, recommend_thickness)
            cur_function = function_thick2lab(scaled_thickness)
            deta_function_and_best = [cur_function[important_ind] - cx_lab_curve[important_ind] for important_ind in important_inds]

            return deta_function_and_best

        def weighted_metric(a, b, weights):
            ll = len(a)
            diff = [(a[i]-b[i])*(a[i]-b[i]) for i in range(ll)]
            diff = [diff[i]*weights[i] for i in range(ll)]

            return np.mean(diff)

        def evaluate_config(config):
            y1 = get_y1([config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
            metric = mean_squared_error(y1, preds_dim11)

            # weights = [0.16, 1, 1, 1.3, 1.3, 1.3, 1.3, 1.3, 1, 1.2, 1.2, 1, 1]
            # weights = [0.16] + [1] * (len(y1)-1)
            weights = [1]*len(y1)
            metric = weighted_metric(y1, preds_dim11, weights)

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
        print(search_res)
        thickness_res = [np.round(search_res['config']['x1'], 2), 4, np.round(search_res['config']['x2'], 2),np.round(search_res['config']['x3'], 2),np.round(search_res['config']['x4'], 2),np.round(search_res['config']['x5'], 2), 35]
        plt.plot(aa, real_curve_slim, color='pink', label='cur_curve: {}'.format(''.join(str(a)+', ' for a in pre_thickness)))
        plt.plot(aa, slim_best_cx_curve, color='cornflowerblue', label='to get best_curve: {}'.format(''.join(str(a)+', ' for a in thickness_res)))
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(oven1_oven2.split('_')[1])))
        # plt.show()
        plt.close()


def compare_pred_realCurve(x_test, scaled_thickness, save_dir):
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    ll = len(aa)
    xgb_models = []
    for i in range(ll):
        params = json.load(open(r'./parameter_{}_tf.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./ml_{}.model'.format(i))
        xgb_models.append(model)
    x_test = x_test.tolist()
    sensor_thick_2_OvenNo_data = json.load(open(sensor_thick_2_OvenNo, 'r'))
    OvenNoPair_Lab_data = json.load(open(OvenNoPair_Labs, 'r'))

    test_real = []
    test_preds = []
    for iindex, single_test_data in enumerate(x_test):
        single_preds = []
        oven1_oven2 = sensor_thick_2_OvenNo_data[''.join(str(np.round(a, 3))+',' for a in x_test[iindex])]
        real_curve_slim = [OvenNoPair_Lab_data[oven1_oven2][1][r] for r in important_inds]
        for xgb_model in xgb_models:
            single_pred = xgb_model.predict(np.array([single_test_data]))
            single_preds.append(single_pred)
        # deta(f, m), m = f-deta
        single_f = function_thick2lab(scaled_thickness[iindex])
        single_f = [single_f[i] for i in important_inds]
        pred_m = [single_f[i]-single_preds[i] for i in range(len(important_inds))]
        test_real.append(real_curve_slim)
        test_preds.append(pred_m)
        plt.plot(aa, real_curve_slim, label='real_curve', color='pink')
        plt.plot(aa, pred_m, label='pred_curve', color='cornflowerblue')
        plt.legend()
        plt.savefig(os.path.join(save_dir, './{}.png'.format(iindex)))
        plt.close()
    diff = [[abs(test_real[i][k] - test_preds[i][k]) for k in range(ll)] for i in range(len(test_real))]
    diff_13 = np.mean(diff, axis=0)
    print(diff_13)


if __name__ == '__main__':

    cc_lab_curve = [7.3356, 4.8663, 2.5198, 1.4276, 0.8527, 0.4359, 0.3289, 0.4565, 0.5573, 0.6074, 0.8496, 1.0507,
                   1.1332, 1.0802, 1.0985, 1.202, 1.2691, 1.2089, 1.065, 0.9647, 0.9341, 0.9569, 0.9191, 0.8223, 0.6908,
                   0.6092, 0.5654, 0.5678, 0.5585, 0.5084, 0.4354, 0.3652, 0.3153, 0.301, 0.2963, 0.2923, 0.2769,
                   0.2354, 0.1909, 0.158, 0.1339, 0.1295, 0.1184, 0.1111, 0.0976, 0.0863, 0.0688, 0.064, 0.0584, 0.0639,
                   0.0748, 0.0901, 0.1025, 0.1135, 0.1261, 0.1485, 0.1761, 0.2271, 0.2806, 0.3443, 0.4229, 0.5029,
                   0.5719, 0.6471, 0.7214, 0.788, 0.8605, 0.9497, 1.052, 1.1625, 1.2945, 1.4421, 1.6149, 1.7815, 1.947,
                   2.1099, 2.2678, 2.4232, 2.5676, 2.7107, 2.8521]

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    data5 = r'./1105_cx_data.csv'
    datas = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4), pd.read_csv(data5)]
    # double_curve_check(datas)
    # best_cx_thickness = [24.82, 5, 11.54, 36.32, 101.3, 93.4, 25]
    # best_cc_thickness = [24.12, 5, 12.17, 36.49, 104.6, 96.45, 25]

    # important_inds = [0, 3, 6, 14, 24, 34, 44, 54, 64, 74, 80]
    important_inds = [0, 3, 6, 14, 19, 24, 29, 34, 44, 54, 64, 74, 80]
    x_data_file = r'./X_tf.npy'
    y_data_file = r'./Y_tf.npy'
    sensor_thick_2_OvenNo = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\cx_data_result\sensor_thick_2_oveNno.json'
    OvenNoPair_Labs = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\cx_data_result\OvenNoPair_Labs.json'

    # load embedding_ed sensor_feature
    # sensor_embedding_feature = np.load(r'./168dims_sensor_feature.npy').tolist()
    # sensor_embedding_feature = np.load(r'./tf_feature.npy').tolist()
    sensor_embedding_feature = np.load(r'./all_tf_sensor_features264.npy')     # np.load(r'./all_tf_sensor_features.npy')
    
    # 提取到了tf_lstm特征的OvenNos
    Ovens = np.load(r"./Ovens.npy").tolist()
    data_size, _ = sensor_embedding_feature.shape[:]
    assert data_size == len(Ovens)
    sensor_embedding_feature = sensor_embedding_feature.tolist()
    Ovens_embedding_sensor_feature = dict()
    for ind, oven in enumerate(Ovens):
        Ovens_embedding_sensor_feature[oven] = sensor_embedding_feature[ind]
    X, Y = generate_data(datas, important_inds, x_data_file, y_data_file, sensor_thick_2_OvenNo, OvenNoPair_Labs, Ovens_embedding_sensor_feature)
    print(X.shape, Y.shape)
    Y = Y.tolist()

    # ['xgboost', 'lightgbm', 'catboost']
    model_flag = 0

    ml_models = []
    y_tests = []
    y_trains = []
    for ind, important_ind in enumerate(important_inds):
        y = np.array([[Y_[ind]] for Y_ in Y])
        X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=669)
        y_tests.append([a[0] for a in y_test])
        y_trains.append([a[0] for a in y_train])
        if ind == 0:
            np.save(r'./x_test_tf.npy', X_test)
            np.save(r'./x_train_tf.npy', X_train)

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

        ml_model = cross_val(model_flag, X_train, y_train, X_test, y_test, ind, important_inds)
        ml_models.append(ml_model)

    x_test = np.load(r'./x_test_tf.npy')
    #2. model_pred and real_cuvre check
    save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\lstm_model_and_real_compare'
    scaled_thickness = np.load(r'./scaled_thickness.npy')
    compare_pred_realCurve(x_test, scaled_thickness, save_dir)


    #3. thickness search
    save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\1106_tf_li_gong_review_{}'.format(model_flag)
    best_cx_curve = [6.6524, 4.5071, 2.2922, 1.1451, 0.7214, 0.4108, 0.2415, 0.3834, 0.5492, 0.5964, 0.6647, 0.8838,
                     1.0936, 1.0963, 0.992, 0.972, 1.0551, 1.1061, 1.0686, 0.927, 0.8003, 0.7419, 0.7491, 0.7563,
                     0.7154, 0.6076, 0.4921, 0.4414, 0.4152, 0.4251, 0.4191, 0.3947, 0.3412, 0.2696, 0.2235, 0.1996,
                     0.1907, 0.1965, 0.1844, 0.1592, 0.1281, 0.0889, 0.0598, 0.0446, 0.031, 0.0375, 0.0398, 0.0386,
                     0.0287, 0.0257, 0.0249, 0.0326, 0.0593, 0.093, 0.1355, 0.1803, 0.2255, 0.268, 0.3089, 0.3568,
                     0.4025, 0.4695, 0.5404, 0.6328, 0.7406, 0.8652, 1.0015, 1.1388, 1.2783, 1.4115, 1.543, 1.6709,
                     1.8066, 1.9292, 2.0527, 2.1884, 2.3387, 2.5097, 2.6949, 2.8866, 3.1036]
    slim_best_cx_curve = [best_cx_curve[i] for i in important_inds]
    # Test(model_flag, x_test, best_cx_curve, slim_best_cx_curve, important_inds, sensor_thick_2_OvenNo, OvenNoPair_Labs, save_dir)