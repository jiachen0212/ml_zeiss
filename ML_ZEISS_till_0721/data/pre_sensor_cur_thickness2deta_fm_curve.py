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
    with open(r'./parameter_{}_152.json'.format(ind), 'w') as js_file:
        js_file.write(data)


def hyperparameter_searching(X, Y, ind):

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

    report_best_scores(search.cv_results_, ind, 5)


def save_model(model, model_path):
    model.save_model(model_path)

def cross_val(X_train, Y_train, x_test, y_test, ind, cx_lab_curve, important_inds):

    parameters = json.load(open(r'./parameter_{}_152.json'.format(ind), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, Y_train)
    save_model(xgb_model, r'./xgb_{}.model'.format(ind))

    # eval model
    ok, ng = 0, 0
    x_test = x_test.tolist()
    y_test = y_test.tolist()
    for iind in range(len(x_test)):
        pre_test_data = xgb_model.predict(np.array([x_test[iind]]))
        # print("important ind {}, best_curve value: {}, deta_real : {}, deta_pred: {}, diff: {}".format(important_inds[ind]*5+380, cx_lab_curve[ind], y_test[iind][0], pre_test_data[0], y_test[iind][0]-pre_test_data[0]))
        # print("重要频段点预测和真实是否趋势一致: {}".format((y_test[iind][0] * pre_test_data[0]) > 0))
        if (y_test[iind][0] * pre_test_data[0]) > 0:
            ok += 1
        else:
            ng += 1

        # loss = mean_squared_error(np.array(pre_test_data), np.array(y_test[ind]) )
    print("频段{}, deta趋势一致: {}, 趋势不一致: {}, all data size: {}".format(important_inds[ind]*5+380, ok, ng, ok+ng))
    return xgb_model


def eval_model(parameters, x, ind):
    model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    model.load_model(r'./xgb_{}.model'.format(ind))
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


def get_sensor_feature(col, sensor=None):
    thickness_col = sensor['ACT_O1_QCMS_THICKNESS_CH1']
    step_features = []
    thick_inds = []
    ll = len(col)-1
    # 数值截断取每个step段
    for i in range(ll):
        if thickness_col[i] > 0 and thickness_col[i+1] == 0:
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


def scale_deta_thickness(base_thick, cur_thick, scale_reate):
    # zhige: scale 一下 deta_thick, 让function里膜厚调整在lab曲线上显著点
    inds = [0, 2, 3, 4, 5]
    base_cur_diff = [cur_thick[ind] - base_thick[ind] for ind in inds]

    # 这个函数应该还需要精细设计, 现在是简单的做x3处理.
    base_cur_diff = [a*scale_reate for a in base_cur_diff]
    base_cur_diff = [base_cur_diff[0]] + [0] + base_cur_diff[1:] + [0]
    # print("base_cur_diff: ", base_cur_diff)
    modified_thick = [base_cur_diff[k] + base_thick[k] for k in range(7)]
    # print("modified_thick: ", modified_thick)

    return modified_thick


def generate_data(ds, important_inds, x_data_file, y_data_file, sensor_thick_2_OvenNo, OvenNoPair_Labs):
    if os.path.exists(x_data_file):
        return np.load(x_data_file), np.load(y_data_file)
    else:
        sensor_and_thick2OvenNo = dict()
        OvenNos_pairs = dict()
        X = []
        Y = []
        scaled_thicknesss = []
        for i in range(4):
            data = ds[i]
            ll = len(ds[i])
            for j in range(ll-1):
                sensor_file_name1, sensor_file_name2 = data.iloc[j]['FileID'][3:], data.iloc[j+1]['FileID'][3:]
                # 控制炉号连续, diff<=3
                if int(sensor_file_name2) - int(sensor_file_name1) <= 3:
                    sensor1_csv_path = data.iloc[j]['path_sensor_data']
                    sensor1_data = pd.read_csv(sensor1_csv_path)
                    sensor_data1 = sensor1_data['ACT_O1_QCMS_THICKNESS_CH1']
                    # pre_sensor
                    pre_sensor_feature = get_sensor_feature(sensor_data1, sensor=sensor1_data)
                    # pre_thickness
                    pre_thickness = [float(strip_a(a)) for a in data.iloc[j]['Thickness'].split(',')]
                    # cur_thickness
                    cur_thickness = [float(strip_a(a)) for a in data.iloc[j+1]['Thickness'].split(',')]
                    scale_reate = 1
                    scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness, scale_reate)
                    cur_function = function_thick2lab(scaled_thickness)
                    x = pre_sensor_feature + slim_thickness(cur_thickness)
                    assert len(x) == 7+5
                    # 滤波处理下~
                    cur_lab = [float(strip_a(a)) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                    cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                    pre_lab = [float(strip_a(a)) for a in data.iloc[j]['single_lab_curve'].split(',')]
                    pre_lab = savgol_filter(pre_lab, 15, 5).tolist()
                    # plt.plot([i*5+380 for i in range(81)], pre_lab)
                    # plt.show()
                    # Y: 重要频段的, deta(f, m)
                    y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in important_inds]
                    # 410频段处判断异常样本, 并剔除
                    if cur_lab[6] > 4:
                        print('异常样本')
                    else:
                        X.append(x)
                        # 12dims的x作为key, ovenno1_ovenno2炉号作为value
                        sensor_and_thick2OvenNo[''.join(str(np.round(a, 3)) for a in x)] = "{}_{}".format(data.iloc[j]['OvenNo'], data.iloc[j+1]['OvenNo'])
                        OvenNos_pairs["{}_{}".format(data.iloc[j]['OvenNo'], data.iloc[j+1]['OvenNo'])] = [pre_lab, cur_lab]
                        Y.append(y_important_inds)
                        scaled_thicknesss.append(scaled_thickness)

        # 落盘没有normalize的x和炉号之间的关联dict
        data = json.dumps(sensor_and_thick2OvenNo)
        with open(sensor_thick_2_OvenNo, 'w') as js_file:
            js_file.write(data)
        # 落盘连续的炉号和对应的曲线
        data = json.dumps(OvenNos_pairs)
        with open(OvenNoPair_Labs, 'w') as js_file:
            js_file.write(data)

        # 统计x的均值方差
        mean_ = np.mean(X, axis=0)
        std_ = np.std(X, axis=0)
        ff = open('./X_mean_std.txt', 'w')
        ff.write(''.join(str(a) + ',' for a in mean_))
        ff.write('\n')
        ff.write(''.join(str(a) + ',' for a in std_))
        ff.close()

        ff = open('./X_mean_std.txt', 'r').readlines()
        mean_ = [float(a) for a in ff[0].split(',')[:-1]]
        std_ = [float(b) for b in ff[1].split(',')[:-1]]
        print("base x: {}".format(X[0]))
        X = [[(x[k] - mean_[k]) / std_[k] for k in range(12)] for x in X]
        # X = [x[:8] + [0] + x[8:] + [0] for x in X]
        print("normalized x: {}".format(X[0]))

        X, Y = np.array(X), np.array(Y)
        np.save(x_data_file, X)
        np.save(y_data_file, Y)
        np.save('./scaled_thickness.npy', np.array(scaled_thicknesss))

        # check x key 没有重复
        assert len(sensor_and_thick2OvenNo) == X.shape[0]

        return X, Y


def generate_data_plus(sensor_cols, ds, important_inds, x_data_file, y_data_file, sensor_thick_2_OvenNo, OvenNoPair_Labs):
    if os.path.exists(x_data_file):
        return np.load(x_data_file), np.load(y_data_file)
    else:
        sensor_and_thick2OvenNo = dict()
        OvenNos_pairs = dict()
        X = []
        Y = []
        scaled_thicknesss = []
        for i in range(4):
            data = ds[i]
            ll = len(ds[i])
            for j in range(ll-1):
                sensor_file_name1, sensor_file_name2 = data.iloc[j]['FileID'][3:], data.iloc[j+1]['FileID'][3:]
                if int(sensor_file_name2) - int(sensor_file_name1) <= 3:
                    sensor1_csv_path = data.iloc[j]['path_sensor_data']
                    sensor1_data = pd.read_csv(sensor1_csv_path)
                    pre_sensor_feature = []
                    for sensor_col in sensor_cols:
                        sensor_data1 = sensor1_data[sensor_col]
                        pre_sensor_feature.extend(get_sensor_feature(sensor_data1, sensor=sensor1_data))
                    # pre_thickness
                    pre_thickness = [float(strip_a(a)) for a in data.iloc[j]['Thickness'].split(',')]
                    # cur_thickness
                    cur_thickness = [float(strip_a(a)) for a in data.iloc[j+1]['Thickness'].split(',')]
                    scale_reate = 1
                    scaled_thickness = scale_deta_thickness(pre_thickness, cur_thickness, scale_reate)
                    cur_function = function_thick2lab(scaled_thickness)
                    x = pre_sensor_feature + slim_thickness(cur_thickness)
                    assert len(x) == 7*len(sensor_cols)+5
                    # 滤波处理下~
                    cur_lab = [float(strip_a(a)) for a in data.iloc[j + 1]['single_lab_curve'].split(',')]
                    cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                    pre_lab = [float(strip_a(a)) for a in data.iloc[j]['single_lab_curve'].split(',')]
                    pre_lab = savgol_filter(pre_lab, 15, 5).tolist()
                    # plt.plot([i*5+380 for i in range(81)], pre_lab)
                    # plt.show()
                    # Y: 重要频段的, deta(f, m)
                    y_important_inds = [cur_function[important_ind] - cur_lab[important_ind] for important_ind in important_inds]
                    # 410频段处判断异常样本, 并剔除
                    if cur_lab[6] > 4:
                        print('异常样本')
                    else:
                        tmp = np.array(x)
                        tmp[np.isnan(tmp)] = 0
                        x = tmp.tolist()
                        X.append(x)
                        sensor_and_thick2OvenNo[''.join(str(np.round(a, 3)) for a in x)] = "{}_{}".format(data.iloc[j]['OvenNo'], data.iloc[j+1]['OvenNo'])
                        OvenNos_pairs["{}_{}".format(data.iloc[j]['OvenNo'], data.iloc[j+1]['OvenNo'])] = [pre_lab, cur_lab]
                        Y.append(y_important_inds)
                        scaled_thicknesss.append(scaled_thickness)

        # 落盘没有normalize的x和炉号之间的关联dict
        data = json.dumps(sensor_and_thick2OvenNo)
        with open(sensor_thick_2_OvenNo, 'w') as js_file:
            js_file.write(data)
        # 落盘连续的炉号和对应的曲线
        data = json.dumps(OvenNos_pairs)
        with open(OvenNoPair_Labs, 'w') as js_file:
            js_file.write(data)

        # 统计x的均值方差
        mean_ = np.mean(X, axis=0)
        std_ = np.std(X, axis=0)
        std_ = [a+1e-6 for a in std_]
        ff = open('./X_mean_std.txt', 'w')
        ff.write(''.join(str(a) + ',' for a in mean_))
        ff.write('\n')
        ff.write(''.join(str(a) + ',' for a in std_))
        ff.close()

        ff = open('./X_mean_std.txt', 'r').readlines()
        mean_ = [float(a) for a in ff[0].split(',')[:-1]]
        std_ = [float(b) for b in ff[1].split(',')[:-1]]
        print("base x: {}".format(X[0]))
        X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(mean_))] for x in X]
        print("normalized x: {}".format(X[0]))

        X, Y = np.array(X), np.array(Y)
        np.save(x_data_file, X)
        np.save(y_data_file, Y)
        np.save('./scaled_thickness.npy', np.array(scaled_thicknesss))

        # check x key 没有重复
        assert len(sensor_and_thick2OvenNo) == X.shape[0]

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


def overfiting(X, Y):
    dfull = xgb.DMatrix(X, Y)

    param1 = json.load(open(r'./parameter_{}_152.json', 'r'))
    num_round = 200

    cvresult1 = xgb.cv(param1, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()


def inference(cx_lab_curve, pre_thickness, xgb_models, pre_sensor_data, sensor_data, important_inds):
    print("inference: ")
    y_outputs = []
    for xgb_model in xgb_models:
        # x_input: pre_sensor_feature + cur_thicikness
        pre_sensor_feature = get_sensor_feature(pre_sensor_data, sensor=sensor_data)
        x_input = pre_sensor_feature + slim_thickness(pre_thickness)
        print("before normalized x: {}".format(x_input))

        ff = open('./X_mean_std.txt', 'r').readlines()
        mean_ = [float(a) for a in ff[0].split(',')[:-1]]
        std_ = [float(b) for b in ff[1].split(',')[:-1]]
        x_input = [(x_input[k] - mean_[k]) / std_[k] for k in range(12)]
        print("normalized x: {}".format(x_input))

        y_output = xgb_model.predict(np.array([x_input]))[0]
        y_outputs.append(y_output)

    assert len(y_outputs) == len(important_inds)

    # search thickness
    finetune_thre1 = [0.1, 0.1, 0.8, 1.3, 1.3]

    def get_y1(thickness_list):
        recommend_thickness = [thickness_list[0]] + [4] + thickness_list[1:] + [35]
        scale_reate = 1
        scaled_thickness = scale_deta_thickness(pre_thickness, recommend_thickness, scale_reate)
        cur_function = function_thick2lab(scaled_thickness)
        deta_function_and_best = [cur_function[important_ind] - cx_lab_curve[important_ind] for important_ind in important_inds]

        return deta_function_and_best


    def evaluate_config(config):
        y1 = get_y1([config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
        metric = mean_squared_error(y1, y_outputs)
        # print(y1, y_outputs)
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
                                 'x2': pre_thickness[1],
                                 'x3': pre_thickness[2],
                                 'x4': pre_thickness[3],
                                 'x5': pre_thickness[4]},  # a initial (partial) config with low cost
        metric='metric',  # the name of the metric used for optimization
        mode='min',  # the optimization mode, 'min' or 'max'
        num_samples=-1,  # the maximal number of configs to try, -1 means infinite
        time_budget_s=60,  # the time budget in seconds 优化时间限制, 60s停止优化
        local_dir='logs/',  # the local directory to store logs
    )

    print(analysis.best_trial.last_result)  # the best trial's result



def fix_pre_sensor_modify_thickness(x_test, important_inds, save_dir, nums=20):
    xgb_models = []
    for i in range(11):
        params = json.load(open(r'./parameter_{}_152.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./xgb_{}.model'.format(i))
        xgb_models.append(model)

    # 手动测试nums条样本
    x_test = x_test.tolist()[:nums]
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test_no_normalize = [[test[r]*std_[r]+mean_[r] for r in range(12)] for test in x_test]

    for pre_sensor_test_thick in x_test_no_normalize:
        print(pre_sensor_test_thick[7:])

    # [0]*7, 表示pre_sensor部分固定住不变~
    modified = [0]*7 + [0, 0, 0, 0.8, 1]
    x_test_modified = [[test_[i]+modified[i] for i in range(12)] for test_ in x_test_no_normalize]
    x_test_modified = [[(x[k] - mean_[k]) / std_[k] for k in range(12)] for x in x_test_modified]
    preds = []
    preds_ = []
    for xgb_model in xgb_models:
        x_test = np.array(x_test)
        x_test_modified = np.array(x_test_modified)
        pred = xgb_model.predict(x_test)
        pred_ = xgb_model.predict(x_test_modified)
        preds.append(pred)
        preds_.append(pred_)
    preds = [[preds[k][j] for k in range(11)] for j in range(nums)]
    preds_ = [[preds_[k][j] for k in range(11)] for j in range(nums)]
    tmp = [0, 0, 0, 0, 1, 1, 0]

    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    for i in range(nums):
        base_thick = x_test_no_normalize[i][7:]
        base_thick = [base_thick[0]] + [4] + base_thick[1:] + [35]
        modified_thick = [base_thick[k] + tmp[k] for k in range(7)]
        function1 = function_thick2lab(base_thick)
        function1 = [function1[k] for k in important_inds]
        function2 = function_thick2lab(modified_thick)
        function2 = [function2[k] for k in important_inds]
        A = [function1[k] - preds[i][k] for k in range(11)]
        B = [function2[k] - preds_[i][k] for k in range(11)]
        plt.plot(aa, A, color='pink', label='base')
        plt.plot(aa, B, color='blue', label='modified')
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(i)))
        plt.close()
        # plt.show()


def hope_ycy(x_test, y_test, important_inds, sensor_thick_2_OvenNo, OvenNoPair_Labs, save_dir):
    xgb_models = []
    for i in range(11):
        params = json.load(open(r'./parameter_{}_152.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./xgb_{}.model'.format(i))
        xgb_models.append(model)

    # 手动测试nums条样本
    x_test = x_test.tolist()
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test_no_normalize = [[test[r]*std_[r]+mean_[r] for r in range(12)] for test in x_test]
    sensor_thick_2_OvenNo_data = json.load(open(sensor_thick_2_OvenNo, 'r'))
    OvenNoPair_Lab_data= json.load(open(OvenNoPair_Labs, 'r'))


    pre_sensor_test_thicks = []
    for pre_sensor_test_thick in x_test_no_normalize:
        pre_sensor_test_thicks.append([pre_sensor_test_thick[7]] + [4] + pre_sensor_test_thick[8:] + [35])

    deta_y_truth = [[y_test[k][j] for k in range(11)] for j in range(len(x_test))]
    functions = [function_thick2lab(thick) for thick in pre_sensor_test_thicks]
    functions = [[f[ind] for ind in important_inds] for f in functions]
    # 真实机台的测试集曲线
    y_truths = [[functions[k][j]-deta_y_truth[k][j] for j in range(11)] for k in range(len(x_test))]
    y_truths = y_truths[1:]

    lls = len(x_test)
    xs = []
    for i in range(1, lls):
        pre_sensor = x_test[i-1][:7]
        next_thick = x_test[i][7:]
        x = pre_sensor + next_thick
        xs.append(x)
    y_preds = []
    for xgb_model in xgb_models:
        pred = xgb_model.predict(np.array(xs))
        y_preds.append(pred)
    deta_y_preds = [[y_preds[k][j] for k in range(11)] for j in range(lls-1)]
    functions_ = functions[1:]
    y_model_pred = [[functions_[k][j]-deta_y_preds[k][j] for j in range(11)] for k in range(lls-1)]
    assert len(y_model_pred) == lls-1
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    aa_all = [380+5*i for i in range(81)]
    for i in range(lls-1):
        no_normalized_x = ''.join(str(np.round(a, 3)) for a in x_test_no_normalize[i])
        OvenNo_pair = sensor_thick_2_OvenNo_data[no_normalized_x]
        pre_cur_labs = OvenNoPair_Lab_data[OvenNo_pair]
        plt.plot(aa, y_model_pred[i], color='pink', label='pre_sensor+set_next_thickness_model_pred')
        plt.plot(aa, y_truths[i], color='blue', label='pre_sensor+cur_thickness_model_inference')
        plt.plot(aa_all, pre_cur_labs[0], color='yellow', label='pre_real_curve')
        plt.plot(aa_all, pre_cur_labs[1], color='cornflowerblue', label='cur_real_curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(OvenNo_pair)))
        plt.close()
        # plt.show()



def eval_test_data(x_test, important_inds, y_tests, title='train data'):
    xgb_models = []
    for i in range(11):
        params = json.load(open(r'./parameter_{}_152.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./xgb_{}.model'.format(i))
        xgb_models.append(model)
    preds = []
    for xgb_model in xgb_models:
        pred = xgb_model.predict(x_test)
        preds.append(pred)
    preds = [[preds[k][j] for k in range(11)] for j in range(x_test.shape[0])]
    gt = [[y_tests[k][j] for k in range(11)] for j in range(x_test.shape[0])]
    x_test = x_test.tolist()
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test = [[x[r]*std_[r] + mean_[r] for r in range(7, 12)] for x in x_test]
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

    #     y_output_ = xgb_model.predict(np.array([x_modified]))[0]
    #     y_outputs_.append(y_output_)
    #
    # y_outputs = [function_lab1[k] - y_outputs[k] for k in range(11)]
    # y_outputs_ = [function_lab2[k] - y_outputs_[k] for k in range(11)]



def Test(x_test, slim_best_cx_curve, important_inds, sensor_thick_2_OvenNo, OvenNoPair_Labs, save_dir):
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    xgb_models = []
    for i in range(11):
        params = json.load(open(r'./parameter_{}_152.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./xgb_{}.model'.format(i))
        xgb_models.append(model)

    x_test = x_test.tolist()
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test_no_normalize = [[test[r] * std_[r] + mean_[r] for r in range(12)] for test in x_test]
    sensor_thick_2_OvenNo_data = json.load(open(sensor_thick_2_OvenNo, 'r'))
    OvenNoPair_Lab_data = json.load(open(OvenNoPair_Labs, 'r'))

    pre_sensor_test_thicks = []
    for pre_sensor_test_thick in x_test_no_normalize:
        pre_sensor_test_thicks.append([pre_sensor_test_thick[7]] + [4] + pre_sensor_test_thick[8:] + [35])

    for iindex, single_test_data in enumerate(x_test):
        oven1_oven2 = sensor_thick_2_OvenNo_data[''.join(str(np.round(a, 3)) for a in x_test_no_normalize[iindex])]
        real_curve_slim = [OvenNoPair_Lab_data[oven1_oven2][1][r] for r in important_inds]
        preds_dim11 = []
        for xgb_model in xgb_models:
            preds_dim11.append(xgb_model.predict(np.array([single_test_data]))[0])
        assert len(preds_dim11) == len(important_inds)

        pre_thickness = [x_test_no_normalize[iindex][7:][0]] + [4] + x_test_no_normalize[iindex][7:][1:] + [35]
        # search thickness
        finetune_thre1 = [0.1, 0.1, 0.8, 1.3, 1.3]

        def get_y1(thickness_list):
            recommend_thickness = [thickness_list[0]] + [4] + thickness_list[1:] + [35]
            scale_reate = 1
            scaled_thickness = scale_deta_thickness(pre_thickness, recommend_thickness, scale_reate)
            cur_function = function_thick2lab(scaled_thickness)
            deta_function_and_best = [cur_function[important_ind] - cx_lab_curve[important_ind] for important_ind in important_inds]

            return deta_function_and_best

        def evaluate_config(config):
            y1 = get_y1([config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
            metric = mean_squared_error(y1, preds_dim11)
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
                                     'x2': pre_thickness[1],
                                     'x3': pre_thickness[2],
                                     'x4': pre_thickness[3],
                                     'x5': pre_thickness[4]},  # a initial (partial) config with low cost
            metric='metric',  # the name of the metric used for optimization
            mode='min',  # the optimization mode, 'min' or 'max'
            num_samples=-1,  # the maximal number of configs to try, -1 means infinite
            time_budget_s=60,  # the time budget in seconds 优化时间限制, 60s停止优化
            local_dir='logs/',  # the local directory to store logs
        )

        search_res = analysis.best_trial.last_result
        print(search_res)
        # 'config': {'x1': 24.84106057156333, 'x2': 11.639999999999999, 'x3': 35.52, 'x4': 101.90757651341296, 'x5': 93.53385108232484}
        thickness_res = [np.round(search_res['config']['x1'], 2), 4, np.round(search_res['config']['x2'], 2),np.round(search_res['config']['x3'], 2),np.round(search_res['config']['x4'], 2),np.round(search_res['config']['x5'], 2), 35]
        plt.plot(aa, real_curve_slim, color='pink', label='cur_curve: {}'.format(''.join(str(a)+', ' for a in pre_thickness)))
        plt.plot(aa, slim_best_cx_curve, color='cornflowerblue', label='to get best_curve: {}'.format(''.join(str(a)+', ' for a in thickness_res)))
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(oven1_oven2.split('_')[1])))
        # plt.show()
        plt.close()



# 11.3 night
def compare_pred_realCurve(x_test, scaled_thickness, save_dir):
    aa = [important_ind * 5 + 380 for important_ind in important_inds]
    xgb_models = []
    for i in range(11):
        params = json.load(open(r'./parameter_{}_152.json'.format(i), 'r'))
        model = xgb.XGBRegressor(objective="reg:linear", **params)
        model.load_model(r'./xgb_{}.model'.format(i))
        xgb_models.append(model)
    x_test = x_test.tolist()
    ff = open('./X_mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    x_test_no_normalize = [[test[r] * std_[r] + mean_[r] for r in range(len(mean_))] for test in x_test]
    sensor_thick_2_OvenNo_data = json.load(open(sensor_thick_2_OvenNo, 'r'))
    OvenNoPair_Lab_data = json.load(open(OvenNoPair_Labs, 'r'))

    for iindex, single_test_data in enumerate(x_test):
        single_preds = []
        oven1_oven2 = sensor_thick_2_OvenNo_data[''.join(str(np.round(a, 3)) for a in x_test_no_normalize[iindex])]
        real_curve_slim = [OvenNoPair_Lab_data[oven1_oven2][1][r] for r in important_inds]
        for xgb_model in xgb_models:
            single_pred = xgb_model.predict(np.array([single_test_data]))
            single_preds.append(single_pred)
        # deta(f, m), m = f-deta
        single_f = function_thick2lab(scaled_thickness[iindex])
        single_f = [single_f[i] for i in important_inds]
        pred_m = [single_f[i] - single_preds[i] for i in range(len(important_inds))]
        plt.plot(aa, real_curve_slim, label='real_curve', color='pink')
        plt.plot(aa, pred_m, label='pred_curve', color='cornflowerblue')
        plt.legend()
        plt.savefig(os.path.join(save_dir, './{}.png'.format(iindex)))
        plt.close()


if __name__ == '__main__':

    cx_lab_curve = [6.6524, 4.5071, 2.2922, 1.1451, 0.7214, 0.4108, 0.2415, 0.3834, 0.5492, 0.5964, 0.6647, 0.8838,
                   1.0936, 1.0963, 0.992, 0.972, 1.0551, 1.1061, 1.0686, 0.927, 0.8003, 0.7419, 0.7491, 0.7563, 0.7154,
                   0.6076, 0.4921, 0.4414, 0.4152, 0.4251, 0.4191, 0.3947, 0.3412, 0.2696, 0.2235, 0.1996, 0.1907,
                   0.1965, 0.1844, 0.1592, 0.1281, 0.0889, 0.0598, 0.0446, 0.031, 0.0375, 0.0398, 0.0386, 0.0287,
                   0.0257, 0.0249, 0.0326, 0.0593, 0.093, 0.1355, 0.1803, 0.2255, 0.268, 0.3089, 0.3568, 0.4025, 0.4695,
                   0.5404, 0.6328, 0.7406, 0.8652, 1.0015, 1.1388, 1.2783, 1.4115, 1.543, 1.6709, 1.8066, 1.9292,
                   2.0527, 2.1884, 2.3387, 2.5097, 2.6949, 2.8866, 3.1036]
    cc_lab_curve = [7.3356, 4.8663, 2.5198, 1.4276, 0.8527, 0.4359, 0.3289, 0.4565, 0.5573, 0.6074, 0.8496, 1.0507,
                   1.1332, 1.0802, 1.0985, 1.202, 1.2691, 1.2089, 1.065, 0.9647, 0.9341, 0.9569, 0.9191, 0.8223, 0.6908,
                   0.6092, 0.5654, 0.5678, 0.5585, 0.5084, 0.4354, 0.3652, 0.3153, 0.301, 0.2963, 0.2923, 0.2769,
                   0.2354, 0.1909, 0.158, 0.1339, 0.1295, 0.1184, 0.1111, 0.0976, 0.0863, 0.0688, 0.064, 0.0584, 0.0639,
                   0.0748, 0.0901, 0.1025, 0.1135, 0.1261, 0.1485, 0.1761, 0.2271, 0.2806, 0.3443, 0.4229, 0.5029,
                   0.5719, 0.6471, 0.7214, 0.788, 0.8605, 0.9497, 1.052, 1.1625, 1.2945, 1.4421, 1.6149, 1.7815, 1.947,
                   2.1099, 2.2678, 2.4232, 2.5676, 2.7107, 2.8521]

    # double_curve_check(datas)
    # best_cx_thickness = [24.82, 5, 11.54, 36.32, 101.3, 93.4, 25]
    # best_cc_thickness = [24.12, 5, 12.17, 36.49, 104.6, 96.45, 25]

    sensor_cols = ["ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH", "ACT_V1_THERMOVAC_PREVLINE",
                   "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP",
                   "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE",
                   "ACT_Q10_CURRENT_ANODE", "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE",
                   "ACT_Q10_CURRENT_NEUTRAL", "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "STA_Q10_IONSOURCE_SHUTTER_IOP",
                   "ACT_V1_MEISSNER_POLYCOLDTEMP"]

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    datas = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4)]

    # 11.4 加入所有sensor_col的平稳性特征
    all_sensor_cols = 0

    important_inds = [0, 3, 6, 14, 24, 34, 44, 54, 64, 74, 80]
    x_data_file = r'./X_7.npy'
    y_data_file = r'./Y_7.npy'
    sensor_thick_2_OvenNo = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\cx_data_result\sensor_thick_2_oveNno.json'
    OvenNoPair_Labs = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\cx_data_result\OvenNoPair_Labs.json'
    if not all_sensor_cols:
        X, Y = generate_data(datas, important_inds, x_data_file, y_data_file, sensor_thick_2_OvenNo, OvenNoPair_Labs)
    else:
        x_data_file = r'./X_152.npy'
        y_data_file = r'./Y_152.npy'
        X, Y = generate_data_plus(sensor_cols, datas, important_inds, x_data_file, y_data_file, sensor_thick_2_OvenNo, OvenNoPair_Labs)
    print(X.shape, Y.shape)
    Y = Y.tolist()

    xgb_models = []
    y_tests = []
    y_trains = []
    for ind, important_ind in enumerate(important_inds):
        y = np.array([[Y_[ind]] for Y_ in Y])
        # random_state seed固定, 保证每个频段的训测数据拆分一致
        X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=33)
        y_tests.append([a[0] for a in y_test])
        y_trains.append([a[0] for a in y_train])
        if ind == 0:
            np.save(r'./x_test_152.npy', X_test)
            np.save(r'./x_train_152.npy', X_train)

        # train and eval model
        hyperparameter_searching(X, y, ind)
        # overfiting(X, Y)

        xgb_model = cross_val(X_train, y_train, X_test, y_test, ind, cx_lab_curve, important_inds)
        xgb_models.append(xgb_model)

    # eval test_data
    x_test = np.load(r'./x_train_152.npy')
    # eval_test_data(x_test, important_inds, y_trains, title='train data')
    # x_test = np.load(r'./x_test_152.npy')
    # eval_test_data(x_test, important_inds, y_tests, title='test data')

    # 固定pre_sensor部分, 手动调整膜厚, 查看前后曲线是否变化, 是否符合生产经验..
    # save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\mix_data\固定sensor_feature手动调膜厚'
    # fix_pre_sensor_modify_thickness(x_test, important_inds, save_dir, nums=x_test.shape[0])

    # 在测试集中, 找变化明显的thickness变化对, 膜厚设置成下一炉的值, 看看模型预测的曲线, 是不是接近真实的机台曲线
    # save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\cc_pre_sensor_cur_thickness2deta_fm_curve\固定sensor_feature-设置为下一炉的膜厚值'
    # hope_ycy(x_test, y_tests, important_inds, sensor_thick_2_OvenNo, OvenNoPair_Labs, save_dir)


    # 1.Inference
    # pre_thickness = [24.82, 5.0, 11.54, 36.32, 101.8, 94.0, 25.0]
    # pre_sensor = pd.read_csv(r'D:\work\project\卡尔蔡司AR镀膜\正背面数据\9.11~9.17\9.11~9.17\机台文件\21091235.csv')
    # pre_sensor_data = pre_sensor['ACT_O1_QCMS_THICKNESS_CH1']
    # inference(cx_lab_curve, pre_thickness, xgb_models, pre_sensor_data, pre_sensor, important_inds)

    # 2.测试数据, 输出膜厚推优结果给蔡司review下~
    # save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\1028_li_gong_review'
    # best_cx_curve = [6.6524, 4.5071, 2.2922, 1.1451, 0.7214, 0.4108, 0.2415, 0.3834, 0.5492, 0.5964, 0.6647, 0.8838, 1.0936, 1.0963, 0.992, 0.972, 1.0551, 1.1061, 1.0686, 0.927, 0.8003, 0.7419, 0.7491, 0.7563, 0.7154, 0.6076, 0.4921, 0.4414, 0.4152, 0.4251, 0.4191, 0.3947, 0.3412, 0.2696, 0.2235, 0.1996, 0.1907, 0.1965, 0.1844, 0.1592, 0.1281, 0.0889, 0.0598, 0.0446, 0.031, 0.0375, 0.0398, 0.0386, 0.0287, 0.0257, 0.0249, 0.0326, 0.0593, 0.093, 0.1355, 0.1803, 0.2255, 0.268, 0.3089, 0.3568, 0.4025, 0.4695, 0.5404, 0.6328, 0.7406, 0.8652, 1.0015, 1.1388, 1.2783, 1.4115, 1.543, 1.6709, 1.8066, 1.9292, 2.0527, 2.1884, 2.3387, 2.5097, 2.6949, 2.8866, 3.1036]
    # slim_best_cx_curve = [best_cx_curve[i] for i in important_inds]
    # Test(x_test, slim_best_cx_curve, important_inds, sensor_thick_2_OvenNo, OvenNoPair_Labs, save_dir)

    #3. model_pred and real_cuvre check
    save_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS-dev\data\model_and_real_7_train'
    scaled_thickness = np.load(r'./scaled_thickness.npy')
    compare_pred_realCurve(x_test, scaled_thickness, save_dir)


