# coding=utf-8

import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
from scipy.signal import savgol_filter
from function_thick2lab import function_thick2lab
from pass_utils import *
import matplotlib.pyplot as plt
from flaml import tune
sys.path.append("..")
from tsf_feature_extraction import generate_tsf_feature


def generate_data(data, train_sensor_feature):
    '''
    model1: thick1 = thick2
    model2: thick1 != thick2

    '''

    model1_x, model1_y = [], []
    model2_x, model2_y = [], []
    ll = len(data)
    for i in range(ll-1):
        evt_name1, evt_name2 = data.iloc[i]['FileID'][3:], data.iloc[i + 1]['FileID'][3:]
        evt_name2 = int(evt_name2[-3:])
        evt_name1 = int(evt_name1[-3:])
        cccx1, cccx2 = data.iloc[i]['CCCX'], data.iloc[i + 1]['CCCX']
        type1, type2 = data.iloc[i]['Type'], data.iloc[i + 1]['Type']
        if (cccx1 == cccx2) and (type1 == type2) and (evt_name2 - evt_name1 <= 2) and ('1.67' not in type1):
            # ligong: "1.6和1.67不太一样" 可能是膜厚带来的曲线变化趋势and幅度,不那么相同..
            # print(cccx1, evt_name2, evt_name1, type1)
            # evt_name相差<=2为前后炉数据对, 并且近视老花同类型, cxcc正背面一致
            pre_oven = str(data.iloc[i]['OvenNo'])
            cur_oven = str(data.iloc[i+1]['OvenNo'])
            if (pre_oven not in bad_ovens) and (cur_oven not in bad_ovens):
                thick1 = [float(a.strip("['] ")) for a in data.iloc[i]['Thickness'].split(',')]
                thick2 = [float(a.strip("['] ")) for a in data.iloc[i + 1]['Thickness'].split(',')]
                try:
                    pre_sensor_feature = train_sensor_feature[pre_oven]
                except:
                    print('bad data pair~')
                    continue
                pre_lab = [float(a.strip("['] ")) for a in data.iloc[i]['single_lab_curve'].split(',')]
                cur_lab = [float(a.strip("['] ")) for a in data.iloc[i + 1]['single_lab_curve'].split(',')]
                pre_lab = savgol_filter(pre_lab, 15, 5).tolist()
                cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                pre_lab = [max(0, a) for a in pre_lab]
                cur_lab = [max(0, a) for a in cur_lab]
                if is_thick_modified(thick2, thick1):
                    # 前后炉膜厚设置有变化
                    x_for_model1 = pre_lab + pre_sensor_feature
                    thick1_function = function_thick2lab(thick1)
                    thick2_function = function_thick2lab(thick2)
                    slim_thick1_function = [thick1_function[ind] for ind in important_inds]
                    slim_cur_lab = [cur_lab[ind] for ind in important_inds]
                    model2_x.append([x_for_model1, slim_thick1_function, slim_cur_lab])
                    slim_deta_function = [thick2_function[ind] - thick1_function[ind] for ind in important_inds]
                    model2_y.append([slim_deta_function, thick1 + thick2])
                else:
                    # 前后炉膜厚设置无变化
                    x = pre_lab + pre_sensor_feature
                    model1_x.append(x)
                    function_lab = function_thick2lab(thick1)
                    slim_y_minus = [function_lab[ind] - cur_lab[ind] for ind in important_inds]
                    model1_y.append(slim_y_minus)

    # 对model1_x做归一化
    x_mean = np.mean(model1_x, axis=0)
    x_std = np.std(model1_x, axis=0)
    x_std = [a + 1e-6 for a in x_std]
    l = len(x_std)
    model1_x = [[(x[i] - x_mean[i]) / x_std[i] for i in range(l)] for x in model1_x]

    ff = open('./model1_x_mean_std{}{}.txt'.format(seed, pca), 'w')
    for x_mean_ in x_mean:
        ff.write(str(x_mean_) + ',')
    ff.write('\n')
    for x_std_ in x_std:
        ff.write(str(x_std_) + ',')
    ff.write('\n')

    model1_y = np.array(model1_y)
    model2_y = np.array(model2_y)
    model1_x = np.array(model1_x)
    model2_x = np.array(model2_x)
    print("model1 data_size: {}, model2 data_size: {}".format(model1_x.shape[0], model2_x.shape[0]))
    np.save('./model1_x_seed{}_pca{}.npy'.format(seed, pca), model1_x)
    np.save('./model1_y_seed{}_pca{}.npy'.format(seed, pca), model1_y)
    np.save('./model2_x_seed{}_pca{}.npy'.format(seed, pca), model2_x)
    np.save('./model2_y_seed{}_pca{}.npy'.format(seed, pca), model2_y)

    return model1_x, model1_y, model2_x, model2_y


def generate_online_test_data(online_test_data, test_sensor_feature):
    test_len = len(online_test_data)
    test_x = []
    for i in range(test_len):
        oven = str(online_test_data.iloc[i]['OvenNo'])
        lab = [float(a.strip("['] ")) for a in online_test_data.iloc[i]['single_lab_curve'].split(',')]
        lab = savgol_filter(lab, 15, 5).tolist()
        lab = [max(0, a) for a in lab]
        thick = [float(a.strip("['] ")) for a in online_test_data.iloc[i]['Thickness'].split(',')]
        try:
            sensor_feature = test_sensor_feature[oven]
        except:
            continue
        function_lab = function_thick2lab(thick)
        slim_function_lab = [function_lab[ind] for ind in important_inds]
        test_x.append([lab + sensor_feature, slim_function_lab, thick, oven])

    np.save('./test_x_seed{}_pca{}.npy'.format(seed, pca), np.array(test_x))

    return test_x


def normalize_model1_x(model1_x):
    ff = open('./model1_x_mean_std{}{}.txt'.format(seed, pca), 'r').readlines()
    x_mean = ff[0].split(',')[:-1]
    x_mean = [float(a) for a in x_mean]
    x_std = ff[1].split(',')[:-1]
    x_std = [float(a) for a in x_std]
    l = len(x_std)
    model1_x = [[(x[i] - x_mean[i]) / x_std[i] for i in range(l)] for x in model1_x]
    model1_x = np.array(model1_x)

    return model1_x


def generate_forward_check_data(all_train_data, train_sensor_feature):
    '''
    分别生成model1,2 forward_check 所需的data
    '''
    model1_x, model1_y = [], []
    model2_x, model2_y = [], []
    for data in all_train_data:
        ll = len(data)
        for i in range(ll-1):
            evt_name1, evt_name2 = data.iloc[i]['FileID'][3:], data.iloc[i + 1]['FileID'][3:]
            evt_name2 = int(evt_name2[-3:])
            evt_name1 = int(evt_name1[-3:])
            if evt_name2 - evt_name1 <= 3:
                pre_oven = str(data.iloc[i]['OvenNo'])
                cur_oven = str(data.iloc[i+1]['OvenNo'])
                if (pre_oven not in bad_ovens) and (cur_oven not in bad_ovens):
                    thick1 = [float(a.strip("['] ")) for a in data.iloc[i]['Thickness'].split(',')]
                    thick2 = [float(a.strip("['] ")) for a in data.iloc[i + 1]['Thickness'].split(',')]
                    try:
                        pre_sensor_feature = train_sensor_feature[pre_oven]
                    except:
                        continue
                    pre_lab = [float(a.strip("['] ")) for a in data.iloc[i]['single_lab_curve'].split(',')]
                    cur_lab = [float(a.strip("['] ")) for a in data.iloc[i + 1]['single_lab_curve'].split(',')]
                    pre_lab = savgol_filter(pre_lab, 15, 5).tolist()
                    cur_lab = savgol_filter(cur_lab, 15, 5).tolist()
                    pre_lab = [max(0, a) for a in pre_lab]
                    cur_lab = [max(0, a) for a in cur_lab]
                    slim_cur_lab = [cur_lab[ind] for ind in important_inds]
                    if is_thick_modified(thick2, thick1):
                        # 前后炉膜厚设置有变化
                        x_for_model1 = pre_lab + pre_sensor_feature
                        thick1_function = function_thick2lab(thick1)
                        thick2_function = function_thick2lab(thick2)
                        slim_thick1_function = [thick1_function[ind] for ind in important_inds]
                        model2_x.append([x_for_model1, slim_thick1_function])
                        slim_deta_function = [thick2_function[ind] - thick1_function[ind ]for ind in important_inds]
                        model2_y.append([slim_deta_function, thick1 + thick2, slim_cur_lab, cur_oven])
                    else:
                        # 前后炉膜厚设置无变化
                        x = pre_lab + pre_sensor_feature
                        model1_x.append(x)
                        function_lab = function_thick2lab(thick1)
                        slim_function_lab = [function_lab[ind] for ind in important_inds]
                        model1_y.append([slim_function_lab, slim_cur_lab, cur_oven])

    # 对model1_x做归一化
    model1_x = normalize_model1_x(model1_x)
    print('generate_forward_check_data done ~')

    return model1_x, model1_y, model2_x, model2_y


def train_model1(X, model1_y, ff_train):
    '''
    x = pre_lab + pre_sensor_feature
    model1_x.append(x)
    y_minus = [function_lab[k] - cur_lab[k] for k in range(lab_lens)]
    slim_y_minus = [y_minus[ind] for ind in important_inds]
    model1_y.append(slim_y_minus)

    '''

    ll = len(important_inds)
    Y = model1_y.tolist()
    model1s = []
    for i in range(ll):
        y = np.array([[Y_[i]] for Y_ in Y])
        param_file = r'./parameter_{}_index{}_seed{}_pca{}_minus.json'.format('model1', i, seed, pca)
        if not os.path.exists(param_file):
            xgb_hyperparameter_searching(X, y, param_file)
        model1 = cross_val(X, y, i, seed, pca, ff_train, param_file, model_flag='model1')
        
        # import xgboost as xgb
        # import pickle
        # xgb_model = pickle.load(open("avictcApplXgModel_20200213.pkl", "rb"))
        # digraph = xgb.to_graphviz(model1, num_trees=0)
        # digraph.format = 'png'
        # digraph.view('./xgb_demo')

        model1s.append(model1)

    return model1s


def pred_model1(model1s, model1_x):
    y1_inds = []
    model_lens = len(model1s)
    sample_lens = model1_x.shape[0]
    for ind, model1 in enumerate(model1s):
        y1 = model1.predict(model1_x).tolist()
        y1_inds.append(y1)
    samples_y_inds = [[y1_inds[i][j] for i in range(model_lens)] for j in range(sample_lens)]

    return samples_y_inds


def pre_lab_base(model1s, model2_x):
    x_for_model1 = [x[0] for x in model2_x]
    model1_x_normalized = normalize_model1_x(x_for_model1)
    model_lens = len(model1s)
    sample_lens = model1_x_normalized.shape[0]
    # 调用model1的输出
    samples_y_inds = pred_model1(model1s, model1_x_normalized)
    slim_function1s = [x[1] for x in model2_x]
    lab_base = [[slim_function1s[k][i] - samples_y_inds[k][i] for i in range(model_lens)] for k in range(sample_lens)]

    return lab_base



def train_model2(model1s, model2_x, model2_y, ff_train):
    '''
    model2_x.append([x_for_model1, slim_thick1_function, slim_cur_lab])
    model2_y.append([slim_deta_function, thick1 + thick2])

    '''
    model2_x = model2_x.tolist()
    model2_y = model2_y.tolist()

    # model2 input: thick1 + thick2
    thick1_thick2 = [x[1] for x in model2_y]
    x_mean = np.mean(thick1_thick2, axis=0)
    x_std = np.std(thick1_thick2, axis=0)
    x_std = [a+1e-6 for a in x_std]
    l = len(x_std)
    ff = open('./model2_thick12_mean_std.txt', 'w')
    for x_mean_ in x_mean:
        ff.write(str(x_mean_) + ',')
    ff.write('\n')
    for x_std_ in x_std:
        ff.write(str(x_std_) + ',')
    ff.write('\n')
    thick1_thick2_normalized = [[(x[i] - x_mean[i]) / x_std[i] for i in range(l)] for x in thick1_thick2]
    slim_deta_function = [x[0] for x in model2_y]
    model_lens, sample_lens = len(model1s), len(model2_x)
    slim_cur_labs = [x[2] for x in model2_x]
    lab_base = pre_lab_base(model1s, model2_x)
    cur_lab_minus_base_lab = [[slim_cur_labs[k][i] - lab_base[k][i] for i in range(model_lens)] for k in range(sample_lens)]
    y2 = [[slim_deta_function[k][i] - cur_lab_minus_base_lab[k][i] for i in range(model_lens)] for k in range(sample_lens)]

    thick12_y2 = dict()
    for ind, y in enumerate(y2):
        thick12 = thick1_thick2[ind]
        # deta_thick = [thick12[7:][i] - thick12[:7][i] for i in range(7)]
        # print("deta thickness: {}".format(deta_thick))
        # print("deta function: {}".format(slim_deta_function[ind]))
        # print("y2: {}".format(y))
        thick12_y2[''.join(str(a) for a in thick12)] = y
    # print(len(thick12_y2))

    X = np.array(thick1_thick2_normalized)
    ll = len(important_inds)
    model2s = []
    for i in range(ll):
        y = np.array([[Y_[i]] for Y_ in y2])
        param_file = r'./parameter_{}_index{}_seed{}_pca{}_minus.json'.format('model2', i, seed, pca)
        if not os.path.exists(param_file):
            xgb_hyperparameter_searching(X, y, param_file)
        model2 = cross_val(X, y, i, seed, pca, ff_train, param_file, model_flag='model2')
        model2s.append(model2)

    return model2s


def normalize_thick12(thick1_thick2, flag=None):

    ff = open('./model2_thick12_mean_std.txt', 'r').readlines()
    x_mean = ff[0].split(',')[:-1]
    x_mean = [float(a) for a in x_mean]
    x_std = ff[1].split(',')[:-1]
    x_std = [float(a) for a in x_std]
    if flag:
        return [x_mean, x_std]
    else:
        l = len(x_std)
        thick1_thick2 = [[(x[i] - x_mean[i]) / x_std[i] for i in range(l)] for x in thick1_thick2]
        thick1_thick2 = np.array(thick1_thick2)

        return thick1_thick2


def show_model2_detail(thick1_thick2, y2s, deta_function_thick2_minus_thick1):
    for ind, thick12 in enumerate(thick1_thick2):
        diff = [np.round(thick12[7:][i] - thick12[:7][i], 2) for i in range(7)]
        modified_layer = [i for i in range(7) if abs(diff[i]) > 0.05]
        thickness_diff = ''
        for layer in modified_layer:
            thickness_diff += "layer{} modified {}, ".format(layer, diff[layer])
        residual_lab = [deta_function_thick2_minus_thick1[ind][r] - y2s[ind][r] for r in range(14)]
        print("{}, f-m: {}, deta_function_thick2-1: {} \n".format(thickness_diff, y2s[ind], deta_function_thick2_minus_thick1[ind]))
        # residual_lab


def get_model12_forward_lab(model1s, model2_x, model2_y, model2s):
    # get model12_pred_lab
    ll = len(model1s)
    lab_base = pre_lab_base(model1s, model2_x)
    slim_deta_function = [x[0] for x in model2_y]
    thick1_thick2 = [x[1] for x in model2_y]
    sample_lens = len(thick1_thick2)
    thick1_thick2 = normalize_thick12(thick1_thick2, flag=False)
    y2_inds = []
    for ind, model2 in enumerate(model2s):
        y2 = model2.predict(thick1_thick2).tolist()
        y2_inds.append(y2)
    samples_y2_inds = [[y2_inds[i][j] for i in range(ll)] for j in range(sample_lens)]

    # 输出model2的具体"贡献".
    # show_model2_detail(thick1_thick2, samples_y2_inds, slim_deta_function)

    # full_pred_lab = lab_base + (f_thick2-1 - y2)
    pred_cur_lab = [[lab_base[k][r] + slim_deta_function[k][r] - samples_y2_inds[k][r] for r in range(ll)] for k in
                    range(sample_lens)]

    return pred_cur_lab, lab_base


def forward_model_check(save_dir, lab_mse_txt, model1_x, model1_y, model2_x, model2_y, data_flag=None):
    # load model1,2
    ll = len(important_inds)
    model1s, model2s = load_model1_model2(ll, seed, pca)

    # 随机测试train中模型见过的10条数据前向拟合情况
    if data_flag == "train":
        model2_x = model2_x[20: 30]
        model2_y = model2_y[20: 30]
        model1_x = model1_x[20: 30]
        model1_y = model1_y[20: 30]

        # save用于模型预测曲线的趋势验证
        a, b = np.array(model2_x), np.array(model2_y)
        np.save(r'./train_model2_x{}{}.npy'.format(seed, pca), a)
        np.save(r'./train_model2_y{}{}.npy'.format(seed, pca), b)

    # model1 forward test
    samples_y_inds = pred_model1(model1s, model1_x)
    diff = [0]*ll
    sample_lens = len(samples_y_inds)
    for ind, y in enumerate(model1_y):
        slim_f, real_slim_cur_lab, cur_oven = y[0], y[1], y[2]
        pred_cur_lab = [slim_f[k] - samples_y_inds[ind][k] for k in range(ll)]
        plt.title(cur_oven)
        plt.plot([i for i in range(14)], pred_cur_lab, label='model1_pred', color='cornflowerblue')
        plt.plot([i for i in range(14)], real_slim_cur_lab, label='real_lab', color='pink')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "{}.png".format(cur_oven)))
        plt.close()
        diff = [diff[r] + abs(pred_cur_lab[r] - real_slim_cur_lab[r]) for r in range(ll)]
    red_diff = [np.round(a/sample_lens, 3) for a in diff]
    eval_mse = eval_metric(red_diff)
    line = "seed{}pca{}, model1 mean_diff: {}: {}".format(seed, pca, red_diff, eval_mse)
    print(line)
    lab_mse_txt.write(line + '\n')

    # model2 forward test
    thick12s = [x[1] for x in model2_y]
    slim_real_cur_lab = [x[2] for x in model2_y]
    cur_ovens = [x[3] for x in model2_y]
    sample_lens = len(slim_real_cur_lab)
    pred_cur_lab, lab_base = get_model12_forward_lab(model1s, model2_x, model2_y, model2s)
    diff = [0] * ll
    for ind, pred in enumerate(pred_cur_lab):
        thick12 = thick12s[ind]
        deta_thick = [np.round(thick12[7:][i] - thick12[:7][i], 2) for i in range(7)]
        plt.title("pre_cur_thick_diff: {}".format(deta_thick))
        plt.plot([i for i in range(14)], pred, label='model2_pred', color='cornflowerblue')
        plt.plot([i for i in range(14)], lab_base[ind], label='model1_lab_base', color='blue')
        plt.plot([i for i in range(14)], slim_real_cur_lab[ind], label='real_lab', color='pink')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "{}.png".format(cur_ovens[ind])))
        plt.close()
        # plt.show()
    diff = [[diff[r] + abs(slim_real_cur_lab[k][r] - pred_cur_lab[k][r]) for r in range(ll)] for k in range(sample_lens)]
    res_diff = [np.round(a, 3) for a in np.mean(diff, axis=0)]
    eval_mse = eval_metric(res_diff)
    line = "seed{}pca{}, model2 mean_diff: {}: {}".format(seed, pca, res_diff, eval_mse)
    print(line)
    lab_mse_txt.write(line + '\n')


def model_trend_check(save_dir, model2_x_data, model2_y_data):
    aa = [i for i in range(14)]
    if os.path.exists(model2_x_data):
        model2_x = np.load(model2_x_data, allow_pickle=True).tolist()
        model2_y = np.load(model2_y_data, allow_pickle=True).tolist()
        cur_ovens = [x[3] for x in model2_y]
        ll = len(important_inds)
        model1s, model2s = load_model1_model2(ll, seed, pca)
        pred_cur_lab, lab_base = get_model12_forward_lab(model1s, model2_x, model2_y, model2s)
        # 只修改model2的thick2, model1直接调用
        # model2_y.append([slim_deta_function, thick1 + thick2, slim_cur_lab, cur_oven])
        model2_y_ = []
        for y in model2_y:
            tmp = y[1]
            diffs = [0] * 7 + [0, 0, 0, 0, 1, 0, 0]
            tmp = [tmp[r] + diffs[r] for r in range(14)]
            y_ = [y[0]] + [tmp] + y[2:]
            model2_y_.append(y_)
        pred_cur_lab_modified, _ = get_model12_forward_lab(model1s, model2_x, model2_y_, model2s)
        for ind, oven in enumerate(cur_ovens):
            plt.plot(aa, pred_cur_lab[ind], color='yellow', label='org')
            plt.plot(aa, pred_cur_lab_modified[ind], color='cornflowerblue', label='modified, layer5+1')
            # plt.plot(aa, lab_base[ind], color='pink', label='model1_lab_base')
            plt.legend()
            plt.savefig(os.path.join(save_dir, '{}.png'.format(oven)))
            # plt.show()
            plt.close()


def search_thickness(test_data, save_dir):
    # test_data = np.load('./test_x_seed{}_pca{}.npy'.format(seed, pca), allow_pickle=True)
    # test_x: [lab + sensor_feature, slim_function_lab, thick, oven]
    cur_real_lab = [x[0][:81] for x in test_data]
    slim_cur_real_labs = [[a[ind] for ind in important_inds] for a in cur_real_lab]
    aa = [i for i in range(14)]
    ovens = [x[-1] for x in test_data]
    slim_thick1 = [x[1] for x in test_data]
    thick1s = [x[2] for x in test_data]
    ll = len(important_inds)
    model1s, model2s = load_model1_model2(ll, seed, pca)
    lab_base = pre_lab_base(model1s, test_data)
    thick12_mean_std = normalize_thick12([], flag=True)
    l_ = len(thick12_mean_std[0])

    def weighted_metric(model_pred_m):
        ll = len(slim_best_cx_curve)
        # diff = [np.power((slim_best_cx_curve[i] - model_pred_m[i]), 2) for i in range(ll)]
        diff = [abs(slim_best_cx_curve[i] - model_pred_m[i]) for i in range(ll)]
        weights = [0.1] * ll
        weights[2], weights[9], weights[4], weights[12], weights[13] = 2, 2, 2, 2, 2
        diff = [diff[i] * weights[i] for i in range(ll)]

        if model_pred_m[4] > 2:
            weights[4] *= 2
        if model_pred_m[12] > 3:
            weights[12] *= 2
        if model_pred_m[2] > 0.26:
            weights[12] *= 2
        diff = [diff[i] * weights[i] for i in range(ll)]
        mean_diff = np.mean(diff)
        tmp_weights = [1] * ll
        tmp_weights[2], tmp_weights[4], tmp_weights[9], tmp_weights[12], tmp_weights[13] = 0, 0, 0, 0, 0
        tmp_diff = [diff[i] * tmp_weights[i] for i in range(ll)]
        a = np.mean(tmp_diff)
        b = mean_diff - a
        print("*** 总mse: {}, 重要频段mse: {}, 除重要频段mse: {} ***".format(mean_diff, b, a))

        return mean_diff

    for ind, oven in enumerate(ovens):
        pre_thickness = thick1s[ind]
        slim_function_thick1 = slim_thick1[ind]
        lab_base1 = lab_base[ind]
        slim_cur_real_lab = slim_cur_real_labs[ind]
        finetune_thre1 = [0.01, 0.01, 0.4, 0.6, 1]
        return_scale = [1] * 7
        # return_scale[4] = 0.8
        def get_forward_pred_lab(searching_thick2):
            recommend_thickness = [searching_thick2[0]] + [4] + searching_thick2[1:] + [35]
            thick12 = pre_thickness + recommend_thickness
            mean_, std_ = thick12_mean_std[0], thick12_mean_std[1]
            thick12_nor = [(thick12[i] - mean_[i]) / std_[i] for i in range(l_)]
            y2s = []
            for ind, model2 in enumerate(model2s):
                y2 = model2.predict(np.array([thick12_nor])).tolist()[0]
                y2s.append(y2)
            # print("model2 output: {}".format(y2s))
            function_thick2 = function_thick2lab(recommend_thickness)
            slim_function_thick2 = [function_thick2[ind] for ind in important_inds]
            deta_slim_function_thick12 = [slim_function_thick2[i] - slim_function_thick1[i] for i in range(ll)]
            recommend_next_lab = [lab_base1[i] + deta_slim_function_thick12[i] - y2s[i] for i in range(ll)]

            return recommend_next_lab

        def evaluate_config(config):
            pred_lab = get_forward_pred_lab([config['x1'], config['x2'], config['x3'], config['x4'], config['x5']])
            # plt.plot([i for i in range(14)], pred_lab, label='searching_thick2_pred_lab{}'.format(ind))
            # plt.legend()
            # plt.show()
            metric = weighted_metric(pred_lab)
            tune.report(metric=metric)
            # tmp save ~
            np.save('./pred_next_lab.npy', np.array(pred_lab))

        analysis = tune.run(
            evaluate_config,
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
            },
            low_cost_partial_config={'x1': pre_thickness[0],
                                     'x2': pre_thickness[2],
                                     'x3': pre_thickness[3],
                                     'x4': pre_thickness[4],
                                     'x5': pre_thickness[5]},  # a initial (partial) config with low cost
            metric='metric',  # the name of the metric used for optimization
            mode='min',  # the optimization mode, 'min' or 'max'
            num_samples=-1,  # the maximal number of configs to try, -1 means infinite
            time_budget_s=15,  # the time budget in seconds
            local_dir='logs/',  # the local directory to store logs
        )

        search_res = analysis.best_trial.last_result
        thickness_res = [np.round(search_res['config']['x1'], 2), 4, np.round(search_res['config']['x2'], 2),
                         np.round(search_res['config']['x3'], 2), np.round(search_res['config']['x4'], 2),
                         np.round(search_res['config']['x5'], 2), 35]

        thickness_diff = [thickness_res[i] - pre_thickness[i] for i in range(7)]
        thickness_diff[1], thickness_diff[-1] = 0, 0
        thickness_diff = [thickness_diff[i] * return_scale[i] for i in range(7)]
        pred_lab = np.load('./pred_next_lab.npy')

        plt.plot(aa, slim_cur_real_lab, color='pink', label='last_machine_curve: {}'.format(''.join(str(np.round(a, 3)) + ', ' for a in pre_thickness)))
        plt.plot(aa, pred_lab, color='dimgray', linewidth=1.0, label='recommend_next_lab, thickness diff: {}'.format(
                     ''.join(str(np.round(a, 2)) + ', ' for a in thickness_diff)))
        plt.plot(aa, slim_best_cx_curve, color='cornflowerblue', label='best_curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir, '{}.png'.format(oven)))
        # plt.show()
        plt.close()


def run():
    if len(sys.argv) != 2:
        sys.exit("python zeiss_mian.py flag")

    global flag, important_inds, seed, best_cx_curve, slim_best_cx_curve, pca, bad_ovens  #train_sensor_feature, test_sensor_feature
    flag = sys.argv[1:][0]
    bad_ovens = []

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
    seeds = [6]
    pcas = [20]

    all_train_data, online_test_data = pd.read_csv('./till_0721_train.csv', encoding='utf-8'), pd.read_csv('./till_0721_test.csv', encoding='utf-8')
    tsf_train_feature, tsf_test_feature = './tsf_train_feature.json', './tsf_test_feature.json'

    train_mse_txt = r'./train.txt'
    test_lab_mse = r'./test_lab_mse_test.txt'
    ff_train = open(train_mse_txt, 'w')
    test_lab_mse_txt = open(test_lab_mse, 'w')

    for seed in seeds:
        for pca in pcas:
            if flag == '0':
                # tsf提取sensor.csv特征
                generate_tsf_feature(all_train_data, online_test_data, tsf_train_feature, tsf_test_feature)

            if flag == '1':
                print("split model1 model2 data, generate online_test data ~")
                train_sensor_feature = json2npy_pca(json.load(open(tsf_train_feature, 'r')), pca_n=pca)
                test_sensor_feature = json2npy_pca(json.load(open(tsf_test_feature, 'r')), pca_n=pca)
                model1_x, model1_y, model2_x, model2_y = generate_data(all_train_data, train_sensor_feature)

            if flag == '2':
                print("train model1 ~")
                try:
                    model1_x, model1_y = np.load('./model1_x_seed{}_pca{}.npy'.format(seed, pca)), np.load('./model1_y_seed{}_pca{}.npy'.format(seed, pca))
                except:
                    print("miss model1 data")
                    continue
                model1s = train_model1(model1_x, model1_y, ff_train)
                print("train model1 done ~")
                try:
                    model2_x, model2_y = np.load('./model2_x_seed{}_pca{}.npy'.format(seed, pca), allow_pickle=True), np.load('./model2_y_seed{}_pca{}.npy'.format(seed, pca), allow_pickle=True)
                except:
                    print("miss model2 data")
                    continue
                model2s = train_model2(model1s, model2_x, model2_y, ff_train)
                print("train model2 done ~")

            if flag == "3":
                save_dir = r'./test_forward'
                makedir(save_dir)
                train_sensor_feature = json2npy_pca(json.load(open(tsf_train_feature, 'r')), pca_n=pca)
                test_sensor_feature = json2npy_pca(json.load(open(tsf_test_feature, 'r')), pca_n=pca)
                # 没见过的test数据上做模型前向拟合测试
                model1_x, model1_y, model2_x, model2_y = generate_forward_check_data([online_test_data], test_sensor_feature)
                forward_model_check(save_dir, test_lab_mse_txt, model1_x, model1_y, model2_x, model2_y, data_flag='test')

                # train data 上抽一些数据做曲线拟合测试, 模型应该是见过ta们的, 效果不应该太差..
                # save_dir = r'./train_forward'
                # makedir(save_dir)
                # train_lab_mse_txt = open(r'./train_lab_mse_test.txt', 'w')
                # model1_x, model1_y, model2_x, model2_y = generate_forward_check_data([all_train_data], train_sensor_feature)
                # forward_model_check(save_dir, train_lab_mse_txt, model1_x, model1_y, model2_x, model2_y, data_flag='train')

            if flag == "4":
                # 只修改model2的thick2, model1直接调用
                model2_x_data = r'./train_model2_x{}{}.npy'.format(seed, pca)
                model2_y_data = r'./train_model2_y{}{}.npy'.format(seed, pca)
                save_dir = r'./trend'
                makedir(save_dir) 
                model_trend_check(save_dir, model2_x_data, model2_y_data)

            if flag == '5':
                save_dir = r'./online_test'
                makedir(save_dir)
                train_sensor_feature = json2npy_pca(json.load(open(tsf_train_feature, 'r')), pca_n=pca)
                test_sensor_feature = json2npy_pca(json.load(open(tsf_test_feature, 'r')), pca_n=pca)
                test_data = generate_online_test_data(online_test_data, test_sensor_feature)
                search_thickness(test_data, save_dir)


if __name__ == '__main__':

    run()
    # flag: 0, generate tsf feature
    # flag: 1, generate data
    # flag: 2, train model1, model2
    # flag: 3, model forward test
    # flag: 4, model_trend test
    # flag: 5, online search thickness
