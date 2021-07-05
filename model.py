# coding=utf-8
import json
import os

import numpy as np
import torch
import torch.optim as optimizers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_load import DataLoader
from data_post_process import data_post_process
from mlp_torch import MLP
from utils.my_mse_loss import my_mse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from util import cnames
from util import calculate_Lab
from util import Select_feature
from torch.autograd import Variable

colors = list(cnames.keys())
from sklearn.neural_network import MLPRegressor


def get_important_x():
    weights = [1 for i in range(81)]
    nms = [380 + i * 5 for i in range(81)]
    a = [380, 400, 405, 410, 435, 440, 445, 635, 640, 645, 780]
    for n in a:
        weights[nms.index(n)] = 2
    # print(weights)
    return weights


def mlp_fun(test_x, test_y, train_x, train_y):
    # mlp regression
    mlpr = MLPRegressor(hidden_layer_sizes=(1,),  # 4层隐含层
                        activation='tanh',
                        solver='adam',
                        alpha=0.0001,  # L2惩罚参数
                        max_iter=1,
                        random_state=123, )
    mlpr.fit(train_x, train_y)
    pre_y = mlpr.predict(test_x)
    print("mean absolute error:", metrics.mean_absolute_error(test_y, pre_y))


def compute_loss(t, y):
    # return nn.MSELoss()(y, t)
    return my_mse_loss()(y, t)


def show_y_pred(y, gt_y=None, epo=None, best=None, flag='eval'):
    sample_num, dims = y.shape
    plt.title('{} epoch {} lab_curve'.format(flag, epo + 1))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        if single_gt_y[24] > 2:
            if i == 0:
                plt.plot(x, single_gt_y, color='pink', label='gt_bad')
                plt.plot(x, single_y, color='black', label='mlp_bad')
            else:
                plt.plot(x, single_gt_y, color='pink')
                plt.plot(x, single_y, color='black')
            # plt.legend()
        else:
            if i == 0:
                plt.plot(x, single_gt_y, color='cornflowerblue', label='origin')
                plt.plot(x, single_y, color='moccasin', label='mlp regression')
            else:
                plt.plot(x, single_gt_y, color='cornflowerblue')
                plt.plot(x, single_y, color='moccasin')
        # if i == 0:
        #     plt.plot(x, single_gt_y, color='cornflowerblue', label='origin')
        #     plt.plot(x, single_y, color='moccasin', label='mlp regression')
        # else:
        #     plt.plot(x, single_gt_y, color='cornflowerblue')
        #     plt.plot(x, single_y, color='moccasin')
    if best:
        plt.plot(x, best, color='red', label='target')
    plt.legend()
    plt.savefig("lab_curve.png")
    plt.show()


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('modify-thickness loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.savefig("pretrain_loss.png")
    plt.show()


def generate_data(data_part1, file1, file2, evt_cc_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                  evt_33number, base_data_dir, CC_dir, CX_dir, bad_js, num33_hc_js, number33_thick_js, data_json,
                  thick14_hc3_sensor16_lab_js, thick14_hc3_sensor64_lab_js, feature135_lab_js, full_135feature_js,
                  flag=0):  # flag=0,默认选最新最多的特征

    # concate_data(data_part1, feature135_lab_js, full_135feature_js)
    # 可备选的,使用的json数据,分别有: 135, 33, 97 dims-feature
    X_list = [full_135feature_js, feature135_lab_js, thick14_hc3_sensor16_lab_js, thick14_hc3_sensor64_lab_js]
    tmp = X_list[flag]
    if not os.path.exists(tmp):
        data_post_process(file1, file2, evt_cc_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                          evt_33number, base_data_dir, CC_dir, CX_dir, bad_js, num33_hc_js, number33_thick_js,
                          data_json,
                          thick14_hc3_sensor16_lab_js, thick14_hc3_sensor64_lab_js, feature135_lab_js).run()
        print("data process done!")
    else:
        print("data has already processed! start mlp！！!")

    with open(tmp, encoding="utf-8") as reader:
        thicknesshc_curve = json.load(reader)

    Y = []
    for thicknesshc, lab_curve in thicknesshc_curve.items():
        Y.append(lab_curve)
    Y = [[float(i) for i in a] for a in Y]
    X = list(thicknesshc_curve.keys())
    # X = [i[:-1] for i in X]
    X = [i.split(',') for i in X]
    print("all dara_lens: {}".format(len(X)))
    # check 750 反射率
    f = open(r'./bad_750.txt', 'w')
    count = 0
    bad_evt_name = []
    for i in range(len(Y)):
        if float(Y[i][import_index]) > 4.5:
            bad_evt_name.append(X[i][-1])
            f.write(X[i][-1] + ',')
            count += 1
    print('bad 750 rate_value: {}'.format(len(bad_evt_name)))
    X = [i[:-1] for i in X]  # 把evt_name从x的最后一位剔除
    X = [[float(i) for i in a] for a in X]
    # 手动调整某基层膜厚的值,看看曲线在哪些频段会产生很大变化否?
    # X = [[i[0],i[1],i[2]*1.5,i[3]*1.5,i[4]*2,i[5]*2,i[6]] for i in X]
    # added 0703
    # 对除14层膜厚之外的121维特征进行筛选
    X = Select_feature(X, Y)
    Y = np.array(Y)
    # X = np.array(X)
    # print(X.shape, Y.shape)
    return X, Y


def compare_res(best):
    best_ = np.array(best)
    y1 = np.load(r'./train.npy')
    y2 = np.load(r'./fine_tune.npy')
    mse1 = []
    mse2 = []
    sample_num, dims = y1.shape
    plt.title('compare lab_curve')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        a = y1[i, :]
        b = y2[i, :]
        mse1.append(metrics.mean_squared_error(a, best_))
        mse2.append(metrics.mean_squared_error(b, best_))
        plt.plot(x, a, color='cornflowerblue')
        plt.plot(x, b, color='hotpink')
        if i == 0:
            plt.plot(x, a, color='cornflowerblue', label='mlp')
            plt.plot(x, b, color='lightpink', label='modified-thickness')
        else:
            plt.plot(x, a, color='cornflowerblue')
            plt.plot(x, b, color='lightpink')
    plt.plot(x, best, color='red', label='target')
    plt.legend()
    plt.savefig("compare_lab_curve.png")
    plt.show()
    print("base mse: {}, fine_tune mse: {}".format(np.mean(mse1), np.mean(mse2)))
    print(np.mean(mse1) > np.mean(mse2))


def run(DataLoader, scale, train_x, train_y, model, train_dataloader, val_dataloader, epochs, best, is_train=True,
        optimizer=None):
    if is_train:
        loss_list = []
        for epoch in range(epochs):
            train_loss = 0
            # print('-' * 10, 'epoch: {}'.format(epoch + 1), '-' * 10)
            for ii, (data, label) in enumerate(train_dataloader):
                # print(data[0])
                # print(label[0])
                # print(data.shape, 'train')
                input = Variable(data, requires_grad=False)
                target = Variable(label)
                optimizer.zero_grad()
                score = model(input)
                loss = compute_loss(score, target)
                # print(metrics.mean_squared_error(score.detach().numpy(), target.detach().numpy()))
                # print('-' * 10, 'epoch {} loss: {}'.format(epoch, loss), '-' * 10)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            loss_list.append(train_loss)
            if (epoch + 1) % 1000 == 0:
                model.eval()
                for ii, (input, org) in enumerate(val_dataloader):
                    # print(input.shape, 'val')
                    model.eval()
                    pred = model(input)
                    y = pred.detach().numpy()
                    show_y_pred(y, org, epo=epoch, flag='validation')
            if epoch == epochs - 1:
                model.eval()
                for ii, (input, org) in enumerate(train_dataloader):
                    model.eval()
                    pred = model(input)
                    y = pred.detach().numpy()
                    show_y_pred(y, org, epo=epoch, flag='train')
        plot_loss(loss_list)
        torch.save(model.state_dict(), "./mlp.pth")
        all_data = DataLoader((train_x, train_y), batch_size=train_x.shape[0], batch_first=False, device=device)
        for (data, label) in all_data:
            model.eval()
            preds = model(data)
            y_pred = preds.detach().numpy()
            np.save(r'./train.npy', y_pred)
            show_y_pred(y_pred, gt_y=label, epo=epochs - 1, best=best, flag='train')
    else:
        model.load_state_dict(torch.load("./mlp.pth"))
        # params = list(model.named_parameters())
        # print(params[0])   # bn2 + fc2 * 3 = 8
        # 冻结mlp内部参数
        for index, p in enumerate(model.parameters()):
            p.requires_grad = False
        loss_list = []
        for epoch in range(epochs):
            train_loss = 0
            for ii, (data, label) in enumerate(train_dataloader):
                # 用标准曲线作为target,逼近膜厚去拟合最佳曲线
                target = best * data.shape[0]
                target = np.array(target)
                target = np.reshape(target, (data.shape[0], -1))
                target = Variable(torch.from_numpy(target).float())
                # target = label
                data = Variable(data, requires_grad=True)
                # optimizer = torch.optim.SGD({data},
                #                             weight_decay=5e-3, lr=1e-7, momentum=0.5)
                optimizer = optimizers.Adam(model.parameters(),
                                            lr=1e-8,
                                            betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
                optimizer.zero_grad()
                score = model(data)
                loss = compute_loss(score, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if epoch == epochs - 1:
                    model.eval()
                    preds = model(data)
                    y_pred = preds.detach().numpy()
                    np.save(r'./fine_tune.npy', y_pred)
                    X = scale.inverse_transform(data.detach().numpy())
                    # print(X)
                    np.save(r'./modified_x.npy', X)
            train_loss /= len(train_dataloader)
            loss_list.append(train_loss)
            print('-' * 10, 'loss: {}'.format(train_loss), '-' * 10)
        plot_loss(loss_list)
        print(loss_list.index(min(loss_list)))  # 返回fine-tune阶段min_loss出现的epoch
        print(max(loss_list), min(loss_list))


def data_info(X, Y):
    '''

    :param X:
    :param Y:
    :return: 检查一个膜厚设置输入, 是否对应多条lab曲线.
    '''
    print("X shape:", X.shape)
    print("Y shape", Y.shape)
    data_length = X.shape[0]
    all_same = [True for i in X[0]]
    print(all_same)

    for i in range(1, data_length):
        for index, val in enumerate(X[0]):
            if X[i][index] != val:
                all_same[index] = False
    print(all_same)

    for i in range(data_length):
        for j in range(i):
            Xi = X[i]
            Xj = X[j]
            identical = (Xi == Xj).all()
            if identical:
                print("For input X {} and {} are identical:{}".format(i, j, Xi))
                Yi = Y[i]
                Yj = Y[j]
                y_identical = (Yi == Yj).all()
                if y_identical:
                    print("\tY {} and {} are the same.".format(i, j))
                else:
                    y_delta = Yi - Yj
                    print("\tY {} and {} not the same.".format(i, j))
                    # print("\tY[{}]:{}".format(i,Yi))
                    # print("\tY[{}]:{}".format(j,Yj))
                    print("\tDelta: {}".format(sum(abs(y_delta))))


def concate_data(a, b, c):
    js1 = json.load(open(a, 'r'))
    js2 = json.load(open(b, 'r'))
    print("pre_data_size: {}, cur_data_size: {}".format(len(js1), len(js2)))
    all_js = dict()
    for k, v in js1.items():
        all_js[k] = v
    for k, v in js2.items():
        all_js[k] = v
    data = json.dumps(all_js)
    with open(c, 'w') as js_file:
        js_file.write(data)


if __name__ == "__main__":

    x = [380 + i * 5 for i in range(81)]
    import_index = x.index(750)

    # 1train or 0modified_thickness
    flag = 1
    # get_important_x()

    # 标准lab曲线
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    _ = calculate_Lab(best)
    part_root_dir1 = r'D:\work\project\卡尔蔡司AR镀膜\第二批7.1\0701'
    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批'
    base_data_dir = os.path.join(root_dir, r'33机台文件')
    sub_dir = r'0705'
    js_save_path = os.path.join(root_dir, sub_dir)
    if not os.path.exists(js_save_path):
        os.mkdir(js_save_path)
    # evt_cc_dir = os.path.join(root_dir, r'机台文件\1.6&1.67_DVS_CC')
    evt_cc_dir = os.path.join(root_dir, r'33机台文件')
    CC_dir = os.path.join(root_dir, r'33机台文件')
    CX_dir = os.path.join(root_dir, r'33机台文件')
    file1 = os.path.join(root_dir, r'匹配关系2021.1~2021.6.xlsx')
    file2 = os.path.join(root_dir, r'33# DVS双面膜色2021.1~2021.6.xlsx')
    # 此文档用于关联周期信息,筛选相同膜厚设置值所对应的lab曲线
    process_data = os.path.join(root_dir, r'镀膜炉工艺记录2021.1~2021.6.xlsx')
    # .json都是数据处理中需要落盘的信息
    data_js = os.path.join(root_dir, sub_dir, 'thickness_lab_curve.json')
    refine_data_json = os.path.join(root_dir, sub_dir, 'refine_thickness_lab_curve.json')
    bad_js = os.path.join(root_dir, sub_dir, 'bad_thick_lab.json')
    oneone_evt_thickness = os.path.join(root_dir, sub_dir, 'oneone_evt_thickness.json')
    evt_33number = os.path.join(root_dir, sub_dir, 'evt_33number.json')
    # 加入3维耗材信息
    num33_hc_js = os.path.join(root_dir, sub_dir, '33_hc.json')
    number33_thick_js = os.path.join(root_dir, sub_dir, '33number_thickness.json')
    thick_hc_lab_js = os.path.join(root_dir, sub_dir, r'thick_hc_lab.json')
    # 加入16维sensor列时序特征
    thick14_hc3_sensor16_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor16_lab.json')
    # 加入64维 8step sensor时序特征
    csv_dict_js = os.path.join(root_dir, sub_dir, 'evtname_sensor_name_value.json')
    thick14_hc3_sensor64_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor64_lab.json')
    # 再加入19列有意义数据的38维特征
    feature135_lab_js = os.path.join(root_dir, sub_dir, 'feature135_lab.json')

    # merge two_part_data_json
    data_part1 = os.path.join(part_root_dir1, 'all.json')
    full_135feature_js = os.path.join(root_dir, sub_dir, 'all.json')

    if flag == 3:
        data_class = data_post_process(file1, file2, evt_cc_dir, data_js, process_data, refine_data_json,
                                       oneone_evt_thickness,
                                       evt_33number, base_data_dir, CC_dir, CX_dir, num33_hc_js, bad_js,
                                       number33_thick_js, thick_hc_lab_js,
                                       thick14_hc3_sensor16_lab_js, thick14_hc3_sensor64_lab_js, feature135_lab_js)
        # data_class.clean_data_machineid()
        # data_class.clean_data_nthickness()

    X, Y = generate_data(data_part1, file1, file2, evt_cc_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                         evt_33number, base_data_dir, CC_dir, CX_dir, bad_js, num33_hc_js, number33_thick_js,
                         thick_hc_lab_js, thick14_hc3_sensor16_lab_js, thick14_hc3_sensor64_lab_js, feature135_lab_js,
                         full_135feature_js)

    # X[np.isnan(X)] = 0.0
    batch_size = X.shape[0]
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]
    hiden_dim = 50
    epochs_train = 3000
    # 调整膜厚值
    epochs_finetune = 1000
    # 数据规整化
    scale = StandardScaler(with_mean=True, with_std=True)
    # 注意后面观察膜厚的变化,需要用到它的逆操作: X = scale.inverse_transform(X)
    X_ = scale.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X_, Y, test_size=0.25, random_state=3)
    print("train size: {}".format(train_x.shape[0]))
    print("validation size: {}".format(test_x.shape[0]))
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    val_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    optimizer_train = optimizers.Adam(model.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)  # L2正则
    if flag == 1:
        run(DataLoader, scale, train_x, train_y, model, train_dataloader, val_dataloader, epochs_train, best,
            optimizer=optimizer_train)
    elif flag == 0:
        run(DataLoader, scale, train_x, train_y, model, train_dataloader, val_dataloader, epochs_finetune, best,
            is_train=False)
        compare_res(best)
        # 怎么剔除异常点? 怎么使得每一个样本都刚好的逼近标准曲线？[膜厚设置值-实测>2*rate,考虑剔除]
        # data_info(X, Y)

'''
lab曲线相识度度量:
a = [11.27, 7.28, 5.08, 3.53, 2.08, 1.12, 0.77, 0.73, 0.64, 0.74, 1.18, 1.4, 1.6, 1.65, 1.73, 1.92, 2.02, 1.99, 1.84,
     1.68, 1.61, 1.64, 1.65, 1.54, 1.3, 1.09, 0.96, 0.97, 0.98, 0.94, 0.86, 0.72, 0.59, 0.57, 0.51, 0.49, 0.49, 0.54,
     0.51, 0.45, 0.37, 0.3, 0.27, 0.23, 0.2, 0.3, 0.26, 0.24, 0.22, 0.14, 0.17, 0.2, 0.23, 0.36, 0.31, 0.35, 0.39, 0.52,
     0.52, 0.52, 0.56, 0.67, 0.78, 0.87, 1.06, 1.25, 1.5, 1.65, 1.85, 2.1, 2.29, 2.49, 2.7, 2.92, 3.14, 3.33, 3.62,
     3.75, 4.1, 4.31, 4.67]

best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
        1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
        0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
        0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
        3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
import numpy as np
import matplotlib.pyplot as plt

a_ = np.array(a)
b_ = np.array(best)
print(np.corrcoef([a_, b_]))
#
weights = [1 for i in range(81)]
nms = [380 + i * 5 for i in range(81)]
t = [380, 400, 405, 410, 435, 440, 445, 635, 640, 645, 780]
for n in t:
    weights[nms.index(n)] = 2

res = [(a[i]-best[i])**2*weights[i] for i in range(81)]
print(np.mean(res))
plt.xlabel("Wave-length")
plt.ylabel("Reflectance")

plt.plot(nms, best, color='pink', label='best curve')
plt.plot(nms, a, color='black', label='sample_curve')
plt.legend()
plt.show()

'''
