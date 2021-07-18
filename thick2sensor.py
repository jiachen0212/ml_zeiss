# coding=utf-8
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optimizers
from sklearn.model_selection import train_test_split
from data_load import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.my_mse_loss import my_mse_loss1
from utils.my_mse_loss import my_mse_loss2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.a2 = nn.ReLU()
        # self.l3 = nn.Linear(hidden_dim, output_dim)
        # self.a3 = nn.ReLU()
        # self.l4 = nn.Linear(hidden_dim, hidden_dim)
        # self.a4 = nn.ReLU()
        # self.l5 = nn.Linear(hidden_dim, output_dim)
        # self.a5 = nn.ReLU()

        # self.layers = [self.l1, self.a1, self.l2, self.a2, self.l3, self.a3, self.l4, self.a4, self.l5, self.a5]
        self.layers = [self.l1, self.a1, self.l2, self.a2]


    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
        return x

def compute_loss(t, y, n):
    return my_mse_loss2()(y, t, n)
    # return nn.MSELoss()(y, t)


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('thickhc17-sensor-thickness loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()



def show_y_pred(inds, y, gt_y=None, epo=None, flag='train'):
    n, dims = y.shape
    plt.title('{} epoch {}'.format(flag, epo + 1))
    plt.xlabel("iter")
    plt.ylabel("sensor value")
    x = [i for i in range(dims)]
    for i in inds:
    # for i in range(n):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        if i == 0:
            plt.plot(x, single_gt_y, color='pink', label='gt')
            plt.plot(x, single_y, color='cornflowerblue', label='thickhc2sensor')
        else:
            plt.plot(x, single_gt_y, color='pink')
            plt.plot(x, single_y, color='cornflowerblue')
    plt.legend()
    plt.show()


def show_all_data(X, data):
    '''
    std比较大的两个index: 0 8
    通过与mean值的大小比较划分开数据,得到不那么分散的几个数据群..
    数据不那么分散不代表机器的隐含状态接近,属于统一part...
    '''
    part08_1 = []
    part08_2 = []
    mean_data = np.mean(data, axis=0)
    # diff_data = data - mean_data
    # diff_data = np.array([[abs(i) for i in a] for a in diff_data])
    # diff_mean = np.mean(diff_data, axis=0)
    x = [i for i in range(data.shape[1])]
    for i in range(data.shape[0]):
        if data[i][8] < mean_data[8]:
            if data[i][0] < mean_data[0]/2:
                # plt.plot(x, data[i], color='pink')
                part08_1.append([X[i], data[i]])
            elif mean_data[0]/2 < data[i][0] < mean_data[0]:
                plt.plot(x, data[i], color='cornflowerblue')
                part08_2.append([X[i], data[i]])
        # else:
        #     plt.plot(x, data[i], color='black')
    plt.legend()
    plt.show()




# flag = 1 拟合model1
# flag = 0 微调阶段
# flag = 2 拆分洗数据
flag = 2

if flag == 1:
    data = json.load(open(r'./thick10_sensor16.json', 'r'))
    X = []
    Y = []
    for k, v in data.items():
        tmp = [v[1], v[6], v[9], v[14]]
        Y.append(tmp)
        X.append([float(i) for i in k.split(',')[:-1]])
        # if len(Y) == 100:
        #     break
        assert len(k.split(',')[:-1]) == 10
        assert len(v) == 15

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)

    # show_all_data(X, Y)
    scale = StandardScaler(with_mean=True, with_std=True)
    X = scale.fit_transform(X)  # X = scale.inverse_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.28, random_state=4)
    print("train size: {}".format(train_x.shape[0]))
    print("validation size: {}".format(test_x.shape[0]))

    batch_size = X.shape[0]
    input_dim = X.shape[1]
    hiden_dim = 20
    output_dim = Y.shape[1]
    epochs = 1000
    step2_epochs = 400
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    validation_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
    all_dataloader = DataLoader((X, Y), batch_size=batch_size, batch_first=False, device=device)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-4)

    loss_list = []
    n = 150
    for epoch in range(epochs):
        train_loss = 0
        # 这里比较特殊,因为是洗数据, 所以用all data??
        for ii, (data, label) in enumerate(all_dataloader):
            input = Variable(data, requires_grad=False)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss, minloss_index = compute_loss(score, target, n)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(all_dataloader)
        # print("epoch{}, loss: {}".format(epoch, train_loss))
        loss_list.append(train_loss)
        if (epoch + 1) % 500 == 0:
            for (data, label) in validation_dataloader:
                model.eval()
                pred = model(data)
                loss, minloss_index = compute_loss(pred, label, n)
                score = pred.detach().numpy()
                print("validation loss: {}".format(loss))
                show_y_pred(minloss_index, score, gt_y=label, epo=epoch, flag='validation')
        if epoch == epochs - 1:
            for (data, label) in all_dataloader:
                model.eval()
                pred = model(data)
                loss, minloss_index = compute_loss(pred, label, n)
                score = pred.detach().numpy()
                show_y_pred(minloss_index, score, gt_y=label, epo=epoch, flag='train')
                # 落盘这洗干净的100个样本
                f = open(r'./part1.txt', 'w')
                for ind in minloss_index:
                    f.write(str(ind)+',')

                label_np = label.detach().numpy()
                data_np = data.detach().numpy()
                minloss_index.sort()
                count = 0
                for ind in minloss_index:
                    label_np = np.delete(label_np, ind-count, axis=0)
                    data_np = np.delete(data_np, ind-count, axis=0)
                    count += 1
                np.save(r'./data.npy', data_np)
                np.save(r'./lab.npy', label_np)

    torch.save(model.state_dict(), "./thickhc2sensor1.pth")
    plot_loss(loss_list)


elif flag == 2:
    ns = [0, 0, 120, 80, 60, 50, 50, 50, 60]
    for i in range(2, 9):
        X = np.load(r'./data.npy')
        Y = np.load(r'./lab.npy')
        assert X.shape[0] == Y.shape[0]
        print("all data size: {}".format(X.shape[0]))

        scale = StandardScaler(with_mean=True, with_std=True)
        X = scale.fit_transform(X)
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.28, random_state=4)
        print("train size: {}".format(train_x.shape[0]))
        print("validation size: {}".format(test_x.shape[0]))

        batch_size = X.shape[0]
        input_dim = X.shape[1]
        hiden_dim = 40     # 20
        output_dim = Y.shape[1]
        epochs = 1000
        step2_epochs = 400
        model = MLP(input_dim, hiden_dim, output_dim).to(device)
        print(model)
        train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
        validation_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
        all_dataloader = DataLoader((X, Y), batch_size=batch_size, batch_first=False, device=device)
        optimizer = optimizers.Adam(model.parameters(),
                                    lr=0.001,
                                    betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-4)

        loss_list = []
        for epoch in range(epochs):
            train_loss = 0
            for ii, (data, label) in enumerate(all_dataloader):
                input = Variable(data, requires_grad=False)
                target = Variable(label)
                optimizer.zero_grad()
                score = model(input)
                loss, minloss_index = compute_loss(score, target, ns[i])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(all_dataloader)
            loss_list.append(train_loss)
            if (epoch + 1) % 500 == 0:
                for (data, label) in validation_dataloader:
                    model.eval()
                    pred = model(data)
                    loss, minloss_index = compute_loss(pred, label, ns[i])
                    score = pred.detach().numpy()
                    print("validation loss: {}".format(loss))
                    show_y_pred(minloss_index, score, gt_y=label, epo=epoch, flag='validation')
            if epoch == epochs - 1:
                for (data, label) in all_dataloader:
                    model.eval()
                    pred = model(data)
                    loss, minloss_index = compute_loss(pred, label, ns[i])
                    score = pred.detach().numpy()
                    show_y_pred(minloss_index, score, gt_y=label, epo=epoch, flag='train')
                    # 落盘这洗干净的100个样本
                    f = open(r'./part{}.txt'.format(i), 'w')
                    for ind in minloss_index:
                        f.write(str(ind) + ',')

                    label_np = label.detach().numpy()
                    data_np = data.detach().numpy()
                    minloss_index.sort()
                    count = 0
                    for ind in minloss_index:
                        label_np = np.delete(label_np, ind - count, axis=0)
                        data_np = np.delete(data_np, ind - count, axis=0)
                        count += 1
                    np.save(r'./data.npy', data_np)
                    np.save(r'./lab.npy', label_np)

        torch.save(model.state_dict(), "./thickhc2sensor{}.pth".format(i))
        plot_loss(loss_list)






























'''
膜厚微调阶段,可先不管

elif flag == 0:
    # finetune data
    # modify = [1,4,11,13]
    for y in Y:
        y[1] *= 1.3
        y[4] *= 1.015
        y[11] *= 1.015
        y[13] *= 1.05
    X = []
    Y = []
    sub_data = json.load(open('./f1716lab.json', 'r'))
    data_large = list(sub_data.keys())
    for data in data_large:
        data = data.split(',')[:-1]
        X.append([float(i) for i in data[:17]])
        Y.append([float(i) for i in data[17:]])
    X = np.array(X)
    Y = np.array(Y)
    scale = StandardScaler(with_mean=True, with_std=True)
    X = scale.fit_transform(X)
    all_data = DataLoader((X, Y), batch_size=batch_size, batch_first=False, device=device)
    model.load_state_dict(torch.load("./thickhc2sensor.pth"))
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False
    loss_list = []
    for epoch in range(step2_epochs):
        train_loss = 0
        for ii, (data, label) in enumerate(all_data):
            if epoch == 0:
                inverse_data = scale.inverse_transform(data.detach().numpy())
                np.save(r'./start_thickhc.npy', inverse_data)
                np.save(r'./start_y.npy', label.detach().numpy())
            data = Variable(data, requires_grad=True)
            optimizer = optimizers.Adam({data},
                                        lr=0.05,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            optimizer.zero_grad()
            score = model(data)
            loss, minloss_index = compute_loss(score, label)
            loss.backward()
            # print(data.grad[0])
            optimizer.step()
            train_loss += loss.item()
            if (epoch + 1) % 200 == 0:
                score = model(data).detach().numpy()
                show_y_pred(score, gt_y=label, epo=epoch, flag='train')
            if epoch == step2_epochs - 1:
                inverse_data = scale.inverse_transform(data.detach().numpy())
                print(inverse_data)
                np.save(r'./end_thickhc.npy', inverse_data)
                np.save(r'./end_y.npy', label.detach().numpy())
        train_loss /= len(train_dataloader)
        loss_list.append(train_loss)
        # print('-' * 10, 'loss: {}'.format(train_loss), '-' * 10)
    plot_loss(loss_list)

'''
