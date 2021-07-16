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
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.a3 = nn.ReLU()

        self.layers = [self.l1, self.a1, self.l2, self.a2]

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
        return x

def compute_loss(t, y):
    return my_mse_loss2()(y, t)


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('thickhc17-sensor-thickness loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()



def show_y_pred(y, gt_y=None, epo=None, flag='train'):
    sample_num, dims = y.shape
    plt.title('{} epoch {}'.format(flag, epo + 1))
    plt.xlabel("iter")
    plt.ylabel("sensor value")
    x = [i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        if i == 0:
            plt.plot(x, single_gt_y, color='pink', label='gt')
            plt.plot(x, single_y, color='black', label='thickhc2sensor')
        else:
            plt.plot(x, single_gt_y, color='pink')
            plt.plot(x, single_y, color='black')
    plt.legend()
    plt.show()


data = json.load(open(r'./thick10_sensor16.json', 'r'))
X = []
Y = []
thickhc_lab = dict()
for k, v in data.items():
    X.append([float(i) for i in k.split(',')[:-1]])
    Y.append(v)
    assert len(k.split(',')[:-1]) == 10
    assert len(v) == 16

X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

print("before scale: X: {}".format(X[0]))
scale = StandardScaler(with_mean=True, with_std=True)
X = scale.fit_transform(X)    # X = scale.inverse_transform(X)
print("after scale: X: {}".format(X[0]))

# print("before scale: Y: {}".format(Y[0]))
# scale1 = StandardScaler(with_mean=True, with_std=True)
# Y = scale1.fit_transform(Y)
# print("after scale: Y: {}".format(Y[0]))

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.28, random_state=4)
print("train size: {}".format(train_x.shape[0]))
print("validation size: {}".format(test_x.shape[0]))

batch_size = X.shape[0]
input_dim = X.shape[1]
hiden_dim = 100
output_dim = Y.shape[1]
epochs = 500
step2_epochs = 400
model = MLP(input_dim, hiden_dim, output_dim).to(device)
print(model)
train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
validation_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
optimizer = optimizers.Adam(model.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-4)
flag = 1

if flag == 1:
    loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data, requires_grad=False)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss = compute_loss(score, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        # print("epoch{}, loss: {}".format(epoch, train_loss))
        loss_list.append(train_loss)
        if (epoch + 1) % 100 == 0:
            for (data, label) in train_dataloader:
                score = model(data).detach().numpy()
                # print(label[0])
                # print(score[0])
                show_y_pred(score, gt_y=label, epo=epoch, flag='train')
            for (data, label) in validation_dataloader:
                model.eval()
                pred = model(data)
                loss = compute_loss(pred, label)
                score = pred.detach().numpy()
                # print("pred: {}".format(pred[0]))
                print("validation loss: {}".format(loss))
                show_y_pred(score, gt_y=label, epo=epoch, flag='validation')
    torch.save(model.state_dict(), "./thickhc2sensor.pth")
    plot_loss(loss_list)

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
            loss = compute_loss(score, label)
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


