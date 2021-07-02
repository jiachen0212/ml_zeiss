# coding=utf-8
import json

import numpy as np


# path = r'D:\work\project\第二批7.1\机台文件'
# files = os.listdir(path)
# for f in files:
#     if '.7z' in f:
#         os.remove(os.path.join(path, f))


def load_data(data_js):
    data = json.load(open(data_js, 'r'))
    Y = []
    for f, lab in data.items():
        Y.append(lab)
    X = list(data.keys())
    X = [i[:-1] for i in X]
    X = [i.split(',') for i in X]
    X = [[float(i) for i in a] for a in X]
    X = np.array(X)
    Y = [[float(i) for i in a] for a in Y]
    Y = np.array(Y)
    print(X.shape, Y.shape)
    return X, Y


data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\org_refine_thickness_lab_curve.json'
load_data(data_js)
