# coding=utf-8
import json
import numpy as np
import os


path = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'
files = os.listdir(path)
for f in files:
    if '.7z' in f or '.zip' in f:
        os.remove(os.path.join(path, f))


def load_data(data_js):
    data = json.load(open(data_js, 'r'))
    Y = []
    for f, lab in data.items():
        Y.append(lab)
    X = list(data.keys())
    # 剔除末尾的','
    X = [i[:-1] for i in X]
    X = [i.split(',') for i in X]
    # 剔除evt_name
    X = [i[:-1] for i in X]
    X = [[float(i) for i in a] for a in X]
    X = np.array(X)
    Y = [[float(i) for i in a] for a in Y]
    # Y = np.array(Y)
    X = np.array(X)
    return X, Y



if __name__ == "__main__":
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\第二批7.1\0701\all.json'
    # [380, 405, 440, 640, 750, 780]
