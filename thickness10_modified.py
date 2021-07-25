# coing=utf-8
import numpy as np
import json

lab_s = np.load(r'./start_labnumber.npy').tolist()
start = np.load(r'./start_x.npy')

lab_m = np.load(r'./modified_labnumber.npy').tolist()
modify = np.load(r'./modified_x.npy')

print(len(lab_s))


ind_ = []
for lab in lab_s:
    ind_.append(lab_m.index(lab))
print(ind_)

flag = 2

if flag == 100:
    # 直接调膜厚值优化lab曲线
    for j in range(len(ind_)):
        s = start[j]
        ll = len(s)
        m = modify[ind_[j]]
        diff = [m[i]-s[i] for i in range(ll)]
        for i in range(1, 5):
            info = "背面镀膜, 第{}层调整数值{}".format(i+2, diff[i])
            print(info)
        for i in range(5, 10):
            if i == 5:
                info = "正面镀膜, 第{}层调整数值{}".format(i-4, diff[i])
            else:
                info = "正面镀膜, 第{}层调整数值{}".format(i - 3, diff[i])
            print(info)
        print('\n')


if flag == 1:
    # 统计 sensor std 的 index 和 修改值
    f = open('./std_value.txt', 'w')
    f_index = open('./std_index.txt', 'w')
    # 这里暂时实现的很粗糙, 所有炉号的std index一起做一个set.. 后续可优化
    modify_std_index = []
    for j in range(len(ind_)):
        s = start[j]
        ll = len(s)
        m = modify[ind_[j]]
        diff = [m[i]-s[i] for i in range(ll)]
        # 记录炉号信息
        f.write(str(lab_s[j][-1])+'\n')
        for i in range(7):
            info = "背面std, step{}调整数百分值{}".format(i+2, 100*diff[i]/s[i])
            if 0.09 <= abs(100*diff[i]/s[i]) < 1:
                modify_std_index.append(i+1)
            print(info)
            f.write(info + '\n')
        print('===')
        for i in range(7, 14):
            info = "正面std, step{}层调整数值{}".format(i-5, 100*diff[i]/s[i])
            if 0.09 <= abs(100*diff[i]/s[i]) < 1:
                modify_std_index.append(i+2)
            print(info)
            f.write(info + '\n')
        print('\n')
        f.write('\n')

    modify_std_index = list(set(modify_std_index))
    print(modify_std_index)
    for a in modify_std_index:
        f_index.write(str(a)+',')


if flag == 2:
    # 统计thick10因std值变化而改变的值
    f = open(r'./modify.txt', 'w')
    thickness = [25, 13, 37, 102, 91.5]
    for j in range(len(ind_)):
        s = start[j]
        ll = len(s)
        m = modify[ind_[j]]
        diff = [m[i]-s[i] for i in range(ll)]
        for i in range(5):
            if i == 0:
                info = "背面layer1, 调整数百分值: {}, 大约的修改值: {}".format(100 * diff[i] / s[i], diff[i]*thickness[i]/s[i])
                print(info)
            else:
                info = "背面layer{}, 调整数百分值: {}, 大约的修改值: {}".format(i+2, 100*diff[i]/s[i], diff[i]*thickness[i]/s[i])
                print(info)
            f.write(info+'\n')
        print('\n')
        for i in range(5, 10):
            if i == 5:
                info = "正面 layer1, 调整数百分值: {}, 大约的修改值: {}".format(100 * diff[i] / s[i], diff[i]*thickness[i-5]/s[i])
                print(info)
            else:
                info = "正面layer{}, 调整数百分值: {}, 大约的修改值: {}".format(i-3, 100*diff[i]/s[i], diff[i]*thickness[i-5]/s[i])
                print(info)
            f.write(info + '\n')
        print('\n')
        f.write('\n')