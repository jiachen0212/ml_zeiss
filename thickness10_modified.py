# coing=utf-8
import numpy as np
import json

lab_s = np.load(r'./start_lab.npy').tolist()
start = np.load(r'./start_x.npy')

lab_m = np.load(r'./modified_lab.npy').tolist()
modify = np.load(r'./modified_x.npy')

ind_ = []
for lab in lab_s:
    ind_.append(lab_m.index(lab))
print(ind_)

# 直接修改10层膜厚
# for j in range(len(ind_)):
#     s = start[j]
#     ll = len(s)
#     m = modify[ind_[j]]
#     diff = [m[i]-s[i] for i in range(ll)]
#     for i in range(1, 5):
#         info = "背面镀膜, 第{}层调整数值{}".format(i+2, diff[i])
#         print(info)
#     for i in range(5, 10):
#         if i == 5:
#             info = "正面镀膜, 第{}层调整数值{}".format(i-4, diff[i])
#         else:
#             info = "正面镀膜, 第{}层调整数值{}".format(i - 3, diff[i])
#         print(info)
#     print('\n')



# 修改 sensor std
for j in range(len(ind_)):
    s = start[j]
    ll = len(s)
    m = modify[ind_[j]]
    diff = [m[i]-s[i] for i in range(ll)]
    for i in range(8):
        info = "背面std, step{}调整数百分比{}".format(i+1, abs(100*diff[i]/s[i]))
        print(info)
    for i in range(8, 16):
        info = "正面std, step{}层调整数值{}".format(i-7, abs(100*diff[i]/s[i]))
        print(info)
    print('\n')