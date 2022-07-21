# coding=utf-8
import os
import matplotlib.pyplot as plt
import pandas as pd
from function_thick2lab import function_thick2lab

def strip_a(a):
    a = a.strip("['] ")
    return a


def get_function_machine_lab(cx_data):
    '''
    还是用dict吧, 方便关联正背的炉号

    '''

    lens = len(cx_data)
    OvenNo_m_f_labs = dict()
    for i in range(lens):
        OvenNo = cx_data.iloc[i]['OvenNo']
        if OvenNo not in OvenNo_m_f_labs:
            OvenNo_m_f_labs[OvenNo] = []
        # machine lab_curve
        lab1 = [float(strip_a(a)) for a in cx_data.iloc[i]['single_lab_curve'].split(',')]
        OvenNo_m_f_labs[OvenNo].append(lab1)

        # function lab_curve
        thicks = [float(strip_a(a)) for a in cx_data.iloc[i]['Thickness'].split(',')]
        lab2 = function_thick2lab(thicks)
        OvenNo_m_f_labs[OvenNo].append(lab2)

    return OvenNo_m_f_labs


def plt_cx_mf_cc_mf(cxfm, ccfm, aa, OvenNo):
    plt.title(OvenNo)
    # m
    plt.plot(aa, cxfm[0], color='pink', label='machine cx lab')
    plt.plot(aa, ccfm[0], color='cornflowerblue', label='machine cc lab')
    # f
    plt.plot(aa, cxfm[1], color='red', label='function cx lab')
    plt.plot(aa, ccfm[1], color='black', label='function cc lab')
    plt.legend()
    path_ = r'C:\Users\15974\Desktop\2'
    fig_path = os.path.join(path_, OvenNo+'.png')
    plt.savefig(fig_path)
    plt.close()
    # plt.show()



cx_file = r'./1008_cx_data.csv'
cc_file = r'./1008_cc_data.csv'
cx_data = pd.read_csv(cx_file)
cc_data = pd.read_csv(cc_file)

# 剔除cc中的异常样本
# print(len(cc_data))
# selected_ = cc_data.OvenNo.apply(lambda x: x not in [33321091708, 33321100106]).reset_index(drop=True)
# cc_data = cc_data[selected_].reset_index(drop=True)
# print(len(cc_data))
# cc_data.to_csv(r'./1008_cc_data.csv', index=False)


aa = [380+i*5 for i in range(81)]
OvenNo_m_f_labs_cx = get_function_machine_lab(cx_data)
OvenNo_m_f_labs_cc = get_function_machine_lab(cc_data)

# 对比machine 和 function 的正背趋势是否一致
for OvenNo, cx_m_f_lab in OvenNo_m_f_labs_cx.items():
    try:
        cc_m_f_lab = OvenNo_m_f_labs_cc[OvenNo]
        plt_cx_mf_cc_mf(cx_m_f_lab, cc_m_f_lab, aa, str(OvenNo))
    except:
        continue

# 把function的正背曲线都画出来; 把machine的正背曲线都画出来..
m_cx = []
m_cc = []
f_cx = []
f_cc = []
for OvenNo, cx_m_f_lab in OvenNo_m_f_labs_cx.items():
    try:
        cc_m_f_lab = OvenNo_m_f_labs_cc[OvenNo]
        m_cx.append(cx_m_f_lab[0])
        m_cc.append(cc_m_f_lab[0])
        f_cx.append(cx_m_f_lab[1])
        f_cc.append(cc_m_f_lab[1])
    except:
        continue

plt.title('machine cx and cc')
for i, m_cx_ in enumerate(m_cx):
    if i == 0:
        plt.plot(aa, m_cx_, color='pink', label='machine_cx')
    else:
        plt.plot(aa, m_cx_, color='pink')
for i, m_cc_ in enumerate(m_cc):
    if i == 0:
        plt.plot(aa, m_cc_, color='cornflowerblue', label='machine_cc')
    else:
        plt.plot(aa, m_cc_, color='cornflowerblue')
plt.legend()
plt.show()
plt.close()

plt.title('function cx and cc')
for i, f_cx_ in enumerate(f_cx):
    if i == 0:
        plt.plot(aa, f_cx_, color='red', label='machine_cx')
    else:
        plt.plot(aa, f_cx_, color='red')
for i, f_cc_ in enumerate(f_cc):
    if i == 0:
        plt.plot(aa, f_cc_, color='black', label='machine_cc')
    else:
        plt.plot(aa, f_cc_, color='black')
plt.legend()
plt.show()
plt.close()
