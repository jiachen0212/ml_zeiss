import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_path):
    theory_para = pd.read_excel(file_path, sheet_name=[4], header=[1, 2])[4]
    theory_para_columns = ['波长', '基材', 'HC', '1st layer', '2nd layer', '3th layer', '4th layer', '5th layer',
                           '6th layer', '7th layer']
    theory_para = theory_para[theory_para_columns]

    thickness = pd.read_excel(file_path, sheet_name=[5], header=1)[5]
    thickness_columns = ['Layer', '标准膜厚/nm']
    thickness = thickness[thickness_columns]
    return theory_para, thickness


def cal_cos_theta(n, n0=1, sin_theta0=1):
    return np.sqrt(1 - np.square(n0 * sin_theta0 / n))


def cal_delte(cur_lambda, eta, d, cos_theta):
    return np.pi * 2 * eta * d * cos_theta / cur_lambda


def cal_Rs(theory_para, refer_thickness_new, layer_num, n0):
    Rs = []
    for index, row in theory_para.iterrows():
        cur_lambda = row.iloc[0]
        print('----------------------cur_lambda: %d----------------------' % cur_lambda)
        eta_s = complex(row['基材']['n'], row['基材']['k'])
        res_matrix = np.diag([1] * 2)
        for layer in range(layer_num):
            n, k = row.iloc[layer * 2 + 3], row.iloc[layer * 2 + 4]
            d = refer_thickness_new[layer]
            print('n: %.3f\tk:%.3f\t\td:%.1f' % (n, k, d))
            eta = complex(n, k)
            cos_theta = cal_cos_theta(eta)
            delte = cal_delte(cur_lambda, eta, d, cos_theta)
            cur_matrix = np.array([[np.cos(delte), complex(0, np.sin(delte)) / eta],
                                   [complex(0, np.sin(delte)) * eta, np.cos(delte)]])
            res_matrix = np.dot(cur_matrix, res_matrix)
        res_matrix = np.dot(res_matrix, np.array([1, eta_s]))
        res_matrix /= res_matrix[0]
        Y = res_matrix[-1]
        r = (n0 - Y) / (n0 + Y)
        R = (np.square(r.real) + np.square(r.imag)) * 100
        Rs.append(R)
    return Rs


def plot(Rs_all, result_path, pic_name):
    labels = ['original_thickness', 'HC +10%', 'HC -10%',
              'Layer1 +10%', 'Layer1 -10%', 'Layer2 +10%', 'Layer2 -10%', 'Layer3 +10%', 'Layer3 -10%',
              'Layer4 +10%', 'Layer4 -10%', 'Layer5 +10%', 'Layer5 -10%', 'Layer6 +10%', 'Layer6 -10%',
              'Layer7 +10%', 'Layer7 -10%']

    for index, data in enumerate(Rs_all):
        if index == 0:
            continue
        plt.figure(labels[index])
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title(pic_name + labels[index])
        ax = plt.subplot(111)
        ax.plot(range(360, len(Rs_all[0]) * 5 + 360, 5), Rs_all[0], linewidth=4, label=labels[0])
        ax.plot(range(360, len(data) * 5 + 360, 5), data, linewidth=2, label=labels[index])  # color='b'

        x_locator = plt.MultipleLocator(40)
        x_formatter = plt.FormatStrFormatter('%d')
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)

        y_locator = plt.MultipleLocator(2)
        y_formatter = plt.FormatStrFormatter('%.1f')
        ax.yaxis.set_major_locator(y_locator)
        ax.yaxis.set_major_formatter(y_formatter)

        plt.margins(x=0)
        ax.yaxis.grid(True, which='major')
        plt.legend(loc='best', frameon=False, prop={'size': 6})

        plt.savefig(os.path.join(result_path, 'pics', pic_name + labels[index] + '.png'))
    plt.show()


def res_to_excel(Rs_all, result_path):
    Rs_df = pd.DataFrame(np.array(Rs_all).T, index=[i for i in range(360, len(Rs_all[0]) * 5 + 360, 5)],
                         columns=['original_thickness', 'HC +10%', 'HC -10%',
                                  'Layer1 +10%', 'Layer1 -10%', 'Layer2 +10%', 'Layer2 -10%',
                                  'Layer3 +10%', 'Layer3 -10%', 'Layer4 +10%', 'Layer4 -10%',
                                  'Layer5 +10%', 'Layer5 -10%', 'Layer6 +10%', 'Layer6 -10%',
                                  'Layer7 +10%', 'Layer7 -10%'])
    Rs_df.to_excel(os.path.join(result_path, 'R_record.xlsx'))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    file_path = 'resources/data/Lab计算及膜厚范围.xlsx'
    result_path = 'results'
    layer_num, n0 = 8, 1

    theory_para, thickness = read_data(file_path)
    # print(theory_para.head(10))
    # print(theory_para.columns)
    # print(thickness.head(10))
    # print(thickness.columns)

    read_thickness = thickness['标准膜厚/nm'].tolist()
    refer_thickness = [2000]
    refer_thickness.extend(read_thickness)
    refer_thickness = np.array(refer_thickness)
    refer_thickness = np.array([0, 32.5, 4, 14.3, 48.1, 112.2, 118.95, 35])
    # refer_thickness = refer_thickness * 0.85

    Rs_all = []
    print('refer_thickness:', refer_thickness)
    Rs = cal_Rs(theory_para, refer_thickness, layer_num, n0)
    Rs_all.append(Rs)
    for i in range(len(refer_thickness) * 2):
        refer_thickness_new = refer_thickness.copy()
        if i % 2 == 0:
            refer_thickness_new[i // 2] += refer_thickness[i // 2] * 0.1
        else:
            refer_thickness_new[i // 2] -= refer_thickness[i // 2] * 0.1
        print('refer_thickness:', refer_thickness_new)
        Rs = cal_Rs(theory_para, refer_thickness_new, layer_num, n0)
        Rs_all.append(Rs)

    pic_name = '光谱曲线'
    plot(Rs_all, result_path, pic_name)
    res_to_excel(Rs_all, result_path)
