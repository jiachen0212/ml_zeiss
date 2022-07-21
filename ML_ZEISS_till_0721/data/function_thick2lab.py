# coding=utf-8
import pandas as pd
import numpy as np
import warnings


# function for lab_curve
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
        # print('----------------------cur_lambda: %d----------------------' % cur_lambda)
        eta_s = complex(row['基材']['n'], row['基材']['k'])
        res_matrix = np.diag([1] * 2)
        for layer in range(layer_num):
            n, k = row.iloc[layer * 2 + 3], row.iloc[layer * 2 + 4]
            d = refer_thickness_new[layer]
            # print('n: %.3f\tk:%.3f\t\td:%.1f' % (n, k, d))
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




def function_thick2lab(thicks):

    refer_thickness_cx = np.array([2000, thicks[0]*1.3, 4, thicks[2]*1.1, thicks[3]*1.3, thicks[4]*1.1, thicks[5]*1.3, 35])
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    file_path = './Lab计算及膜厚范围.xlsx'
    layer_num, n0 = 8, 1

    theory_para, thickness = read_data(file_path)
    Rs_cx = cal_Rs(theory_para, refer_thickness_cx, layer_num, n0)

    return Rs_cx