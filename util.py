# coding=utf-8
from math import fabs, copysign

import numpy as np
import xlrd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

# plot colre_names:
cnames = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'black': '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgray': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'}

'''
输入lab曲线 best 即可得到lab值
'''


def fun1(x, y, s):
    a = np.sum([x[i] * s[i] for i in range(81)])
    b = np.sum([y[i] * s[i] for i in range(81)])
    res = 100 * a / b
    return res


def fun2(x, y, s, r):
    a = np.sum([x[i] * s[i] * r[i] for i in range(81)])
    b = np.sum([y[i] * s[i] for i in range(81)])
    res = a / b
    return res


def fun3(Xxn):
    if Xxn > 0.008856:
        fXxn = copysign(fabs(Xxn) ** (1 / 3), Xxn)
    else:
        fXxn = 7.787 * Xxn + 16 / 116

    return fXxn


def calculate_Lab(best):
    # best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
    #         1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
    #         0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
    #         0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
    #         3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    remda = [380 + 5 * i for i in range(81)]
    XYZ_fun = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\Lab计算及膜厚范围.xlsx'
    wb = xlrd.open_workbook(XYZ_fun)
    data = wb.sheet_by_name(r'色分配函数')
    fx = data.col_values(2)[4:]
    # print(fx, 'fx')
    fy = data.col_values(3)[4:]
    # print(fy, 'fy')
    fz = data.col_values(4)[4:]
    # print(fz, 'fz')
    Xn = fun1(fx, fy, remda)
    Yn = fun1(fy, fy, remda)
    Zn = fun1(fz, fy, remda)
    X = fun2(fx, fy, remda, best)
    Y = fun2(fy, fy, remda, best)
    Z = fun2(fz, fy, remda, best)
    Xxn = X / Xn
    Yyn = Y / Yn
    Zzn = Z / Zn
    fXxn = fun3(Xxn)
    fYyn = fun3(Yyn)
    fZzn = fun3(Zzn)
    if Yyn > 0.008856:
        L = 116 * copysign(fabs(Yyn) ** (1 / 3), Yyn) - 16
    else:
        L = 903.3 * Yyn
    a = 500 * (fXxn - fYyn)
    b = 200 * (fYyn - fZzn)
    print("Lab value: L: {}, a: {}, b: {}".format(L, a, b))
    return L, a, b


def top_k_feature(remed, Y, X, all, n):
    Y = np.array([a[remed] for a in Y])
    X_slim = SelectKBest(f_regression, k=n).fit_transform(X, Y)
    a = X[0].tolist()
    b = X_slim[0].tolist()
    for i in b:
        all.append(a.index(i))


def Select_feature(X, Y):
    '''

    :param X: numpy
    :param Y: list
    :return:
    '''
    import_index = [0, 5, 12, 52, 74, 80]
    # 对14层膜厚之后的121维特征做重要性筛选
    X = [a[14:] for a in X]
    X = np.array(X)
    X[np.isnan(X)] = 0.0
    # X数据规整化
    scale = StandardScaler(with_mean=True, with_std=True)
    X = scale.fit_transform(X)
    # sklearn 实现特征构造
    # X = PolynomialFeatures(degree=2).fit_transform(X)
    all = []
    for i in import_index:
        # 每个频段选top20
        top_k_feature(i, Y, X, all, n=20)
    slim_feature = list(set(all))
    print("特征筛选后的特征维度: {}".format(len(slim_feature)))
    # X_slim = None
    X_slim = X[:, :14]  # n,14
    for ind in slim_feature:
        if X_slim is not None:
            tmp = np.reshape(X[:, ind], (-1, 1))
            X_slim = np.hstack((X_slim, tmp))
        else:
            X_slim = X[:, ind]
    return X_slim
