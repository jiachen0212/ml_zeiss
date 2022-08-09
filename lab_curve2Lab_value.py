def calculate_Lab(curve_):
    # S = [33.0, 39.92, 47.4, 55.17, 63.3, 71.81, 80.6, 89.53, 98.1, 105.8, 112.4, 117.75, 121.5, 123.45, 124.0, 123.6,
    #      123.1, 123.3, 123.8, 124.09, 123.9, 122.92, 120.7, 116.9, 112.1, 106.98, 102.3, 98.81, 96.9, 96.78, 98.0,
    #      99.94, 102.1, 103.95, 105.2, 105.67, 105.3, 104.11, 102.3, 100.15, 97.8, 95.43, 93.2, 91.22, 89.7, 88.83, 88.4,
    #      88.19, 88.1, 88.06, 88.0, 87.86, 87.8, 87.99, 88.2, 88.2, 87.9, 87.22, 86.3, 85.3, 84.0, 82.21, 80.2, 78.24,
    # 找客户要一下准确的光谱函数
    S = [float(a) for a in open('./D65.txt').readlines()]
    XYZ_fun = './Lab计算及膜厚范围.xlsx'
    wb = xlrd.open_workbook(XYZ_fun)

    data = wb.sheet_by_name(r'色分配函数')
    Xbar = data.col_values(5)[4:]
    Ybar = data.col_values(6)[4:]
    Zbar = data.col_values(7)[4:]

    def fun1(Xbar, Ybar, S):
        a = np.sum([Xbar[i] * S[i] for i in range(81)])
        b = np.sum([Ybar[i] * S[i] for i in range(81)])
        res = 100 * a / b
        return res

    Xn = fun1(Xbar, Ybar, S)
    Yn = fun1(Ybar, Ybar, S)
    Zn = fun1(Zbar, Ybar, S)

    def fun2(fx, fy, S, curve_):
        a = np.sum([fx[i] * S[i] * curve_[i] for i in range(81)])
        b = np.sum([fy[i] * S[i] for i in range(81)])
        res = a / b
        return res

    X = fun2(Xbar, Ybar, S, curve_)
    Y = fun2(Ybar, Ybar, S, curve_)
    Z = fun2(Zbar, Ybar, S, curve_)
    Xxn = X / Xn
    Yyn = Y / Yn
    Zzn = Z / Zn

    def fun3(Xxn):
        if Xxn > 0.008856:
            fXxn = copysign(fabs(Xxn) ** (1 / 3), Xxn)
        else:
            fXxn = 7.787 * Xxn + 16 / 116

        return fXxn

    fXxn = fun3(Xxn)
    fYyn = fun3(Yyn)
    fZzn = fun3(Zzn)
    if Yyn > 0.008856:
        L = 116 * copysign(fabs(Yyn) ** (1 / 3), Yyn) - 16
    else:
        L = 903.3 * Yyn
    a = 500 * (fXxn - fYyn)
    b = 200 * (fYyn - fZzn)

    return np.round(L, 3), np.round(a, 3), np.round(b, 3)
