# coding=utf-8
import json
import os

# wb = pd.read_excel(file, engine='openpyxl')
import xlrd


def evt_33(file, evt_cc_dir):
    '''
    :return:
    获取到33#膜色文件与EVT文件对应表.xlsx文件中的evt文件列表;
    以上evt文件列表, 包含cx和cc. 即正反面, 在1.6&1.67_DVS_CC中挑选出存在的evt_files

    '''
    # step1.
    wb = xlrd.open_workbook(file)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    evt_dict = dict()  # evt_name: 33321043002编号
    for i in range(1, rows):
        # print(data.row_values(i))  # 每一行都是一个list
        evt_dict[data.cell(i, 5).value + '.CSV'] = data.cell(i, 2).value
    # step2.
    evt_dict_keys = list(evt_dict.keys())
    evt_cc_list = os.listdir(evt_cc_dir)
    for evt in evt_dict_keys:
        if evt not in evt_cc_list:
            del evt_dict[evt]
    # print(evt_dict)   # 167个
    # evt_dict：可在1.6&1.67_DVS_CC下找到膜厚值,且可根据33#膜色文件与EVT文件对应表.xlsx找到对应的33321043002编号
    return evt_dict


# step3.
def lab_surve33(file, evt_dict, data_js):
    '''
    :return:
    处理下膜色数据.xlsx, 把33321043002编号和膜色曲线list_value关联起来, 并且直接取第四层数据
    处理evt_dict 和 lab_curve, 关联起膜厚值和膜色曲线

    '''
    # step3.
    wb = xlrd.open_workbook(file)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    numberss_dict = dict()
    lab_curve = dict()
    for i in range(1, rows):
        numberss_dict[data.cell(i, 2).value] = numberss_dict.get(data.cell(i, 2).value, 0) + 1
        # 我们取第四层的膜色曲线为基准
        if numberss_dict[data.cell(i, 2).value] == 4:
            # print(data.cell(i, 2).value, data.row_values(i)[19:-1])
            lab_curve[data.cell(i, 2).value] = data.row_values(i)[19:-1]
    # print(lab_curve)    # 33121060503: [膜色曲线list value]

    # step4.
    evt_name_lab_curve = dict()
    for evt_name, number in evt_dict.items():
        if number in lab_curve:
            evt_name_lab_curve[evt_name] = lab_curve[number]
    thickness_lab_curve = dict()
    for evt_name in evt_name_lab_curve:
        evt_path = os.path.join(evt_cc_dir, evt_name)
        thickness = []
        with open(evt_path, 'r') as file:
            for line in file:
                if "Thickness" in line:
                    thickness.append(line.split(',')[4])
        thickness_lab_curve[''.join(i + ',' for i in thickness) + '{}'.format(evt_name[:-4])] = evt_name_lab_curve[
            evt_name]
    # Y = []
    # for thickness, lab_curve in thickness_lab_curve.items():
    #     Y.append(thickness_lab_curve[thickness])

    # 做一个thickness_lab_curve 的落盘
    data = json.dumps(thickness_lab_curve)
    with open(data_js, 'w') as js_file:
        js_file.write(data)


if __name__ == "__main__":
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thickness_lab_curve.json'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'

    evt_dict = evt_33(file1, evt_cc_dir)
    lab_surve33(file2, evt_dict, data_js)

    # test
    # with open(data_js, encoding="utf-8") as reader:
    #     thickness_lab_curve = json.load(reader)
    #     print(thickness_lab_curve)

    # 画一画原始的lab_curve曲线,看看和拟合出来的差异
    # 数据量化下, 把各个频段的差异值(平方差,方差?)可以算一下(81dims)
