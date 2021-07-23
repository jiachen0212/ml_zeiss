import os
import xlrd

'''
deta 模型验证.
寻找时序上连续的一些样本,做10维 deta_thickness 到 81维 lab 曲线的映射.

2021.07.23 chen_jia 

'''


data_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'
evts = os.listdir(data_dir)
evts = [i for i in evts if "evt" in i]

# 1. 先按天划分样本[一天最多40evt, 最少1evt],
# 然后每一天内的样本,check下时间是否连续.[done, 时间连续]
day_evts = dict()
for evt in evts:
    path = os.path.join(data_dir, evt)
    f = open(path, 'r')
    for index, line in enumerate(f):
        if 'Production Start :' in line:
            day = line.split(',')[1]
            if day not in day_evts:
                day_evts[day] = [[line.split(',')[2], evt]]
            else:
                day_evts[day].append([line.split(',')[2], evt])

part_cycle_evts = []
# note. 注意到这里时间跨度比较大,可能1/7, 1/14机器状态就出现了较大的变化(虽然这一块数据没有给到我们)
# 模型验证后, 若拟合不佳,可考虑把天数gap再缩减下, 控制在连续三天内.
# residual_day = ['1/6/2021', '1/7/2021', '1/14/2021', '1/25/2021', '1/26/2021']
residual_day = ['5/13/2021', '5/14/2021', '5/15/2021', '5/16/2021', '5/17/2021']
for res_day in residual_day:
    for n in day_evts[res_day]:
        part_cycle_evts.append(n[1])


# 2. 还是需要控制在一个清洗周期内,探索n个连续样本产生的n-1条deta数据, deta_thick到deta_lab的关系
number_evt_pair = r'D:\work\project\卡尔蔡司AR镀膜\第三批\匹配关系2021.1~2021.6.xlsx'
wb = xlrd.open_workbook(number_evt_pair)
data = wb.sheet_by_name('Sheet1')
rows = data.nrows
cols = data.ncols
evt_number = dict()
for i in range(1, rows):
    number = data.cell(i, 2).value
    evt_name = data.cell(i, 5).value
    evt_number[(evt_name.lower())+'.csv'] = number

for k, v in evt_number.items():
    print(k, v)

part_clean_cycle_number = r'./clean_cycle_number.txt'
f = open(part_clean_cycle_number, 'w')
for n in part_cycle_evts:
    try:
        number = evt_number[n]
        f.write(number+',')
    except:
        continue