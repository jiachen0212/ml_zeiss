
# 概述

* ML工艺优化代码


## data/ 数据预处理
1. dataloader.py 
	读入数据, 关联: 炉号, file_name, thickness_list, lab_curve, LAB_value, machine_start_time, 耗材, clean_index 等信息

2. features.py
	sensor 数据特征工程文件

3. recordFilter.py
	不同模型需要对数据进行不同的filter, 在此实现

4. pipline.py
	load_data + filter_data


## resources/ 
### data/ 
	1. 存放中间落盘数据
### doc/
	1. 存放阶段性ppt, docs等实验文档记录



## modeling
	doing ..


### cycleBased
	doing ..


### markovChain
	doing ..


## utils
	doing ..