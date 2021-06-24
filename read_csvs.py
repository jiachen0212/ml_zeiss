# coding=utf-8
import os


def get_evt_sensor_dict(sensor_lines, process_lines, dir_name):
    '''
    evt_dict = dict()   # time:process_name

    sensor.csv info:
    running_times = []
    sensor_values = []

    sensor_index_dict = dict()
    '''
    evt_dict = dict()
    for index, line in enumerate(process_lines):
        split_line = line.strip()
        if not split_line:
            continue
        if index >= 5:
            spline = split_line[:-1].split(",")
            if len(spline) > 2:
                # 这是个坑点, 01:04:37 统一写成 1:04:37
                process_time = spline[0].lstrip('0')
                process_name = spline[2]
                # 一个process运转时间可能记录了多个工艺名称
                if process_time in evt_dict.keys():
                    evt_dict[process_time].append(process_name)
                else:
                    evt_dict[process_time] = []
                    evt_dict[process_time].append(process_name)
            else:
                print(dir_name, "have an unknown process in line {}: {}".format(index, split_line))
                # 暂不做什么处理

    running_times = []
    sensor_values = []
    sensor_names = []
    for index, line in enumerate(sensor_lines):
        split_line = line.strip()
        if not split_line:
            continue
        spline = split_line[:-1].split(',')
        if index == 0:
            # 所有sensor参数名
            sensor_names = spline[5:]
        else:
            # 同上evt_dict时间统一格式处理
            running_times.append(spline[1].lstrip('0'))
            sensor_values.append(spline[5:])

    return evt_dict, running_times, sensor_values, sensor_names


def get_time_sensor(evt_dict, running_times, sensor_values, dir_name):
    '''
    evt_dict: time: process
    sensor.csv: running_times, sensor_values

    '''
    time_sensor = dict()
    evt_process_time = list(evt_dict.keys())[:-1]
    lens = len(evt_process_time)
    for i in range(lens - 1):
        start, end = evt_process_time[i], evt_process_time[i + 1]
        try:
            sensor_start = running_times.index(start)
            sensor_end = running_times.index(end)
        except:
            # print("{} has some bug, evt and sensor_value.".format(dir_name))
            continue  # 跳出此次dir_name统计
        # 最后evt中的时间点,sensor.csv中可能找不到,数据就直接读取到sensor.csv的最后一行
        if i == lens - 2:
            sensor_end = len(running_times)
        # check time
        # print(dir_name, start, end)
        # print(running_times[sensor_start-1], running_times[sensor_start], \
        # running_times[sensor_end], running_times[sensor_end+1])
        cur_sensor_value = sensor_values[sensor_start:sensor_end]
        time_sensor[start] = cur_sensor_value
    # check
    # tmp = list(time_sensor.keys())[2]
    # print(dir_name, tmp, time_sensor[tmp][0])
    return time_sensor


def process_sensor(evt_dict, time_sensor):
    '''

    :param evt_dict:
    :param time_sensor:
    :return:
    time is the tmp_param
    '''
    evt_times = list(evt_dict.keys())[:-1]
    # 根据evt的time,对应获取sensor_value
    process_sensor_dict = dict()
    for time in evt_times:
        try:
            sensor_value_list = time_sensor[time]
        except:
            # print(dir_name, time)   # evt文件末尾的几行,无信息量
            continue
        # 一个工艺时间对应多个工艺名称(多个工艺都在执行?),所以for循环下
        # 暂时无法知道一个时间点多个工艺进行中,各个sensor参数值怎么对应.所以暂时都统一copy给各个工艺
        processes = evt_dict[time]
        for process in processes:
            process_sensor_dict[process] = sensor_value_list

    return process_sensor_dict


def get_info(path, miss_info):
    '''
    :param path: dir path
    :return: all_dir's process_sensor_value_dict. a list

    '''
    miss_dir = open(miss_info, 'w')
    all_process_sensor = dict()
    all_sensor_index_dict = dict()
    dir_names = os.listdir(path)
    for dir_name in dir_names:
        sensor_csv = os.path.join(path, dir_name, "{}.csv".format(dir_name))
        process_csv = os.path.join(path, dir_name, "evt{}.csv".format(dir_name))
        try:
            sensor_lines = open(sensor_csv, "r")
            process_lines = open(process_csv, "r")
        except:
            # print("{} loss evt or csv file".format(dir_name))
            # 落盘csv损失的文件夹名称至txt
            miss_dir.write(dir_name + '\n')
            continue
        # step1.
        # 为每一对csv文件,存储evt, sensor dict{}
        evt_dict, running_times, sensor_values, sensor_names = get_evt_sensor_dict(sensor_lines, process_lines,
                                                                                   dir_name)

        # step2.
        # 时间和sensor_value关联:
        # evt中最后一行的时间在sensor中可能找不到,它是evt的end信息
        time_sensor = get_time_sensor(evt_dict, running_times, sensor_values, dir_name)

        # step3.
        # process_name 和 sensor_value联系起来
        process_sensor_dict = process_sensor(evt_dict, time_sensor)
        all_process_sensor[dir_name] = process_sensor_dict
        all_sensor_index_dict[dir_name] = sensor_names
    miss_dir.close()

    return all_process_sensor, all_sensor_index_dict


def process_sensor_param_dict(all_process_sensor, all_sensor_name):
    '''
    :param all_process_sensor: dir_name: each_dict
    each_dict: process_name: sensor_value_list
    sensor_index_dict: index, sensor_name
    :return:

    '''
    csv_dict = dict()
    for dir_name, process_sensor in all_process_sensor.items():
        sensor_name_list = all_sensor_name[dir_name]
        dict_ = dict()
        for process_name, sensor_value_list in process_sensor.items():
            tmp_dict = dict()
            for index, each_sensor in enumerate(sensor_name_list):
                # 获取每一sensor参数的,所有列数据值. each_sensor_name: [value]
                tmp_dict[each_sensor] = [row[index] for row in sensor_value_list]
            dict_[process_name] = tmp_dict
        csv_dict[dir_name] = dict_

    # {dir_name: {process_name: {each_sensor: [value_list]}}}
    return csv_dict


if __name__ == "__main__":
    base_path = r"D:\work\project\卡尔蔡司AR镀膜\AR big data_20210310"
    path = os.path.join(base_path, r"files")
    miss_info = os.path.join(base_path, r"miss_dir.txt")
    all_process_sensor, all_sensor_name = get_info(path, miss_info)
    csv_dict = process_sensor_param_dict(all_process_sensor, all_sensor_name)

    # test
    '''
    for dir_name, v in csv_dict.items():
        for process, sensor in v.items():
            for s_name, value in sensor.items():
                print(s_name, len(value))
    '''
