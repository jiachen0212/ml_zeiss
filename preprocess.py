import json

from evt import *
from res_mapping import *


def get_records(path2lab_values, path2evt_dirs, machine_filer, pecipe_filer):
    lab_records = get_lab_values(path2lab_values)
    records, evt_process_counter = get_evt_records(path2evt_dirs, machine_filer,
                                                   pecipe_filer)
    in_record = 0
    out_record = 0
    for key, value in records.items():
        key = int(key)
        if key in lab_records:
            lab = lab_records[key]
            value.set_lab(lab)
            in_record += 1
        else:
            out_record += 1
            print(key, "has no lab record!")
    return records, evt_process_counter


def overview_dataset(records, evt_process_counter, path_base):
    overview_evt_record(evt_process_counter)
    overview_lab_values(records, path_base)


def overview_evt_record(evt_process_counter):
    print(evt_process_counter)
    process_dict = evt_process_counter["process_name"]
    print("{} kinds of precess.".format(len(process_dict)))
    for process_name in process_dict.keys():
        print("‘{}’".format(process_name), end=",")
    print()


def overview_lab_values(records, path_base):
    length_counter = dict()
    lf_counter = 0
    af_counter = 0
    bf_counter = 0
    sf_counter = 0
    sample_counter = 0
    ok_sample_counter = 0

    layer_filename = {}

    for key, value in records.items():
        sample_counter += 1
        lab_values = value.lab_record
        for lv in lab_values.bool_l:
            if not lv:
                lf_counter += 1
        for av in lab_values.bool_a:
            if not av:
                af_counter += 1
        for bv in lab_values.bool_b:
            if not bv:
                bf_counter += 1

        layer_number = len(lab_values.bool_l)
        length_counter[layer_number] = length_counter.get(layer_number, 0) + 1

        name_list = layer_filename.get(layer_number, [])
        name_list.append(key)
        layer_filename[layer_number] = name_list

        sample_ok = True
        for layer_value in lab_values.bool_layer:
            if not layer_value:
                sample_ok = False
                sf_counter += 1
        if sample_ok:
            # print("{}, OK!".format(key))
            ok_sample_counter += 1

    writer = open(os.path.join(path_base, "layer_namelist.json"), "w", encoding="utf-8")
    json.dump(layer_filename, writer)

    print("l value NG:", lf_counter)
    print("b value NG:", bf_counter)
    print("a value NG:", af_counter)
    print("lab value NG:", sf_counter)
    print("sample number:", sample_counter)
    print("sample NG:", sample_counter - ok_sample_counter)
    print("sample OK:", ok_sample_counter)
    print("sample NG ratio:", (sample_counter - ok_sample_counter) / sample_counter)
    print("layer counter:", length_counter)


def select_abnormal_process(records, evt_process_counter, path_base):
    present_info = dict()
    miss_info = dict()

    processes = evt_process_counter["process_name"].keys()
    for item in processes:
        present_info[item] = []
        miss_info[item] = []

    for index, record in records.items():
        process = record.record_info["process_name"]
        for target in processes:
            if target in process:
                present_info[target].append(index)
            else:
                miss_info[target].append(index)
    writer = open(os.path.join(path_base, "process_miss.json"), "w", encoding="utf-8")
    json.dump(miss_info, writer)
    writer = open(os.path.join(path_base, "process_present.json"), "w", encoding="utf-8")
    json.dump(present_info, writer)


if __name__ == "__main__":
    path_base = r"D:\work\project\卡尔蔡司AR镀膜\AR big data_20210310"
    path2lab_values = os.path.join(path_base, r"lab_result\20210310.xls")
    path2evt_dirs = os.path.join(path_base, r"files")
    machine_filer = {"P6768-33#"}
    pecipe_filer = {"1.6&1.67_DVS_CC"}  # {"1.6&1.67_DVS_CC", "1.67US_DVS_CC"}

    records, evt_process_counter = get_records(path2lab_values, path2evt_dirs, machine_filer, pecipe_filer)

    overview_dataset(records, evt_process_counter, path_base)

    select_abnormal_process(records, evt_process_counter, path_base)
