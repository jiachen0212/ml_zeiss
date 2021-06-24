import os

from evt import Record


def get_evt_records(dir_records, machine_filer=None, pecipe_filer=None):
    print("Filter:", machine_filer, pecipe_filer)
    dir_record_names = os.listdir(dir_records)
    info_all_records = dict()
    filtered_records = dict()
    for dir_name in dir_record_names:
        filename = "evt" + dir_name + ".csv"
        path_single_record = os.path.join(dir_records, dir_name, filename)
        record = Record(path_single_record, machine_filer, pecipe_filer)
        if record.record_info:
            info_all_records = merge_record_dict(info_all_records, record.record_info)
            filtered_records[dir_name] = record
    return filtered_records, info_all_records


def merge_record_dict(record1, record2):
    if not record2:
        return record1

    if "machine_name" not in record1:
        record1["machine_name"] = dict()
    if "pecipe_name" not in record1:
        record1["pecipe_name"] = dict()

    machine_dict = record1["machine_name"]
    machine_dict[record2.get("machine_name", "empty")] = machine_dict.get(record2.get("machine_name", "empty"), 0) + 1
    pecipe_dict = record1["pecipe_name"]
    pecipe_dict[record2.get("pecipe_name", "empty")] = pecipe_dict.get(record2.get("pecipe_name", "empty"), 0) + 1

    his_process = record1.get("process_name", dict())
    current_process = record2.get("process_name", dict())

    for key, value in current_process.items():
        his_process[key] = his_process.get(key, 0) + value

    record1["process_name"] = his_process
    return record1


if __name__ == "__main__":
    # machine_filer = None
    # pecipe_filer = None

    machine_filer = {"P6768-33#"}
    pecipe_filer = {"1.6&1.67_DVS_CC"}  # {"1.6&1.67_DVS_CC", "1.67US_DVS_CC"}

    records_counter, records = get_evt_records(r"D:\Work\projects\ZEISS\Data\AR big data_20210310\files", machine_filer,
                                               pecipe_filer)
    print(records_counter)
