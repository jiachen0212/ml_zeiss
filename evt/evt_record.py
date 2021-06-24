import os


class Record():
    """
    hold info about one evt record.
    """

    def __init__(self, path2record, target_machine=None, target_pecipe=None):
        self.dir_record_base, self.filename = os.path.split(path2record)
        self.record_info = self._load_record(path2record, target_machine, target_pecipe)
        self.abnormal_tag = False
        self.lab_record = None

    def check_process_legal(self):
        if self.record_info:
            return True
        else:
            return False

    def set_lab(self, lab_record):
        self.lab_record = lab_record

    def _load_record(self, path2record, target_machine, target_pecipe):
        reader = open(path2record, "r")
        feedback = dict()
        process_name = dict()
        for index, line in enumerate(reader):
            line = line.strip()
            if not line:
                continue
            elif index == 2:
                machine_name = line.split(",")[1]
                if target_machine and machine_name not in target_machine:
                    print(path2record,
                          "machine_name: {} not in target_machine: {}".format(machine_name, target_machine))
                    return None
                feedback["machine_name"] = machine_name
            elif index == 3:
                pecipe_name = line.split(",")[1]
                if target_pecipe and pecipe_name not in target_pecipe:
                    print(path2record,
                          "pecipe_name: {} not in target_pecipe: {}".format(pecipe_name, target_pecipe))
                    return None
                feedback["pecipe_name"] = pecipe_name
            elif index >= 5:
                spline = line.split(",")
                if len(spline) > 2:
                    key = spline[2]
                else:
                    print(path2record, "have an unknow process in line {}: {}".format(index, line))
                    key = spline[1]
                    self.abnormal_tag = True
                process_name[key] = process_name.get(key, 0) + 1
        feedback["process_name"] = process_name
        return feedback
