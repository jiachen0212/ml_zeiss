import pandas as pd


# def get_lab_values(path2lab_record):
#     excel = pd.read_excel(path2lab_record)
#     values_excel = excel.values
#
#     illegal_l = []
#     illegal_a = []
#     illegal_b = []
#     illegal_sample = []
#
#     feedback = dict()
#
#     for record in values_excel:
#         # L值	A值	B值	L范围	A范围	B范围	文件名	文件ID
#         key = record[7]
#         values = dict()
#         value_l = record[0]
#         value_a = record[1]
#         value_b = record[2]
#
#         bool_l = 3.30 < value_l < 6.80
#         bool_a = -2. < value_a < 2
#         bool_b = -18.00 < value_b < 6.80
#         bool_sample = bool_l and bool_a and bool_b
#         if not bool_l:
#             illegal_l.append(key)
#         if not bool_a:
#             illegal_a.append(key)
#         if not bool_b:
#             illegal_b.append(key)
#         if not bool_sample:
#             illegal_sample.append(key)
#
#         values["l"] = value_l
#         values["a"] = value_a
#         values["b"] = value_b
#         values["bool_l"] = bool_l
#         values["bool_a"] = bool_a
#         values["bool_b"] = bool_b
#         values["bool_sample"] = bool_sample
#
#         feedback[key] = values
#
#
#     print("illegal_l",illegal_l)
#     print("illegal_a",illegal_a)
#     print("illegal_b",illegal_b)
#     print("illegal_sample",illegal_sample)
#     print("toral_records", len(values_excel))
#     print("OK", len(values_excel)-len(illegal_sample))
#     print("NG:\n\tl {}\n\ta {}\n\tb {}\n\tsamples {}".format(len(illegal_l),len(illegal_a),len(illegal_b),len(illegal_sample)))
#     return feedback

class LABRecord():

    def __init__(self, raw_record=None):
        self.value_l = []
        self.value_a = []
        self.value_b = []

        self.bool_l = []
        self.bool_a = []
        self.bool_b = []

        self.bool_layer = []

        if raw_record:
            self.update(raw_record)

    def update(self, raw_record):
        value_l = raw_record[0]
        value_a = raw_record[1]
        value_b = raw_record[2]

        self.value_l.append(value_l)
        self.value_a.append(value_a)
        self.value_b.append(value_b)

        bool_l = 3.30 < value_l < 6.80
        bool_a = -2. < value_a < 2
        bool_b = -18.00 < value_b < 6.80
        self.bool_l.append(True if bool_l else False)
        self.bool_a.append(True if bool_a else False)
        self.bool_b.append(True if bool_b else False)

        if bool_l and bool_a and bool_b:
            self.bool_layer.append(True)
        else:
            self.bool_layer.append(False)


def get_lab_values(path2lab_record):
    excel = pd.read_excel(path2lab_record)
    values_excel = excel.values

    feedback = dict()

    for record in values_excel:
        key = record[7]
        tmp = feedback.get(key, LABRecord())
        tmp.update(record)
        feedback[key] = tmp

    return feedback


if __name__ == "__main__":
    path2lab_values = r"D:\Work\projects\ZEISS\Data\AR big data_20210310\mapping\20210310.xlsx"
    lab_records = get_lab_values(path2lab_values)
    print("DONE！")
