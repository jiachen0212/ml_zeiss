import os
import json
import pandas as pd
from pandas.core.frame import DataFrame
from utils import plot_all_features

from local_settings import *

class SensorData():
    important_features_by_zeiss = ["$Date", "ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH",
                                  "ACT_V1_THERMOVAC_PREVLINE", "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2",
                                  "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP", "ACTN_F1_FLOW1", "ACT_F1_FLOW1",
                                  "ACT_O1_QCMS_THICKNESS", "ACT_O1_QCMS_RATE", "ACT_O1_QCMS_THICKNESS_CH1",
                                  "ACT_O1_QCMS_RATE_CH1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE",
                                  "ACT_Q10_CURRENT_ANODE", "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE",
                                  "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL", "ACT_Q10_ION_FLOW1",
                                  "ACT_Q10_ION_FLOW2", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]

    def __init__(self, sensor_data_df, original_process_timestamp, original_filename):
        self.data = sensor_data_df
        self.original_filename = original_filename
        self.timestamp = pd.to_datetime(DataFrame(eval(original_process_timestamp))[0]) #  pd.to_datetime(eval(original_process_timestamp))
        self.processed_data = []

        self.merge_time()
        self.filter_features()

    def preprocess(self,global_sensor_static,using_min_max = False):
        self.norm_features(global_sensor_static,using_min_max)
        self.split_feature_according_to_process_step()
        self.feature_engineering_for_senser_time_series()

    def merge_time(self):
        self.data["$Date"] = self.data.apply(lambda x: x["$Date"] + " " + x["$Time"], axis=1)

    def filter_features(self):
        ziss_important_columns = self.important_features_by_zeiss
        self.data = self.data[ziss_important_columns].copy()

    def norm_features(self, global_sensor_static, using_min_max = False):
        if using_min_max:
            print("using quartile 0.1 & 0.9 as min & max to norm")
            self.data.iloc[:,1:] = (self.data.iloc[:,1:] - global_sensor_static["quartile_0_1"])/(global_sensor_static["quartile_0_9"] - global_sensor_static["quartile_0_1"])
        else:
            self.data.iloc[:,1:] = (self.data.iloc[:,1:] - global_sensor_static["mean"])/global_sensor_static["std"]

    def split_feature_according_to_process_step(self):
        self.data.loc[:,"$Date"] = pd.to_datetime(self.data["$Date"])
        self.data.set_index("$Date", drop=True)
        timestamp_index = 0
        pre_index = 0
        for index in range(len(self.data)):
            current = self.data.loc[index,"$Date"]
            if timestamp_index == len(self.timestamp):
                break
            if current >= self.timestamp[timestamp_index]:
                timestamp_index += 1
                self.processed_data.append(self.data[pre_index:index])
                pre_index = index
        if pre_index < len(self.data)-1:
            self.processed_data.append(self.data[pre_index:])

    def save_original_fig(self):
        if not self.processed_data:
            raise RuntimeError("call preprocess first!")
        record_number, feature_number = self.processed_data[0].shape

        features_per_image = 10
        current_feature_index = 1
        counter = 0
        while current_feature_index < feature_number:
            end_index = current_feature_index + features_per_image

            dir_path, file_name = os.path.split(self.original_filename)

            target_path = dir_path + os.sep + str(counter) + "_" + file_name[:-4] + ".png"
            plot_all_features(self.processed_data, False, target_path,(current_feature_index,end_index))
            counter += 1
            current_feature_index = end_index

    def feature_engineering_for_senser_time_series(self):
        # min max mean std
        # 积分面积
        print("TODO feature_engineering_for_senser_time_series")

class SensorDataHandler():

    def __init__(self, path_cc, path_cx):
        self.path_cc = path_cc
        self.path_cx = path_cx

        self.cc_sensor_info_dict = dict()
        self.cx_sensor_info_dict = dict()

        self.global_sensor_concat = None
        self.global_statistic = None

        self.get_all_data()

    def preprocess(self,save_fig = False):
        for key, val in self.cc_sensor_info_dict.items():
            val.preprocess(self.get_global_statistic())
            if save_fig:
                val.save_original_fig()
        for key, val in self.cx_sensor_info_dict.items():
            val.preprocess(self.get_global_statistic())
            if save_fig:
                val.save_original_fig()

    def get_all_data_info(self, path_cc, path_cx):
        cc_data = pd.read_csv(path_cc)
        cx_data = pd.read_csv(path_cx)
        return cc_data, cx_data

    def get_global_statistic(self, force_update=False):
        if not force_update:
            with open(path_sensor_static, "r", encoding="utf-8") as reader:
                self.global_statistic = json.load(reader)

        if self.global_statistic is None or force_update:
            self._concate_global_sensor_data()
            self.global_statistic = dict()
            self.global_statistic["mean"] = self.global_sensor_concat.mean().to_json()
            self.global_statistic["std"] = self.global_sensor_concat.std().to_json()
            self.global_statistic["min"] = self.global_sensor_concat.min().to_json()
            self.global_statistic["max"] = self.global_sensor_concat.max().to_json()
            self.global_statistic["quartile_0_1"] = self.global_sensor_concat.quantile(0.1).to_json()
            self.global_statistic["quartile_0_25"] = self.global_sensor_concat.quantile(0.25).to_json()
            self.global_statistic["quartile_0_75"] = self.global_sensor_concat.quantile(0.75).to_json()
            self.global_statistic["quartile_0_9"] = self.global_sensor_concat.quantile(0.9).to_json()
            print("using fresh sensor static data!")
            
            out_path = os.path.join(dir_out_base, "global_sensor_statistic.json")
            with open(out_path,"w",encoding="utf-8") as writer:
                json.dump(self.global_statistic, writer)
            print("global sensor static is updated to {}. Move it to resources/data?".format(out_path))

        for key,val in self.global_statistic.items():
            self.global_statistic[key] = pd.Series(eval(val))
        return self.global_statistic

    def _concate_global_sensor_data(self):
        if self.global_sensor_concat is None:
            tmp = []
            for key, val in self.cc_sensor_info_dict.items():
                tmp.append(val.data)
            for key, val in self.cx_sensor_info_dict.items():
                tmp.append(val.data)

            self.global_sensor_concat = pd.concat(tmp).reset_index(drop=True)

    def filter_sensor_data(self, df):
        index = df["path_sensor_data"].apply(lambda x: os.path.isfile(x))
        df = df[index]
        return df

    def get_all_data(self):
        cc_data, cx_data = self.get_all_data_info(r'D:\Work\codes\ML_ZEISS\resources\data\0910_cc_data.csv',
                                        r'D:\Work\codes\ML_ZEISS\resources\data\0910_cx_data.csv')
        cc_data = self.filter_sensor_data(cc_data)
        cx_data = self.filter_sensor_data(cx_data)

        for row in cc_data.itertuples():
            file_path = getattr(row, "path_sensor_data")
            tmp = SensorData(pd.read_csv(file_path), row.Step_start_timestamp, file_path)

            self.cc_sensor_info_dict[file_path] = tmp
        for row in cx_data.itertuples():
            file_path = getattr(row, "path_sensor_data")
            tmp = SensorData(pd.read_csv(file_path), row.Step_start_timestamp, file_path)
            self.cx_sensor_info_dict[file_path] = tmp


if __name__ == "__main__":
    sdh = SensorDataHandler(r'D:\Work\codes\ML_ZEISS\resources\data\0910_cc_data.csv',
                             r'D:\Work\codes\ML_ZEISS\resources\data\0910_cx_data.csv')

    sdh.get_global_statistic()

    # 开启 画sensor的fig
    sdh.preprocess(True)

    print("DONE!")
