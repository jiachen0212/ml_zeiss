# coding=utf-8
import json
import os

cccx_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'

# 先从异常样本下手
bad_sample = ['33121051807', '33121051808', '33121051809', '33121051801']
number_evts = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\num_evt12.json', 'r'))

useful_info = ['ACT_V1_IONIVAC_CH', 'ACT_V1_PENNINGVAC_CH', 'ACT_V1_THERMOVAC_CH', 'ACT_V1_THERMOVAC_PREVLINE', 'ACT_V1_THERMOVAC_HP', 'ACT_V1_THERMOVAC_HP2', 'ACT_V1_PRESSURE_CH', 'AI_V1_POLYCOLD_TEMP', 'ACTN_F1_FLOW1', 'ACT_F1_FLOW1', 'ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1', 'STAT_LT_CRYSTAL_CH1', 'ACT_HEATER2_TEMPERATURE', 'ACT_Q10_CURRENT_ANODE', 'ACT_Q10_VOLTAGE_ANODE', 'ACT_Q10_CURRENT_CATHODE', 'ACT_Q10_VOLTAGE_CATHODE', 'ACT_Q10_CURRENT_NEUTRAL', 'ACT_Q10_ION_FLOW1', 'ACT_Q10_ION_FLOW2', 'STA_Q10_IONSOURCE_SHUTTER_IOP', 'ACT_V1_MEISSNER_POLYCOLDTEMP']


