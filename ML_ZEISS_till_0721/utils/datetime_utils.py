# coding=utf-8
import pandas as pd
import datetime

def check_one_day_offset(pre, post, file_name="unknow_file"):
    pre_dt = pd.to_datetime(pre)
    post_dt = pd.to_datetime(post)
    delta = post_dt - pre_dt
    if delta.delta < 0:
        oneday = datetime.timedelta(1)
        post_dt = post_dt + oneday
        print("detect one day offset in {}".format(file_name))
    return str(post_dt)