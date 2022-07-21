def filter_sensor_data(df):
    index = df["path_sensor_data"].apply(lambda x: os.path.isfile(x))
    df = df[index]
    return df


if __name__ == "__main__":
    cc_data, cx_data = get_all_data(r'.\resources\data\0910_cc_data.csv', r'.\resources\data\0910_cx_data.csv')

    cc_data = filter_sensor_data(cc_data)
    cx_data = filter_sensor_data(cx_data)
