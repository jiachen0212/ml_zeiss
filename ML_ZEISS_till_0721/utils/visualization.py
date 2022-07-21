import matplotlib.pyplot as plt
import math

def plot_all_features(list_of_dataframe,show=True,save_path=None,feature_range=None):
    cols = 3
    rows = math.ceil(len(list_of_dataframe)/cols)
    dpi = 100
    plt.figure(figsize=(800 / dpi, 800 / dpi), dpi=dpi)
    for index, df in enumerate(list_of_dataframe):
        ax = plt.subplot(rows, cols, index+1)
        # if index>0:
        #     ax.legend().set_visible(False)
        plt.xlabel("process step {}".format(index))
        if feature_range is None:
            df.iloc[:, 1:].plot(ax=ax,legend = False if index>0 else True)
        else:
            start_index, end_index = feature_range
            df.iloc[:, start_index:end_index].plot(ax=ax, legend=False if index > 0 else True)
    if save_path:
        print("saving sensor feature image to {}".format(save_path))
        plt.savefig(save_path,dpi=dpi * 20)
    if show:
        plt.show()
    plt.clf()