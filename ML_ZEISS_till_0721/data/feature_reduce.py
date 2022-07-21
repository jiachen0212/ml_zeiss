import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import VarianceThreshold
# 移出低方差特征
# 剔除某一个值概率超过0.8的特征 方差最小为0.8*0.2=0.16
# sel = VarianceThreshold(threshold=0.16)
# sel.fit_transform(X)


def get_features(seeds):
    for seed in seeds:
        no_thick_sensor_train_features = dict()
        no_thick_sensor_test_features = dict()
        # oven_train = np.load(r'./train_oven_seed{}.npy'.format(seed))
        # oven_test = np.load(r'./test_oven_seed{}.npy'.format(seed))
        # oven_train = np.load(r'./1129train_ovens.npy')
        # oven_test = np.load(r'./1129test_ovens.npy')

        # load lstm_feature
        lstm_train_feature = json.load(open(r'./data1_5_lstm_train_features_seed{}.json'.format(seed), 'r'))
        lstm_test_feature = json.load(open(r'./data6_lstm_test_features_seed{}.json'.format(seed), 'r'))

        # load en_decoder_feature
        en_decoder_train_feature = json.load(open(r'./all_en_decoder_feature_train_seed{}.json'.format(seed), 'r'))
        en_decoder_test_feature = json.load(open(r'./all_en_decoder_feature_test_seed{}.json'.format(seed), 'r'))

        # tsf_feature1 = dict()
        # tsf_feature = json.load(open(r'./tsf_features_1129_all.json', 'r'))
        # # thickness_tsf_feature = json.load(open(r'./tsf_features_thickness.json', 'r'))
        # tsf_train_feature, tsf_test_feature = dict(), dict()
        # for train_o in oven_train:
        #     tsf_feature1[train_o] = tsf_feature[train_o] #  + thickness_tsf_feature[train_o]
        # # data = json.dumps(tsf_feature1)
        # # with open(r'./tsf_train_features{}.json'.format(seed), 'w') as js_file:
        # #     js_file.write(data)
        # # tsf_feature1 = dict()
        # for test_o in oven_test:
        #     tsf_feature1[test_o] = tsf_feature[test_o]  #  + thickness_tsf_feature[test_o]
        # data = json.dumps(tsf_feature1)
        # with open(r'./tsf_test_features{}.json'.format(seed), 'w') as js_file:
        #     js_file.write(data)

        tsf_train_feature = json.load(open(r'./data1_5_train_tsf_feature.json', 'r'))
        tsf_test_feature = json.load(open(r'./data6_test_tsf_feature.json', 'r'))

        for oven, f in tsf_train_feature.items():
            try:
                v = f + lstm_train_feature[oven] + en_decoder_train_feature[oven]
            except:
                continue
            assert len(v) == 21*(16+18+8)
            no_thick_sensor_train_features[oven] = v

        for oven, f in tsf_test_feature.items():
            try:
                v = f + lstm_test_feature[oven] + en_decoder_test_feature[oven]
            except:
                continue
            assert len(v) == 21*(16+18+8)
            no_thick_sensor_test_features[oven] = v

        data = json.dumps(no_thick_sensor_train_features)
        with open(r'./no_thick_train_features{}.json'.format(seed), 'w') as js_file:
            js_file.write(data)
        data = json.dumps(no_thick_sensor_test_features)
        with open(r'./no_thick_test_features{}.json'.format(seed), 'w') as js_file:
            js_file.write(data)
        assert len(no_thick_sensor_test_features) == 25

def get_seed_pca_feature():
    seeds = [33]
    pcas = [10, 20, 50, 100, 200, 300, 500]
    for pca in pcas:
        for seed in seeds:
            train_feature = json.load(open(r'./no_thick_train_features{}.json'.format(seed), 'r'))
            test_feature = json.load(open(r'./no_thick_test_features{}.json'.format(seed), 'r'))
            # train_feature = json.load(open(r'./all_sensor_train_features{}.json'.format(seed), 'r'))
            # test_feature = json.load(open(r'./all_sensor_test_features{}.json'.format(seed), 'r'))
            # train_feature = json.load(open(r'./tsf_train_features{}.json'.format(seed), 'r'))
            # test_feature = json.load(open(r'./tsf_test_features{}.json'.format(seed), 'r'))

            all_data_feature = dict()
            train_ovens = []
            test_ovens = []
            for k, v in train_feature.items():
                all_data_feature[k] = v
                assert len(v) == 882
                train_ovens.append(k)
            for k, v in test_feature.items():
                all_data_feature[k] = v
                test_ovens.append(k)
            tmp = []
            for k, v in all_data_feature.items():
                v = np.array(v)
                v[np.isnan(v)] = 0
                tmp.append(v)
            tmp = np.array(tmp)
            # 滤除低方差
            sel = VarianceThreshold(threshold=0)
            pca_feature = sel.fit_transform(tmp)

            # pca
            # pca_1 = PCA(n_components=pca)
            # pca_feature = pca_1.fit_transform(pca_feature)

            # tsne
            # tsne = TSNE(n_components=pca, method='exact')
            # pca_feature = tsne.fit_transform(pca_feature)

            # kpca
            kpca = KernelPCA(n_components=pca, kernel="rbf")
            pca_feature = kpca.fit_transform(pca_feature)

            train_pca_feature = pca_feature[:len(train_ovens)]
            test_pca_feature = pca_feature[len(train_ovens):]
            train_pca = dict()
            test_pca = dict()
            for ind, oven in enumerate(train_ovens):
                train_pca[oven] = train_pca_feature[ind].tolist()
            for ind, oven in enumerate(test_ovens):
                test_pca[oven] = test_pca_feature[ind].tolist()
            data = json.dumps(train_pca)
            with open(r'./no_thick_kpca_train_seed{}pca{}_pass.json'.format(seed, pca), 'w') as js_file:
                js_file.write(data)
            data = json.dumps(test_pca)
            with open(r'./no_thick_kpca_test_seed{}pca{}_pass.json'.format(seed, pca), 'w') as js_file:
                js_file.write(data)


if __name__ == '__main__':
    get_features([33])
    get_seed_pca_feature()





