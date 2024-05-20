import os
import pandas
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

data_path = "dataset/nd100_2018_2023_new_1.csv"
root_path = "/nasdaq100/"
gpu_id = "7"
stock_num = 87


def process_daily_df_std(df, feature_cols):
    df = df.copy()
    for c in feature_cols:
        df[c] = filter_extreme_3sigma(df[c])
        df[c] = standardize_zscore(df[c])
    return df


def filter_extreme_3sigma(series, n=3):
    mean = series.mean()
    std = series.std()
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)


def standardize_zscore(series):
    std = series.std()
    mean = series.mean()
    return (series - mean) / std


def create_dataset(df, feature_cols, label_col, date_range, hist_len=60, num_cores=1):
    df_group = df.groupby("kdcode")
    param_list = []
    for kdcode in df_group.groups.keys():
        df_comp = df_group.get_group(kdcode)
        param_list.append((df_comp, feature_cols, label_col, hist_len, date_range))
    print("# groups = ", len(param_list))
    result = []
    if num_cores > 1:
        pool = multiprocessing.Pool(num_cores)
        result = pool.starmap(generate_dataset, param_list)
        pool.close()
        pool.join()
    else:
        for params in param_list:
            result.append(generate_dataset(*params))
    return np.concatenate([x for x in result if len(x) > 0])


def generate_dataset(df_comp, feature_cols, label_col, hist_len, date_range):
    ds = []
    date_range = [pd.to_datetime(x) for x in date_range]
    id_vals = df_comp.index.values
    df_comp = df_comp.reset_index(drop=True)
    dt_vals = df_comp["dt"].values
    feature_vals = df_comp[feature_cols].values
    label_vals = df_comp[label_col].values
    for idx, row in df_comp.iterrows():
        dt = dt_vals[idx]
        if idx < hist_len or dt < date_range[0] or dt > date_range[1]:
            continue
        else:
            seq_features = feature_vals[idx + 1 - hist_len : idx + 1]
            ds.append((id_vals[idx], seq_features, label_vals[idx]))
    return ds


def rank_labeling(df, col_label="label", col_return="t2_am-15m_return_rate"):
    df[col_label] = df[col_return].rank(ascending=True, pct=True)
    return df


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def fun_similar(dts_all, df, dt_one):
    df1 = df.copy()
    dt_pred = dts_all[dts_all.index(dt_one) - 60]
    df1 = df1.loc[df1["dt"] >= str(dt_pred)]
    df1 = df1.loc[df1["dt"] <= str(dt_one)]
    df2 = df1[["kdcode", "dt", "shouyi"]]
    df2 = df2.reset_index(drop=True)
    df_grouped = df2.groupby("kdcode")
    df3 = pd.DataFrame()

    data = {}
    for kdcode, group in df_grouped:
        data[kdcode] = group["shouyi"].reset_index(drop=True)
    df3 = pd.DataFrame(data).fillna(0.0)
    df3_T = df3.T.values

    similarities = np.zeros((len(df3_T), len(df3_T)))
    for i in range(len(df3_T)):
        for j in range(len(df3_T)):
            similarities[i, j] = (cosine_similarity(df3_T[i], df3_T[j]) + 1) / 2
    similarities = pd.DataFrame(similarities)
    similarities[similarities < 0.5] = 0
    df5 = similarities.values.tolist()
    return df5


def process_row(
    i,
    stock_choose,
    df1_sw_kdcode_1_list,
    df1_sw_kdcode_2_list,
    dict_kdcode_sw_kdcode_2_1,
    dict_kdcode_sw_kdcode_st_2,
):
    one = []
    for j in range(len(stock_choose)):
        if stock_choose[i] in df1_sw_kdcode_1_list:
            if stock_choose[j] in df1_sw_kdcode_1_list:
                one.append(1)
            elif stock_choose[j] in df1_sw_kdcode_2_list:
                if stock_choose[i] == dict_kdcode_sw_kdcode_2_1[stock_choose[j]]:
                    one.append(1)
                else:
                    one.append(0)
            else:
                one.append(0)
        elif stock_choose[i] in df1_sw_kdcode_2_list:
            if stock_choose[j] in df1_sw_kdcode_1_list:
                if stock_choose[j] == dict_kdcode_sw_kdcode_2_1[stock_choose[i]]:
                    one.append(1)
                else:
                    one.append(0)
            elif stock_choose[j] in df1_sw_kdcode_2_list:
                if (
                    dict_kdcode_sw_kdcode_2_1[stock_choose[i]]
                    == dict_kdcode_sw_kdcode_2_1[stock_choose[j]]
                ):
                    one.append(1)
                else:
                    one.append(0)
            else:
                if stock_choose[i] == dict_kdcode_sw_kdcode_st_2[stock_choose[j]]:
                    one.append(1)
                else:
                    one.append(0)
        else:
            if stock_choose[j] in df1_sw_kdcode_1_list:
                one.append(0)
            elif stock_choose[j] in df1_sw_kdcode_2_list:
                if stock_choose[j] == dict_kdcode_sw_kdcode_st_2[stock_choose[i]]:
                    one.append(1)
                else:
                    one.append(0)
            else:
                if (
                    dict_kdcode_sw_kdcode_st_2[stock_choose[i]]
                    == dict_kdcode_sw_kdcode_st_2[stock_choose[j]]
                ):
                    one.append(1)
                else:
                    one.append(0)
        pass
    return one


class GCGRU(tf.keras.Model):
    def __init__(
        self,
        N,
        F,
        P,
        Units_GCN,
        Units_GRU,
        Units_FC,
        Fixed_Matrices,
        Matrix_Weights,
        Is_Dyn,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super(GCGRU, self).__init__()
        self.N = N
        self.F = F
        self.P = P
        self.mat = Fixed_Matrices

        coe = tf.Variable(1.0, trainable=True)
        self.mats = Matrix_Weights[0] * self.mat[0] * coe
        self.units_gcn = Units_GCN
        self.w_gcn = []
        self.b_gcn = []
        pre = self.F

        for i in range(len(self.units_gcn)):
            aft = self.units_gcn[i]
            w = self.add_weight(
                name="w_GCN",
                shape=(pre, aft),
                initializer=tf.keras.initializers.get(kernel_initializer),
                trainable=True,
            )
            self.w_gcn.append(w)
            b = self.add_weight(
                "b_GCN",
                shape=(aft,),
                initializer=tf.keras.initializers.get(bias_initializer),
                trainable=True,
            )
            self.b_gcn.append(b)
            pre = aft

        self.units_gcn_1 = Units_GCN
        self.w_gcn_1 = []
        self.b_gcn_1 = []
        pre_1 = self.F

        for i in range(len(self.units_gcn_1)):
            aft_1 = self.units_gcn_1[i]
            w_1 = self.add_weight(
                name="w_GCN_1",
                shape=(pre_1, aft_1),
                initializer=tf.keras.initializers.get(kernel_initializer),
                trainable=True,
            )
            self.w_gcn_1.append(w_1)
            b_1 = self.add_weight(
                "b_GCN_1",
                shape=(aft_1,),
                initializer=tf.keras.initializers.get(bias_initializer),
                trainable=True,
            )
            self.b_gcn_1.append(b_1)
            pre_1 = aft_1

        self.units_gru = Units_GRU
        self.w_gru = []
        self.b_gru = []
        C = self.units_gcn[-1]
        F = self.F

        for i in range(len(self.units_gru) - 1):
            H = self.units_gru[i]
            pre = F + C + C + H
            aft = H

            for j in range(3):
                w = self.add_weight(
                    name="w_GRU",
                    shape=(pre, aft),
                    initializer=tf.keras.initializers.get(kernel_initializer),
                    trainable=True,
                )
                self.w_gru.append(w)
                b = self.add_weight(
                    name="b_GRU",
                    shape=(aft,),
                    initializer=tf.keras.initializers.get(bias_initializer),
                    trainable=True,
                )
                self.b_gru.append(b)
            F = aft

        H = self.units_gru[-2]
        G = self.units_gru[-1]
        w = self.add_weight(
            name="w_GRU",
            shape=(H, G),
            initializer=tf.keras.initializers.get(kernel_initializer),
            trainable=True,
        )
        self.w_gru.append(w)
        b = self.add_weight(
            name="b_GRU",
            shape=(G,),
            initializer=tf.keras.initializers.get(bias_initializer),
            trainable=True,
        )
        self.b_gru.append(b)

        self.units_fc = Units_FC
        self.w_fc = []
        self.b_fc = []
        pre = G
        for i in range(len(self.units_fc)):
            aft = self.units_fc[i]
            w = self.add_weight(
                name="w_FC",
                shape=(pre, aft),
                initializer=tf.keras.initializers.get(kernel_initializer),
                trainable=True,
            )
            self.w_fc.append(w)
            b = self.add_weight(
                name="b_FC",
                shape=(aft,),
                initializer=tf.keras.initializers.get(bias_initializer),
                trainable=True,
            )
            self.b_fc.append(b)
            pre = aft

    def Multi_GCN(self, inputs):
        P = inputs.shape[1]
        x_gcn = []

        for t in range(P):
            xt_gcn = inputs[:, t, :, :]
            for i in range(len(self.units_gcn)):
                xt_gcn = self.mats @ xt_gcn @ self.w_gcn[i] + self.b_gcn[i]
                xt_gcn = tf.nn.tanh(xt_gcn)

            x_gcn.append(xt_gcn)
        x_gcn = tf.stack(x_gcn, axis=1)
        return x_gcn

    def Multi_GCN_1(self, inputs, inputs_matrx):
        P = inputs.shape[1]
        x_gcn = []

        for t in range(P):
            xt_gcn = inputs[:, t, :, :]
            inputs_matrx_all = inputs_matrx[:, t, :, :]
            for i in range(len(self.units_gcn_1)):
                xt_gcn = (
                    inputs_matrx_all[t] @ xt_gcn @ self.w_gcn_1[i] + self.b_gcn_1[i]
                )
                xt_gcn = tf.nn.tanh(xt_gcn)

            x_gcn.append(xt_gcn)
        x_gcn = tf.stack(x_gcn, axis=1)
        return x_gcn

    def GRU(self, x, x_gcn, x_gcn_1):
        h_gru = []
        for i in range(len(self.units_gru) - 1):
            H = self.units_gru[i]
            h = tf.zeros_like(x[:, 0, :, :], dtype=tf.float32) @ tf.zeros([self.F, H])
            h_gru.append(h)

        for t in range(self.P):
            xt_gcn = x_gcn[:, t, :, :]
            xt_gcn_1 = x_gcn_1[:, t, :, :]
            xt = x[:, t, :, :]

            for i in range(len(h_gru)):
                ht_1 = h_gru[i]
                x_tgh = tf.concat([xt, xt_gcn, xt_gcn_1, ht_1], axis=2)
                ut = tf.nn.sigmoid(
                    x_tgh @ self.w_gru[3 * i + 0] + self.b_gru[3 * i + 0]
                )
                rt = tf.nn.sigmoid(
                    x_tgh @ self.w_gru[3 * i + 1] + self.b_gru[3 * i + 1]
                )
                x_tghr = tf.concat(
                    [xt, xt_gcn, xt_gcn_1, tf.multiply(rt, ht_1)], axis=2
                )
                ct = tf.nn.tanh(x_tghr @ self.w_gru[3 * i + 2] + self.b_gru[3 * i + 2])
                ht = tf.multiply(ut, ht_1) + tf.multiply((1 - ut), ct)
                xt = ht
                h_gru[i] = ht
        x_gru = tf.nn.sigmoid(ht @ self.w_gru[-1] + self.b_gru[-1])
        return x_gru

    def FC(self, x_gru):
        x = x_gru
        for i in range(len(self.w_fc)):
            x = x @ self.w_fc[i] + self.b_fc[i]
            x = tf.nn.sigmoid(x)
        x_fc = tf.squeeze(x, axis=-1)
        return x_fc

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        inputs_train = inputs[0]
        inputs_matrx = inputs[1]
        x_gcn = self.Multi_GCN(inputs_train)
        x_gcn_1 = self.Multi_GCN_1(inputs_train, inputs_matrx)
        x_gru = self.GRU(inputs_train, x_gcn, x_gcn_1)
        x_fc = self.FC(x_gru)
        return x_fc


class StockrnnBasicModel(object):
    def __init__(
        self,
        model_dt="2022-12-31",
        CUDA_VISIBLE_DEVICES=gpu_id,
        root_data_path=root_path,
        T=20,
        train_his=10,
        epoches_list=[3, 4, 5, 6, 7],
        P=10,
        Is_Dyn=False,
        Units_GCN=[50, 40],
        Units_GRU=[30, 20],
        Units_FC=[10, 1],
        number_of_models=10,
        model_data_path=root_path,
        batch_size=16,
    ):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

        self.T = T
        self.train_his = train_his
        self.epoches_list = epoches_list
        self.P = P
        self.Is_Dyn = Is_Dyn
        self.Units_GCN = Units_GCN
        self.Units_GRU = Units_GRU
        self.Units_FC = Units_FC
        self.number_of_models = number_of_models
        self.model_data_path = model_data_path
        self.batch_size = batch_size
        self.col_return = "t{}_close_return_rate".format(self.T)
        self.col_label_train = "t{}_label".format(self.T)
        self.vwap_col = "close"
        self.model_name = "stockrnn_basic_model"
        self._model_file_format_str = "{}-{}.h5"
        self.root_data_path = root_data_path

        if not os.path.exists(self.root_data_path):
            os.mkdir(self.root_data_path)

        self.train_model_folder = os.path.join(self.root_data_path, "models")
        self.train_model_folder = self.train_model_folder.replace(
            "/models", "/sub_model_data/stockrnn-basic-model-n0/models"
        )

        if not os.path.exists(self.train_model_folder):
            os.makedirs(self.train_model_folder)

        for i in range(self.number_of_models[0], self.number_of_models[-1]):
            path = self.train_model_folder.replace("-model-n0", "-model-n" + str(i))
            if not os.path.exists(path):
                os.makedirs(path)

        self.model_dt = model_dt
        self.model_file_path = os.path.join(
            self.train_model_folder,
            self._model_file_format_str.format(self.model_name, self.model_dt),
        )

        self.predict_folder_path = os.path.join(self.root_data_path, "prediction")
        if not os.path.exists(self.predict_folder_path):
            os.makedirs(self.predict_folder_path)
        self.predict_folder_path_last = self.predict_folder_path + "/"
        self.predict_folder_path = self.predict_folder_path.replace(
            "prediction", "sub_model_data/stockrnn-basic-model-n0/prediction"
        )

        for i in range(self.number_of_models[0], self.number_of_models[-1]):
            path = self.predict_folder_path.replace("-model-n0", "-model-n" + str(i))
            if not os.path.exists(path):
                os.makedirs(path)

        self.predict_folder_save = self.predict_folder_path_last.replace(
            "prediction", "prediction_all"
        )
        for i in self.epoches_list:
            path = self.predict_folder_save + str(i) + "/prediction"
            if not os.path.exists(path):
                os.makedirs(path)

        self.feature_cols = ["close", "open", "high", "low", "volume"]

        if CUDA_VISIBLE_DEVICES != "-1":
            gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
            print(gpus)
            tf.config.experimental.set_visible_devices(
                devices=gpus[0], device_type="GPU"
            )
            tf.config.experimental.set_memory_growth(gpus[0], True)

    def process_features(self, df_features, feature_cols):
        df_features_grouped = df_features.groupby("dt")
        res = []
        for dt in df_features_grouped.groups:
            df = df_features_grouped.get_group(dt)
            processed_df = process_daily_df_std(df, feature_cols)
            res.append(processed_df)
        df_features = pd.concat(res)
        df_features = df_features.dropna(subset=feature_cols)
        return df_features

    def construct_pred_data(
        self,
    ):
        import datetime

        pred_date_range = (
            datetime.datetime.strptime("2023-01-03", "%Y-%m-%d").date(),
            datetime.datetime.strptime("2023-12-29", "%Y-%m-%d").date(),
        )

        df_org = pd.read_csv(data_path)
        kdcodes = df_org["kdcode"].values.tolist()
        result = Counter(kdcodes)
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        kdcodes_last = []
        result = result[0:stock_num]
        for one in result:
            kdcodes_last.append(one[0])
        df_origin_features = df_org[df_org["kdcode"].isin(kdcodes_last)]

        df_origin_features.drop(columns=["prev_close", "adjfactor"], inplace=True)
        df = df_origin_features
        grouped = df.groupby(["dt", "sw_kdcode_2"], as_index=False).agg(
            {
                "kdcode": "first",
                "sw_kdcode_1": "first",
                "close": "mean",
                "open": "mean",
                "high": "mean",
                "low": "mean",
                "volume": "mean",
            }
        )
        grouped["kdcode"] = grouped["sw_kdcode_2"]
        merged_df = pd.concat([df, grouped], ignore_index=True)
        df2 = df_origin_features

        grouped2 = df2.groupby(["dt", "sw_kdcode_1"], as_index=False).agg(
            {
                "kdcode": "first",
                "sw_kdcode_2": "first",
                "close": "mean",
                "open": "mean",
                "high": "mean",
                "low": "mean",
                "volume": "mean",
            }
        )
        grouped2["kdcode"] = grouped2["sw_kdcode_1"]
        merged_df2 = pd.concat([merged_df, grouped2], ignore_index=True)
        df_origin_features = merged_df2
        df_origin_features = df_origin_features.loc[
            df_origin_features["dt"] >= "2022-01-01"
        ]
        df_origin_features = df_origin_features.reset_index(drop=True)

        def custom_sort(kdcode):
            rank1 = 0 if kdcode in kdcodes_last else 1
            rank2 = (
                0
                if kdcode
                == df_origin_features.loc[
                    df_origin_features["kdcode"] == kdcode, "sw_kdcode_2"
                ].values[0]
                else 1
            )
            rank3 = (
                0
                if kdcode
                == df_origin_features.loc[
                    df_origin_features["kdcode"] == kdcode, "sw_kdcode_1"
                ].values[0]
                else 1
            )
            return (rank1, rank2, rank3, kdcode)

        unique_kdcodes = list(set(df_origin_features["kdcode"]))
        sorted_kdcodes = sorted(unique_kdcodes, key=custom_sort)
        stock_choose = sorted_kdcodes
        df_pred_features = df_origin_features
        df_pred_features = self.process_features(df_pred_features, self.feature_cols)
        df_pred_features["dt"] = pd.to_datetime(df_pred_features["dt"])
        ds_data = create_dataset(
            df_pred_features,
            self.feature_cols,
            "kdcode",
            pred_date_range,
            hist_len=self.train_his,
            num_cores=1,
        )
        idx_data = np.array([x[0] for x in ds_data])
        X_data = np.array([x[1] for x in ds_data])
        s_idx = pd.Series(index=idx_data, data=list(range(len(idx_data))))
        idx_pred = s_idx[[i for i in df_pred_features.index if i in s_idx.index]].values
        X_pred = X_data[idx_pred]
        origin_idx_pred = idx_data[idx_pred]
        day_len_pred = int(len(X_pred) / len(stock_choose))
        X_pred_1 = X_pred.reshape(
            day_len_pred, len(stock_choose), self.train_his, len(self.feature_cols)
        )
        print(X_pred_1.shape)
        xx_last = []
        for i in range(day_len_pred):
            x1 = []
            for j in range(self.train_his):
                x2 = []
                for k in range(len(stock_choose)):
                    x2.append(X_pred_1[i][k][j])
                x1.append(x2)
            xx_last.append(x1)
        X_pred = np.array(xx_last)
        print(X_pred.shape)

        dict_index_stock = {}
        for i in range(len(stock_choose)):
            dict_index_stock[i] = stock_choose[i]

        df1_kdcode_list = df_pred_features["kdcode"].values.tolist()
        df1_sw_kdcode_1_list = df_pred_features["sw_kdcode_1"].values.tolist()
        df1_sw_kdcode_2_list = df_pred_features["sw_kdcode_2"].values.tolist()

        dict_kdcode_sw_kdcode_2_1 = {}
        for i in range(len(df1_sw_kdcode_2_list)):
            if df1_sw_kdcode_2_list[i] not in dict_kdcode_sw_kdcode_2_1:
                dict_kdcode_sw_kdcode_2_1[df1_sw_kdcode_2_list[i]] = (
                    df1_sw_kdcode_1_list[i]
                )

        dict_kdcode_sw_kdcode_st_2 = {}
        for i in range(len(df1_kdcode_list)):
            if df1_kdcode_list[i] not in dict_kdcode_sw_kdcode_st_2:
                dict_kdcode_sw_kdcode_st_2[df1_kdcode_list[i]] = df1_sw_kdcode_2_list[i]

        from multiprocessing import Pool
        from functools import partial

        func_partial = partial(
            process_row,
            stock_choose=stock_choose,
            df1_sw_kdcode_1_list=df1_sw_kdcode_1_list,
            df1_sw_kdcode_2_list=df1_sw_kdcode_2_list,
            dict_kdcode_sw_kdcode_2_1=dict_kdcode_sw_kdcode_2_1,
            dict_kdcode_sw_kdcode_st_2=dict_kdcode_sw_kdcode_st_2,
        )
        pool = mp.Pool(mp.cpu_count())
        result = list(
            tqdm(
                pool.imap(func_partial, range(len(stock_choose))),
                total=len(stock_choose),
            )
        )
        pool.close()
        pool.join()
        matrx = np.array(result)

        file_list = os.listdir(self.root_data_path)
        if "matrx_" + str(self.train_his) + "_test.npy" in file_list:
            matrx_1 = np.load(
                self.root_data_path + "matrx_" + str(self.train_his) + "_test.npy"
            )
        else:
            df_org = pd.read_csv(data_path)
            kdcodes = df_org["kdcode"].values.tolist()
            result = Counter(kdcodes)
            result = sorted(result.items(), key=lambda x: x[1], reverse=True)
            kdcodes_last = []
            result = result[0:stock_num]
            for one in result:
                kdcodes_last.append(one[0])

            df_origin_features = df_org[df_org["kdcode"].isin(kdcodes_last)]
            df_origin_features.drop(columns=["adjfactor"], inplace=True)

            df = df_origin_features
            grouped = df.groupby(["dt", "sw_kdcode_2"], as_index=False).agg(
                {
                    "kdcode": "first",
                    "sw_kdcode_1": "first",
                    "close": "mean",
                    "open": "mean",
                    "high": "mean",
                    "low": "mean",
                    "prev_close": "mean",
                    "volume": "mean",
                }
            )
            grouped["kdcode"] = grouped["sw_kdcode_2"]
            merged_df = pd.concat([df, grouped], ignore_index=True)
            df2 = df_origin_features

            grouped2 = df2.groupby(["dt", "sw_kdcode_1"], as_index=False).agg(
                {
                    "kdcode": "first",
                    "sw_kdcode_2": "first",
                    "close": "mean",
                    "open": "mean",
                    "high": "mean",
                    "low": "mean",
                    "prev_close": "mean",
                    "volume": "mean",
                }
            )
            grouped2["kdcode"] = grouped2["sw_kdcode_1"]
            merged_df2 = pd.concat([merged_df, grouped2], ignore_index=True)
            df_origin_features = merged_df2
            df_origin_features = df_origin_features.loc[
                df_origin_features["dt"] <= "2023-12-31"
            ]
            df_origin_features = df_origin_features.reset_index(drop=True)

            def custom_sort(kdcode):
                rank1 = 0 if kdcode in kdcodes_last else 1
                rank2 = (
                    0
                    if kdcode
                    == df_origin_features.loc[
                        df_origin_features["kdcode"] == kdcode, "sw_kdcode_2"
                    ].values[0]
                    else 1
                )
                rank3 = (
                    0
                    if kdcode
                    == df_origin_features.loc[
                        df_origin_features["kdcode"] == kdcode, "sw_kdcode_1"
                    ].values[0]
                    else 1
                )
                return (rank1, rank2, rank3, kdcode)

            unique_kdcodes = list(set(df_origin_features["kdcode"]))
            sorted_kdcodes = sorted(unique_kdcodes, key=custom_sort)
            stock_choose = sorted_kdcodes

            df_features = self.process_features(
                df_origin_features,
                ["close", "open", "high", "low", "volume", "prev_close"],
            )
            dts_all = sorted(list(set(df_origin_features["dt"].values.tolist())))
            dts_choose_1 = ["2022-11-01", "2022-11-02", "2022-11-03", "2022-11-04", "2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14", "2022-11-15", "2022-11-16", "2022-11-17",  "2022-11-18", "2022-11-21", "2022-11-22",
                            "2022-11-23", "2022-11-25", "2022-11-28", "2022-11-29", "2022-11-30", "2022-12-01", "2022-12-02", "2022-12-05", "2022-12-06", "2022-12-07", "2022-12-08", "2022-12-09", "2022-12-12", "2022-12-13", "2022-12-14", "2022-12-15", 
                            "2022-12-16", "2022-12-19", "2022-12-20", "2022-12-21", "2022-12-22", "2022-12-23", "2022-12-27", "2022-12-28", "2022-12-29", "2022-12-30"]
            dts_choose_2 = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-23", "2023-01-24", "2023-01-25", 
                            "2023-01-26", "2023-01-27", "2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02", "2023-02-03", "2023-02-06", "2023-02-07", "2023-02-08", "2023-02-09", "2023-02-10", "2023-02-13", "2023-02-14", "2023-02-15", "2023-02-16", 
                            "2023-02-17", "2023-02-21", "2023-02-22", "2023-02-23", "2023-02-24", "2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02", "2023-03-03", "2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09", "2023-03-10", "2023-03-13", 
                            "2023-03-14", "2023-03-15", "2023-03-16", "2023-03-17", "2023-03-20", "2023-03-21", "2023-03-22", "2023-03-23", "2023-03-24", "2023-03-27", "2023-03-28", "2023-03-29", "2023-03-30", "2023-03-31", "2023-04-03", "2023-04-04", 
                            "2023-04-05", "2023-04-06", "2023-04-10", "2023-04-11", "2023-04-12", "2023-04-13", "2023-04-14", "2023-04-17", "2023-04-18", "2023-04-19", "2023-04-20", "2023-04-21", "2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27", 
                            "2023-04-28", "2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-08", "2023-05-09", "2023-05-10", "2023-05-11", "2023-05-12", "2023-05-15", "2023-05-16", "2023-05-17", "2023-05-18", "2023-05-19", 
                            "2023-05-22", "2023-05-23", "2023-05-24", "2023-05-25", "2023-05-26", "2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02", "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08", "2023-06-09", "2023-06-12", "2023-06-13", 
                            "2023-06-14", "2023-06-15", "2023-06-16", "2023-06-20", "2023-06-21", "2023-06-22", "2023-06-23", "2023-06-26", "2023-06-27", "2023-06-28", "2023-06-29", "2023-06-30", "2023-07-03", "2023-07-05", "2023-07-06", "2023-07-07", 
                            "2023-07-10", "2023-07-11", "2023-07-12", "2023-07-13", "2023-07-14", "2023-07-17", "2023-07-18", "2023-07-19", "2023-07-20", "2023-07-21", "2023-07-24", "2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-31", 
                            "2023-08-01", "2023-08-02", "2023-08-03", "2023-08-04", "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10", "2023-08-11", "2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17", "2023-08-18", "2023-08-21", "2023-08-22", 
                            "2023-08-23", "2023-08-24", "2023-08-25", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-05", "2023-09-06", "2023-09-07", "2023-09-08", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", 
                            "2023-09-15", "2023-09-18", "2023-09-19", "2023-09-20", "2023-09-21", "2023-09-22", "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", 
                            "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20", "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-30", 
                            "2023-10-31", "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-06", "2023-11-07", "2023-11-08", "2023-11-09", "2023-11-10", "2023-11-13", "2023-11-14", "2023-11-15", "2023-11-16", "2023-11-17", "2023-11-20", "2023-11-21", 
                            "2023-11-22", "2023-11-24", "2023-11-27", "2023-11-28", "2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07", "2023-12-08", "2023-12-11", "2023-12-12", "2023-12-13", "2023-12-14", 
                            "2023-12-15", "2023-12-18", "2023-12-19", "2023-12-20", "2023-12-21", "2023-12-22", "2023-12-26", "2023-12-27", "2023-12-28"]
            dts_choose_3 = dts_choose_1[-self.train_his + 1 :] + dts_choose_2
            df = df_features
            df["shouyi"] = (df["open"] - df["prev_close"]) / df["prev_close"]

            param_list = []
            for dt_one in tqdm(dts_choose_3):
                param_list.append((dts_all, df, dt_one))

            print(param_list)

            pool = multiprocessing.Pool(40)
            results = []
            for i in range(len(dts_choose_3)):
                results.append(pool.apply_async(fun_similar, param_list[i]))
            pool.close()
            pool.join()

            matrx_1 = []
            for res in results:
                matrx_1.append(res.get())
            matrx_2 = []
            for one in range(len(dts_choose_2)):
                matrx_2.append(matrx_1[one : one + self.train_his])
            matrx_1 = np.array(matrx_2)
            np.save(
                self.root_data_path + "matrx_" + str(self.train_his) + "_test.npy",
                matrx_1,
            )

        return matrx, matrx_1, stock_choose, X_pred, origin_idx_pred

    def train(self):
        import datetime

        train_date_range = (
            datetime.datetime.strptime("2018-07-02", "%Y-%m-%d").date(),
            datetime.datetime.strptime("2022-12-27", "%Y-%m-%d").date(),
        )

        df_org = pd.read_csv(data_path)
        kdcodes = df_org["kdcode"].values.tolist()
        result = Counter(kdcodes)
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        kdcodes_last = []
        print(result)
        result = result[0:stock_num]
        for one in result:
            kdcodes_last.append(one[0])
        df_origin_features = df_org[df_org["kdcode"].isin(kdcodes_last)]
        df_origin_features.drop(columns=["prev_close", "adjfactor"], inplace=True)
        df = df_origin_features
        grouped = df.groupby(["dt", "sw_kdcode_2"], as_index=False).agg(
            {
                "kdcode": "first",
                "sw_kdcode_1": "first",
                "close": "mean",
                "open": "mean",
                "high": "mean",
                "low": "mean",
                "volume": "mean",
            }
        )
        grouped["kdcode"] = grouped["sw_kdcode_2"]
        merged_df = pd.concat([df, grouped], ignore_index=True)
        df2 = df_origin_features

        grouped2 = df2.groupby(["dt", "sw_kdcode_1"], as_index=False).agg(
            {
                "kdcode": "first",
                "sw_kdcode_2": "first",
                "close": "mean",
                "open": "mean",
                "high": "mean",
                "low": "mean",
                "volume": "mean",
            }
        )
        grouped2["kdcode"] = grouped2["sw_kdcode_1"]
        merged_df2 = pd.concat([merged_df, grouped2], ignore_index=True)
        df_origin_features = merged_df2

        c = self.vwap_col
        n = self.T
        df_origin_features["t1_{}".format(c)] = df_origin_features.groupby("kdcode")[
            c
        ].shift(-1)
        df_origin_features["t{}_{}".format(n, c)] = df_origin_features.groupby(
            "kdcode"
        )[c].shift(-n)
        df_origin_features["t{}_{}_return_rate".format(n, c)] = (
            df_origin_features["t{}_{}".format(n, c)]
        ) / (df_origin_features["t1_{}".format(c)]) - 1
        df_labeled_features = df_origin_features
        df_labeled_features = df_labeled_features.loc[
            df_labeled_features["dt"] <= "2022-12-27"
        ]
        df_labeled_features = df_labeled_features.reset_index(drop=True)

        def custom_sort(kdcode):
            rank1 = 0 if kdcode in kdcodes_last else 1
            rank2 = (
                0
                if kdcode
                == df_labeled_features.loc[
                    df_labeled_features["kdcode"] == kdcode, "sw_kdcode_2"
                ].values[0]
                else 1
            )
            rank3 = (
                0
                if kdcode
                == df_labeled_features.loc[
                    df_labeled_features["kdcode"] == kdcode, "sw_kdcode_1"
                ].values[0]
                else 1
            )
            return (rank1, rank2, rank3, kdcode)

        unique_kdcodes = list(set(df_labeled_features["kdcode"]))
        sorted_kdcodes = sorted(unique_kdcodes, key=custom_sort)
        stock_choose = sorted_kdcodes
        df_labeled_features = self.process_features(
            df_labeled_features, self.feature_cols
        )
        df_labeled_features["dt"] = pd.to_datetime(df_labeled_features["dt"])
        df_labeled_features = df_labeled_features.groupby("dt").apply(
            lambda df: rank_labeling(
                df, col_label=self.col_label_train, col_return=self.col_return
            )
        )
        ds_data = create_dataset(
            df_labeled_features,
            self.feature_cols,
            self.col_label_train,
            train_date_range,
            hist_len=self.train_his,
            num_cores=1,
        )
        idx_data = np.array([x[0] for x in ds_data])
        X_data = np.array([x[1] for x in ds_data])
        Y_data = np.array([x[2] for x in ds_data])
        s_idx = pd.Series(index=idx_data, data=list(range(len(idx_data))))
        idx_train = s_idx[
            [i for i in df_labeled_features.index if i in s_idx.index]
        ].values
        X_train = X_data[idx_train]
        Y_train = Y_data[idx_train]
        origin_idx_train = idx_data[idx_train]
        day_len_train = int(len(X_train) / len(stock_choose))
        X_train_1 = X_train.reshape(
            day_len_train, len(stock_choose), self.train_his, len(self.feature_cols)
        )
        xx_last = []
        for i in range(day_len_train):
            x1 = []
            for j in range(self.train_his):
                x2 = []
                for k in range(len(stock_choose)):
                    x2.append(X_train_1[i][k][j])
                x1.append(x2)
            xx_last.append(x1)
        X_train = np.array(xx_last)
        Y_train = Y_train.reshape(day_len_train, len(stock_choose), 1)

        dict_index_stock = {}
        for i in range(len(stock_choose)):
            dict_index_stock[i] = stock_choose[i]

        df1_kdcode_list = df_labeled_features["kdcode"].values.tolist()
        df1_sw_kdcode_1_list = df_labeled_features["sw_kdcode_1"].values.tolist()
        df1_sw_kdcode_2_list = df_labeled_features["sw_kdcode_2"].values.tolist()

        dict_kdcode_sw_kdcode_2_1 = {}
        for i in range(len(df1_sw_kdcode_2_list)):
            if df1_sw_kdcode_2_list[i] not in dict_kdcode_sw_kdcode_2_1:
                dict_kdcode_sw_kdcode_2_1[df1_sw_kdcode_2_list[i]] = (
                    df1_sw_kdcode_1_list[i]
                )

        dict_kdcode_sw_kdcode_st_2 = {}
        for i in range(len(df1_kdcode_list)):
            if df1_kdcode_list[i] not in dict_kdcode_sw_kdcode_st_2:
                dict_kdcode_sw_kdcode_st_2[df1_kdcode_list[i]] = df1_sw_kdcode_2_list[i]

        from multiprocessing import Pool
        from functools import partial

        func_partial = partial(
            process_row,
            stock_choose=stock_choose,
            df1_sw_kdcode_1_list=df1_sw_kdcode_1_list,
            df1_sw_kdcode_2_list=df1_sw_kdcode_2_list,
            dict_kdcode_sw_kdcode_2_1=dict_kdcode_sw_kdcode_2_1,
            dict_kdcode_sw_kdcode_st_2=dict_kdcode_sw_kdcode_st_2,
        )
        pool = mp.Pool(mp.cpu_count())
        result = list(
            tqdm(
                pool.imap(func_partial, range(len(stock_choose))),
                total=len(stock_choose),
            )
        )
        pool.close()
        pool.join()
        matrx = np.array(result)
        print(matrx.shape)

        file_list = os.listdir(self.root_data_path)
        if "matrx_" + str(self.train_his) + "_train.npy" in file_list:
            matrx_1 = np.load(
                self.root_data_path + "matrx_" + str(self.train_his) + "_train.npy"
            )
            print(matrx_1.shape)
        else:
            df_org = pd.read_csv(data_path)
            kdcodes = df_org["kdcode"].values.tolist()
            result = Counter(kdcodes)
            result = sorted(result.items(), key=lambda x: x[1], reverse=True)
            kdcodes_last = []
            result = result[0:stock_num]
            for one in result:
                kdcodes_last.append(one[0])

            df_origin_features = df_org[df_org["kdcode"].isin(kdcodes_last)]
            df_origin_features.drop(columns=["adjfactor"], inplace=True)
            df = df_origin_features
            grouped = df.groupby(["dt", "sw_kdcode_2"], as_index=False).agg(
                {
                    "kdcode": "first",
                    "sw_kdcode_1": "first",
                    "close": "mean",
                    "open": "mean",
                    "high": "mean",
                    "low": "mean",
                    "prev_close": "mean",
                    "volume": "mean",
                }
            )
            grouped["kdcode"] = grouped["sw_kdcode_2"]
            merged_df = pd.concat([df, grouped], ignore_index=True)
            df2 = df_origin_features

            grouped2 = df2.groupby(["dt", "sw_kdcode_1"], as_index=False).agg(
                {
                    "kdcode": "first",
                    "sw_kdcode_2": "first",
                    "close": "mean",
                    "open": "mean",
                    "high": "mean",
                    "low": "mean",
                    "prev_close": "mean",
                    "volume": "mean",
                }
            )
            grouped2["kdcode"] = grouped2["sw_kdcode_1"]
            merged_df2 = pd.concat([merged_df, grouped2], ignore_index=True)
            df_origin_features = merged_df2

            df_origin_features = df_origin_features.loc[
                df_origin_features["dt"] <= "2022-12-27"
            ]
            df_origin_features = df_origin_features.reset_index(drop=True)

            def custom_sort(kdcode):
                rank1 = 0 if kdcode in kdcodes_last else 1
                rank2 = (
                    0
                    if kdcode
                    == df_origin_features.loc[
                        df_origin_features["kdcode"] == kdcode, "sw_kdcode_2"
                    ].values[0]
                    else 1
                )
                rank3 = (
                    0
                    if kdcode
                    == df_origin_features.loc[
                        df_origin_features["kdcode"] == kdcode, "sw_kdcode_1"
                    ].values[0]
                    else 1
                )
                return (rank1, rank2, rank3, kdcode)

            unique_kdcodes = list(set(df_origin_features["kdcode"]))
            sorted_kdcodes = sorted(unique_kdcodes, key=custom_sort)
            stock_choose = sorted_kdcodes

            df_features = self.process_features(
                df_origin_features,
                ["close", "open", "high", "low", "volume", "prev_close"],
            )
            dts_all = sorted(list(set(df_origin_features["dt"].values.tolist())))
            dts_choose_1 = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08", "2018-01-09", "2018-01-10", "2018-01-11", "2018-01-12", "2018-01-16", "2018-01-17", "2018-01-18", "2018-01-19", "2018-01-22", "2018-01-23", 
                            "2018-01-24", "2018-01-25", "2018-01-26", "2018-01-29", "2018-01-30", "2018-01-31", "2018-02-01", "2018-02-02", "2018-02-05", "2018-02-06", "2018-02-07", "2018-02-08", "2018-02-09", "2018-02-12", "2018-02-13", 
                            "2018-02-14", "2018-02-15", "2018-02-16", "2018-02-20", "2018-02-21", "2018-02-22", "2018-02-23", "2018-02-26", "2018-02-27", "2018-02-28", "2018-03-01", "2018-03-02", "2018-03-05", "2018-03-06", "2018-03-07", 
                            "2018-03-08", "2018-03-09", "2018-03-12", "2018-03-13", "2018-03-14", "2018-03-15", "2018-03-16", "2018-03-19", "2018-03-20", "2018-03-21", "2018-03-22", "2018-03-23", "2018-03-26", "2018-03-27", "2018-03-28", 
                            "2018-03-29", "2018-04-02", "2018-04-03", "2018-04-04", "2018-04-05", "2018-04-06", "2018-04-09", "2018-04-10", "2018-04-11", "2018-04-12", "2018-04-13", "2018-04-16", "2018-04-17", "2018-04-18", "2018-04-19", 
                            "2018-04-20", "2018-04-23", "2018-04-24", "2018-04-25", "2018-04-26", "2018-04-27", "2018-04-30", "2018-05-01", "2018-05-02", "2018-05-03", "2018-05-04", "2018-05-07", "2018-05-08", "2018-05-09", "2018-05-10", 
                            "2018-05-11", "2018-05-14", "2018-05-15", "2018-05-16", "2018-05-17", "2018-05-18", "2018-05-21", "2018-05-22", "2018-05-23", "2018-05-24", "2018-05-25", "2018-05-29", "2018-05-30", "2018-05-31", "2018-06-01", 
                            "2018-06-04", "2018-06-05", "2018-06-06", "2018-06-07", "2018-06-08", "2018-06-11", "2018-06-12", "2018-06-13", "2018-06-14", "2018-06-15", "2018-06-18", "2018-06-19", "2018-06-20", "2018-06-21", "2018-06-22", 
                            "2018-06-25", "2018-06-26", "2018-06-27", "2018-06-28", "2018-06-29"]
            dts_choose_2 = ["2018-07-02", "2018-07-03", "2018-07-05", "2018-07-06", "2018-07-09", "2018-07-10", "2018-07-11", "2018-07-12", "2018-07-13", "2018-07-16", "2018-07-17", "2018-07-18", "2018-07-19", "2018-07-20", "2018-07-23", 
                            "2018-07-24", "2018-07-25", "2018-07-26", "2018-07-27", "2018-07-30", "2018-07-31", "2018-08-01", "2018-08-02", "2018-08-03", "2018-08-06", "2018-08-07", "2018-08-08", "2018-08-09", "2018-08-10", "2018-08-13", 
                            "2018-08-14", "2018-08-15", "2018-08-16", "2018-08-17", "2018-08-20", "2018-08-21", "2018-08-22", "2018-08-23", "2018-08-24", "2018-08-27", "2018-08-28", "2018-08-29", "2018-08-30", "2018-08-31", "2018-09-04", 
                            "2018-09-05", "2018-09-06", "2018-09-07", "2018-09-10", "2018-09-11", "2018-09-12", "2018-09-13", "2018-09-14", "2018-09-17", "2018-09-18", "2018-09-19", "2018-09-20", "2018-09-21", "2018-09-24", "2018-09-25", 
                            "2018-09-26", "2018-09-27", "2018-09-28", "2018-10-01", "2018-10-02", "2018-10-03", "2018-10-04", "2018-10-05", "2018-10-08", "2018-10-09", "2018-10-10", "2018-10-11", "2018-10-12", "2018-10-15", "2018-10-16", 
                            "2018-10-17", "2018-10-18", "2018-10-19", "2018-10-22", "2018-10-23", "2018-10-24", "2018-10-25", "2018-10-26", "2018-10-29", "2018-10-30", "2018-10-31", "2018-11-01", "2018-11-02", "2018-11-05", "2018-11-06", 
                            "2018-11-07", "2018-11-08", "2018-11-09", "2018-11-12", "2018-11-13", "2018-11-14", "2018-11-15", "2018-11-16", "2018-11-19", "2018-11-20", "2018-11-21", "2018-11-23", "2018-11-26", "2018-11-27", "2018-11-28", 
                            "2018-11-29", "2018-11-30", "2018-12-03", "2018-12-04", "2018-12-06", "2018-12-07", "2018-12-10", "2018-12-11", "2018-12-12", "2018-12-13", "2018-12-14", "2018-12-17", "2018-12-18", "2018-12-19", "2018-12-20", 
                            "2018-12-21", "2018-12-24", "2018-12-26", "2018-12-27", "2018-12-28", "2018-12-31", "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07", "2019-01-08", "2019-01-09", "2019-01-10", "2019-01-11", "2019-01-14", 
                            "2019-01-15", "2019-01-16", "2019-01-17", "2019-01-18", "2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-04", "2019-02-05", 
                            "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-25", "2019-02-26", "2019-02-27", 
                            "2019-02-28", "2019-03-01", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-18", "2019-03-19", "2019-03-20", 
                            "2019-03-21", "2019-03-22", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-08", "2019-04-09", "2019-04-10", 
                            "2019-04-11", "2019-04-12", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", 
                            "2019-05-03", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10", "2019-05-13", "2019-05-14", "2019-05-15", "2019-05-16", "2019-05-17", "2019-05-20", "2019-05-21", "2019-05-22", "2019-05-23", 
                            "2019-05-24", "2019-05-28", "2019-05-29", "2019-05-30", "2019-05-31", "2019-06-03", "2019-06-04", "2019-06-05", "2019-06-06", "2019-06-07", "2019-06-10", "2019-06-11", "2019-06-12", "2019-06-13", "2019-06-14", 
                            "2019-06-17", "2019-06-18", "2019-06-19", "2019-06-20", "2019-06-21", "2019-06-24", "2019-06-25", "2019-06-26", "2019-06-27", "2019-06-28", "2019-07-01", "2019-07-02", "2019-07-03", "2019-07-05", "2019-07-08", 
                            "2019-07-09", "2019-07-10", "2019-07-11", "2019-07-12", "2019-07-15", "2019-07-16", "2019-07-17", "2019-07-18", "2019-07-19", "2019-07-22", "2019-07-23", "2019-07-24", "2019-07-25", "2019-07-26", "2019-07-29", 
                            "2019-07-30", "2019-07-31", "2019-08-01", "2019-08-02", "2019-08-05", "2019-08-06", "2019-08-07", "2019-08-08", "2019-08-09", "2019-08-12", "2019-08-13", "2019-08-14", "2019-08-15", "2019-08-16", "2019-08-19", 
                            "2019-08-20", "2019-08-21", "2019-08-22", "2019-08-23", "2019-08-26", "2019-08-27", "2019-08-28", "2019-08-29", "2019-08-30", "2019-09-03", "2019-09-04", "2019-09-05", "2019-09-06", "2019-09-09", "2019-09-10", 
                            "2019-09-11", "2019-09-12", "2019-09-13", "2019-09-16", "2019-09-17", "2019-09-18", "2019-09-19", "2019-09-20", "2019-09-23", "2019-09-24", "2019-09-25", "2019-09-26", "2019-09-27", "2019-09-30", "2019-10-01", 
                            "2019-10-02", "2019-10-03", "2019-10-04", "2019-10-07", "2019-10-08", "2019-10-09", "2019-10-10", "2019-10-11", "2019-10-14", "2019-10-15", "2019-10-16", "2019-10-17", "2019-10-18", "2019-10-21", "2019-10-22", 
                            "2019-10-23", "2019-10-24", "2019-10-25", "2019-10-28", "2019-10-29", "2019-10-30", "2019-10-31", "2019-11-01", "2019-11-04", "2019-11-05", "2019-11-06", "2019-11-07", "2019-11-08", "2019-11-11", "2019-11-12", 
                            "2019-11-13", "2019-11-14", "2019-11-15", "2019-11-18", "2019-11-19", "2019-11-20", "2019-11-21", "2019-11-22", "2019-11-25", "2019-11-26", "2019-11-27", "2019-11-29", "2019-12-02", "2019-12-03", "2019-12-04", 
                            "2019-12-05", "2019-12-06", "2019-12-09", "2019-12-10", "2019-12-11", "2019-12-12", "2019-12-13", "2019-12-16", "2019-12-17", "2019-12-18", "2019-12-19", "2019-12-20", "2019-12-23", "2019-12-24", "2019-12-26", 
                            "2019-12-27", "2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17", 
                            "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-10", 
                            "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", "2020-03-02", "2020-03-03", 
                            "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-16", "2020-03-17", "2020-03-18", "2020-03-19", "2020-03-20", "2020-03-23", "2020-03-24", 
                            "2020-03-25", "2020-03-26", "2020-03-27", "2020-03-30", "2020-03-31", "2020-04-01", "2020-04-02", "2020-04-03", "2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-13", "2020-04-14", "2020-04-15", 
                            "2020-04-16", "2020-04-17", "2020-04-20", "2020-04-21", "2020-04-22", "2020-04-23", "2020-04-24", "2020-04-27", "2020-04-28", "2020-04-29", "2020-04-30", "2020-05-01", "2020-05-04", "2020-05-05", "2020-05-06", 
                            "2020-05-07", "2020-05-08", "2020-05-11", "2020-05-12", "2020-05-13", "2020-05-14", "2020-05-15", "2020-05-18", "2020-05-19", "2020-05-20", "2020-05-21", "2020-05-22", "2020-05-26", "2020-05-27", "2020-05-28", 
                            "2020-05-29", "2020-06-01", "2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-08", "2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", 
                            "2020-06-19", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10", 
                            "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", 
                            "2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-10", "2020-08-11", "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-17", "2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21", 
                            "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-14", 
                            "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-05", 
                            "2020-10-06", "2020-10-07", "2020-10-08", "2020-10-09", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16", "2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22", "2020-10-23", "2020-10-26", 
                            "2020-10-27", "2020-10-28", "2020-10-29", "2020-10-30", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05", "2020-11-06", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-16", 
                            "2020-11-17", "2020-11-18", "2020-11-19", "2020-11-20", "2020-11-23", "2020-11-24", "2020-11-25", "2020-11-27", "2020-11-30", "2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-07", "2020-12-08", 
                            "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-14", "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-28", "2020-12-29", "2020-12-30", 
                            "2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15", "2021-01-19", "2021-01-20", "2021-01-21", "2021-01-22", 
                            "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05", "2021-02-08", "2021-02-09", "2021-02-10", "2021-02-11", "2021-02-12", 
                            "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25", "2021-02-26", "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05", "2021-03-08", 
                            "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12", "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-22", "2021-03-23", "2021-03-24", "2021-03-25", "2021-03-26", "2021-03-29", 
                            "2021-03-30", "2021-03-31", "2021-04-01", "2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08", "2021-04-09", "2021-04-12", "2021-04-13", "2021-04-14", "2021-04-15", "2021-04-16", "2021-04-19", "2021-04-20", 
                            "2021-04-21", "2021-04-22", "2021-04-23", "2021-04-26", "2021-04-27", "2021-04-28", "2021-04-29", "2021-04-30", "2021-05-03", "2021-05-04", "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-10", "2021-05-11", 
                            "2021-05-12", "2021-05-13", "2021-05-14", "2021-05-17", "2021-05-18", "2021-05-19", "2021-05-20", "2021-05-21", "2021-05-24", "2021-05-25", "2021-05-26", "2021-05-27", "2021-05-28", "2021-06-01", "2021-06-02", 
                            "2021-06-03", "2021-06-04", "2021-06-07", "2021-06-08", "2021-06-09", "2021-06-10", "2021-06-11", "2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18", "2021-06-21", "2021-06-22", "2021-06-23", 
                            "2021-06-24", "2021-06-25", "2021-06-28", "2021-06-29", "2021-06-30", "2021-07-01", "2021-07-02", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", 
                            "2021-07-16", "2021-07-19", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-08-02", "2021-08-03", "2021-08-04", "2021-08-05", 
                            "2021-08-06", "2021-08-09", "2021-08-10", "2021-08-11", "2021-08-12", "2021-08-13", "2021-08-16", "2021-08-17", "2021-08-18", "2021-08-19", "2021-08-20", "2021-08-23", "2021-08-24", "2021-08-25", "2021-08-26", 
                            "2021-08-27", "2021-08-30", "2021-08-31", "2021-09-01", "2021-09-02", "2021-09-03", "2021-09-07", "2021-09-08", "2021-09-09", "2021-09-10", "2021-09-13", "2021-09-14", "2021-09-15", "2021-09-16", "2021-09-17", 
                            "2021-09-20", "2021-09-21", "2021-09-22", "2021-09-23", "2021-09-24", "2021-09-27", "2021-09-28", "2021-09-29", "2021-09-30", "2021-10-01", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07", "2021-10-08", 
                            "2021-10-11", "2021-10-12", "2021-10-13", "2021-10-14", "2021-10-15", "2021-10-18", "2021-10-19", "2021-10-20", "2021-10-21", "2021-10-22", "2021-10-25", "2021-10-26", "2021-10-27", "2021-10-28", "2021-10-29", 
                            "2021-11-01", "2021-11-02", "2021-11-03", "2021-11-04", "2021-11-05", "2021-11-08", "2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12", "2021-11-15", "2021-11-16", "2021-11-17", "2021-11-18", "2021-11-19", 
                            "2021-11-22", "2021-11-23", "2021-11-24", "2021-11-26", "2021-11-29", "2021-11-30", "2021-12-01", "2021-12-02", "2021-12-03", "2021-12-06", "2021-12-07", "2021-12-08", "2021-12-09", "2021-12-10", "2021-12-13", 
                            "2021-12-14", "2021-12-15", "2021-12-16", "2021-12-17", "2021-12-20", "2021-12-21", "2021-12-22", "2021-12-23", "2021-12-27", "2021-12-28", "2021-12-29", "2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04", 
                            "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14", "2022-01-18", "2022-01-19", "2022-01-20", "2022-01-21", "2022-01-24", "2022-01-25", "2022-01-26", 
                            "2022-01-27", "2022-01-28", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04", "2022-02-07", "2022-02-08", "2022-02-09", "2022-02-10", "2022-02-11", "2022-02-14", "2022-02-15", "2022-02-16", 
                            "2022-02-17", "2022-02-18", "2022-02-22", "2022-02-23", "2022-02-24", "2022-02-25", "2022-02-28", "2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04", "2022-03-07", "2022-03-08", "2022-03-09", "2022-03-10", 
                            "2022-03-11", "2022-03-14", "2022-03-15", "2022-03-16", "2022-03-17", "2022-03-18", "2022-03-21", "2022-03-22", "2022-03-23", "2022-03-24", "2022-03-25", "2022-03-28", "2022-03-29", "2022-03-30", "2022-03-31", 
                            "2022-04-01", "2022-04-04", "2022-04-05", "2022-04-06", "2022-04-07", "2022-04-08", "2022-04-11", "2022-04-12", "2022-04-13", "2022-04-14", "2022-04-18", "2022-04-19", "2022-04-20", "2022-04-21", "2022-04-22", 
                            "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29", "2022-05-02", "2022-05-03", "2022-05-04", "2022-05-05", "2022-05-06", "2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12", "2022-05-13", 
                            "2022-05-16", "2022-05-17", "2022-05-18", "2022-05-19", "2022-05-20", "2022-05-23", "2022-05-24", "2022-05-25", "2022-05-26", "2022-05-27", "2022-05-31", "2022-06-01", "2022-06-02", "2022-06-03", "2022-06-06", 
                            "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10", "2022-06-13", "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17", "2022-06-21", "2022-06-22", "2022-06-23", "2022-06-24", "2022-06-27", "2022-06-28", 
                            "2022-06-29", "2022-06-30", "2022-07-01", "2022-07-05", "2022-07-06", "2022-07-07", "2022-07-08", "2022-07-11", "2022-07-12", "2022-07-13", "2022-07-14", "2022-07-15", "2022-07-18", "2022-07-19", "2022-07-20", 
                            "2022-07-21", "2022-07-22", "2022-07-25", "2022-07-26", "2022-07-27", "2022-07-28", "2022-07-29", "2022-08-01", "2022-08-02", "2022-08-03", "2022-08-04", "2022-08-05", "2022-08-08", "2022-08-09", "2022-08-10", 
                            "2022-08-11", "2022-08-12", "2022-08-15", "2022-08-16", "2022-08-17", "2022-08-18", "2022-08-19", "2022-08-22", "2022-08-23", "2022-08-24", "2022-08-25", "2022-08-26", "2022-08-29", "2022-08-30", "2022-08-31", 
                            "2022-09-01", "2022-09-02", "2022-09-06", "2022-09-07", "2022-09-08", "2022-09-09", "2022-09-12", "2022-09-13", "2022-09-14", "2022-09-15", "2022-09-16", "2022-09-19", "2022-09-20", "2022-09-21", "2022-09-22", 
                            "2022-09-23", "2022-09-26", "2022-09-27", "2022-09-28", "2022-09-29", "2022-09-30", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07", "2022-10-10", "2022-10-11", "2022-10-12", "2022-10-13", 
                            "2022-10-14", "2022-10-17", "2022-10-18", "2022-10-19", "2022-10-20", "2022-10-21", "2022-10-24", "2022-10-25", "2022-10-26", "2022-10-27", "2022-10-28", "2022-10-31", "2022-11-01", "2022-11-02", "2022-11-03", 
                            "2022-11-04", "2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14", "2022-11-15", "2022-11-16", "2022-11-17", "2022-11-18", "2022-11-21", "2022-11-22", "2022-11-23", "2022-11-25", 
                            "2022-11-28", "2022-11-29", "2022-11-30", "2022-12-01", "2022-12-02", "2022-12-05", "2022-12-06", "2022-12-07", "2022-12-08", "2022-12-09", "2022-12-12", "2022-12-13", "2022-12-14", "2022-12-15", "2022-12-16", 
                            "2022-12-19", "2022-12-20", "2022-12-21", "2022-12-22", "2022-12-23", "2022-12-27"]
            dts_choose_3 = dts_choose_1[-self.train_his + 1 :] + dts_choose_2
            df = df_features
            df["shouyi"] = (df["open"] - df["prev_close"]) / df["prev_close"]

            param_list = []
            for dt_one in tqdm(dts_choose_3):
                param_list.append((dts_all, df, dt_one))

            pool = multiprocessing.Pool(30)
            results = []
            for i in range(len(dts_choose_3)):
                results.append(pool.apply_async(fun_similar, param_list[i]))
            pool.close()
            pool.join()
            matrx_1 = []
            for res in results:
                matrx_1.append(res.get())
            matrx_2 = []
            for one in range(len(dts_choose_2)):
                matrx_2.append(matrx_1[one : one + self.train_his])
            matrx_1 = np.array(matrx_2)
            np.save(
                self.root_data_path + "matrx_" + str(self.train_his) + "_train.npy",
                matrx_1,
            )
            print(matrx_1.shape)

        N = len(stock_choose)
        F = len(self.feature_cols)
        Fixed_Matrices = [matrx]
        Matrix_Weights = [1]

        x_train = tf.constant(X_train, dtype=tf.float32)
        y_train = tf.constant(Y_train, dtype=tf.float32)
        x_train_matrx = tf.constant(matrx_1, dtype=tf.float32)
        print(X_train.shape, Y_train.shape, x_train_matrx.shape)

        for model_num in range(self.number_of_models[0], self.number_of_models[-1]):
            print("###" + str("2022-12-31") + "###model_num=" + str(model_num))

            for epoch in self.epoches_list:
                tf.keras.backend.clear_session()
                model = GCGRU(
                    N,
                    F,
                    self.P,
                    self.Units_GCN,
                    self.Units_GRU,
                    self.Units_FC,
                    Fixed_Matrices,
                    Matrix_Weights,
                    self.Is_Dyn,
                )
                model.build(input_shape=[(None, self.P, N, F), (None, self.P, N, N)])
                model.summary()
                model.compile(
                    loss="mean_squared_error",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00002),
                    metrics=["mae"],
                )

                model_file_path_last = self.model_file_path.replace(
                    "-n0", "-n" + str(model_num)
                )
                model_file_path_last = model_file_path_last.replace(
                    "stockrnn_basic_model-" + str("2022-12-31") + ".h5", ""
                )
                model_file_path_last = (
                    model_file_path_last
                    + str("2022-12-31")
                    + "_epochs_"
                    + str(epoch)
                    + "/"
                )

                if not os.path.exists(model_file_path_last):
                    os.makedirs(model_file_path_last)

                model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_file_path_last + "GCNGRU",
                    monitor="loss",
                    save_weights_only=True,
                    save_best_only=True,
                )
                model.fit(
                    [x_train, x_train_matrx],
                    y_train,
                    batch_size=self.batch_size,
                    epochs=epoch,
                    callbacks=[model_checkpoint],
                )

    def process(
        self,
    ):

        def internal_run():
            matrx, matrx_1, stock_choose, X_pred, origin_idx_pred = (
                self.construct_pred_data()
            )
            x_pred = tf.constant(X_pred, dtype=tf.float32)
            x_pred_matrx = tf.constant(matrx_1, dtype=tf.float32)
            print(x_pred.shape, x_pred_matrx.shape)

            N = len(stock_choose)
            F = len(self.feature_cols)
            Fixed_Matrices = [matrx]
            Matrix_Weights = [1]
            for number in range(self.number_of_models[0], self.number_of_models[-1]):
                print("predict data with model {}\t".format(number))

                for epoch in self.epoches_list:
                    model_file_path_last = self.model_file_path.replace(
                        "-n0", "-n" + str(number)
                    )
                    model_file_path_last = model_file_path_last.replace(
                        "stockrnn_basic_model-" + str(self.model_dt) + ".h5", ""
                    )
                    model_file_path_last = (
                        model_file_path_last
                        + str(self.model_dt)
                        + "_epochs_"
                        + str(epoch)
                    )

                    tf.keras.backend.clear_session()

                    model = GCGRU(
                        N,
                        F,
                        self.P,
                        self.Units_GCN,
                        self.Units_GRU,
                        self.Units_FC,
                        Fixed_Matrices,
                        Matrix_Weights,
                        self.Is_Dyn,
                    )

                    model.build(
                        input_shape=[(None, self.P, N, F), (None, self.P, N, N)]
                    )

                    model.load_weights(model_file_path_last + "/GCNGRU")
                    print(model_file_path_last + "/GCNGRU")
                    pred_list = model.predict([x_pred, x_pred_matrx])
                    pred_list = pred_list.tolist()
                    pred_all = [
                        "2023-01-03.csv", "2023-01-04.csv", "2023-01-05.csv", "2023-01-06.csv", "2023-01-09.csv", "2023-01-10.csv", "2023-01-11.csv", "2023-01-12.csv", "2023-01-13.csv", "2023-01-17.csv",
                        "2023-01-18.csv", "2023-01-19.csv", "2023-01-20.csv", "2023-01-23.csv", "2023-01-24.csv", "2023-01-25.csv", "2023-01-26.csv", "2023-01-27.csv", "2023-01-30.csv", "2023-01-31.csv", 
                        "2023-02-01.csv", "2023-02-02.csv", "2023-02-03.csv", "2023-02-06.csv", "2023-02-07.csv", "2023-02-08.csv", "2023-02-09.csv", "2023-02-10.csv", "2023-02-13.csv", "2023-02-14.csv",
                        "2023-02-15.csv", "2023-02-16.csv", "2023-02-17.csv", "2023-02-21.csv", "2023-02-22.csv", "2023-02-23.csv", "2023-02-24.csv", "2023-02-27.csv", "2023-02-28.csv", "2023-03-01.csv",
                        "2023-03-02.csv", "2023-03-03.csv", "2023-03-06.csv", "2023-03-07.csv", "2023-03-08.csv", "2023-03-09.csv", "2023-03-10.csv", "2023-03-13.csv", "2023-03-14.csv", "2023-03-15.csv",
                        "2023-03-16.csv", "2023-03-17.csv", "2023-03-20.csv", "2023-03-21.csv", "2023-03-22.csv", "2023-03-23.csv", "2023-03-24.csv", "2023-03-27.csv", "2023-03-28.csv", "2023-03-29.csv",
                        "2023-03-30.csv", "2023-03-31.csv", "2023-04-03.csv", "2023-04-04.csv", "2023-04-05.csv", "2023-04-06.csv", "2023-04-10.csv", "2023-04-11.csv", "2023-04-12.csv", "2023-04-13.csv",
                        "2023-04-14.csv", "2023-04-17.csv", "2023-04-18.csv", "2023-04-19.csv", "2023-04-20.csv", "2023-04-21.csv", "2023-04-24.csv", "2023-04-25.csv", "2023-04-26.csv", "2023-04-27.csv",
                        "2023-04-28.csv", "2023-05-01.csv", "2023-05-02.csv", "2023-05-03.csv", "2023-05-04.csv", "2023-05-05.csv", "2023-05-08.csv", "2023-05-09.csv", "2023-05-10.csv", "2023-05-11.csv",
                        "2023-05-12.csv", "2023-05-15.csv", "2023-05-16.csv", "2023-05-17.csv", "2023-05-18.csv", "2023-05-19.csv", "2023-05-22.csv", "2023-05-23.csv", "2023-05-24.csv", "2023-05-25.csv",
                        "2023-05-26.csv", "2023-05-30.csv", "2023-05-31.csv", "2023-06-01.csv", "2023-06-02.csv", "2023-06-05.csv", "2023-06-06.csv", "2023-06-07.csv", "2023-06-08.csv", "2023-06-09.csv",
                        "2023-06-12.csv", "2023-06-13.csv", "2023-06-14.csv", "2023-06-15.csv", "2023-06-16.csv", "2023-06-20.csv", "2023-06-21.csv", "2023-06-22.csv", "2023-06-23.csv", "2023-06-26.csv",
                        "2023-06-27.csv", "2023-06-28.csv", "2023-06-29.csv", "2023-06-30.csv", "2023-07-03.csv", "2023-07-05.csv", "2023-07-06.csv", "2023-07-07.csv", "2023-07-10.csv", "2023-07-11.csv",
                        "2023-07-12.csv", "2023-07-13.csv", "2023-07-14.csv", "2023-07-17.csv", "2023-07-18.csv", "2023-07-19.csv", "2023-07-20.csv", "2023-07-21.csv", "2023-07-24.csv", "2023-07-25.csv",
                        "2023-07-26.csv", "2023-07-27.csv", "2023-07-28.csv", "2023-07-31.csv", "2023-08-01.csv", "2023-08-02.csv", "2023-08-03.csv", "2023-08-04.csv", "2023-08-07.csv", "2023-08-08.csv",
                        "2023-08-09.csv", "2023-08-10.csv", "2023-08-11.csv", "2023-08-14.csv", "2023-08-15.csv", "2023-08-16.csv", "2023-08-17.csv", "2023-08-18.csv", "2023-08-21.csv", "2023-08-22.csv",
                        "2023-08-23.csv", "2023-08-24.csv", "2023-08-25.csv", "2023-08-28.csv", "2023-08-29.csv", "2023-08-30.csv", "2023-08-31.csv", "2023-09-01.csv", "2023-09-05.csv", "2023-09-06.csv",
                        "2023-09-07.csv", "2023-09-08.csv", "2023-09-11.csv", "2023-09-12.csv", "2023-09-13.csv", "2023-09-14.csv", "2023-09-15.csv", "2023-09-18.csv", "2023-09-19.csv", "2023-09-20.csv",
                        "2023-09-21.csv", "2023-09-22.csv", "2023-09-25.csv", "2023-09-26.csv", "2023-09-27.csv", "2023-09-28.csv", "2023-09-29.csv", "2023-10-02.csv", "2023-10-03.csv", "2023-10-04.csv", 
                        "2023-10-05.csv", "2023-10-06.csv", "2023-10-09.csv", "2023-10-10.csv", "2023-10-11.csv", "2023-10-12.csv", "2023-10-13.csv", "2023-10-16.csv", "2023-10-17.csv", "2023-10-18.csv",
                        "2023-10-19.csv", "2023-10-20.csv", "2023-10-23.csv", "2023-10-24.csv", "2023-10-25.csv", "2023-10-26.csv", "2023-10-27.csv", "2023-10-30.csv", "2023-10-31.csv", "2023-11-01.csv",
                        "2023-11-02.csv", "2023-11-03.csv", "2023-11-06.csv", "2023-11-07.csv", "2023-11-08.csv", "2023-11-09.csv", "2023-11-10.csv", "2023-11-13.csv", "2023-11-14.csv", "2023-11-15.csv",
                        "2023-11-16.csv", "2023-11-17.csv", "2023-11-20.csv", "2023-11-21.csv", "2023-11-22.csv", "2023-11-24.csv", "2023-11-27.csv", "2023-11-28.csv", "2023-11-29.csv", "2023-11-30.csv",
                        "2023-12-01.csv", "2023-12-04.csv", "2023-12-05.csv", "2023-12-06.csv", "2023-12-07.csv", "2023-12-08.csv", "2023-12-11.csv", "2023-12-12.csv", "2023-12-13.csv", "2023-12-14.csv",
                        "2023-12-15.csv", "2023-12-18.csv", "2023-12-19.csv", "2023-12-20.csv", "2023-12-21.csv", "2023-12-22.csv", "2023-12-26.csv", "2023-12-27.csv", "2023-12-28.csv"]
                    pred_save_path = (
                        model_file_path_last.replace(str(self.model_dt) + "_", "") + "/"
                    )

                    if not os.path.exists(pred_save_path):
                        os.makedirs(pred_save_path)

                    for i in range(len(pred_list)):
                        data_all = []
                        for j in range(len(pred_list[i])):
                            one = []
                            one.append(stock_choose[j])
                            one.append(pred_all[i][0:10])
                            one.append(pred_list[i][j])
                            data_all.append(one)
                        df = pd.DataFrame(
                            columns=["kdcode", "dt", "score"], data=data_all
                        )
                        df = df.sort_values(["kdcode", "dt"])
                        df.to_csv(
                            pred_save_path + pred_all[i],
                            header=True,
                            index=False,
                            encoding="utf_8_sig",
                        )

        def get_all_predict_data(path1, file_name, dict_day):
            f2 = open(path1 + file_name, "r")
            lines = f2.readlines()
            for line3 in lines:
                line3 = line3.strip()
                line3 = line3.split(",")
                if line3[1] == "dt":
                    continue
                else:
                    dict_day.setdefault(line3[0], []).append(line3[2])
            return dict_day

        def cal_ave(value):
            for i in range(len(value)):
                value[i] = float(value[i])
            return np.mean(value)

        def combine():
            predict_files_path = self.predict_folder_path.replace(
                "prediction", "models/epochs_" + str(self.epoches_list[0])
            )
            files = os.listdir(predict_files_path)
            days_all_last = sorted(files)
            for i in tqdm(range(len(days_all_last))):
                print("calculate last prediction data in {}".format(days_all_last[i]))
                dict_day = {}
                for k in self.epoches_list:
                    for j in range(self.number_of_models[1]):
                        path = (
                            self.predict_folder_path.replace(
                                "-model-n0", "-model-n" + str(j)
                            )
                            + "/"
                        )
                        path = path.replace("prediction", "models/epochs_" + str(k))
                        dict_day = get_all_predict_data(
                            path, days_all_last[i], dict_day
                        )
                    data_all = []
                    for key in dict_day:
                        one = []
                        one.append(key)
                        one.append(str(days_all_last[i])[0:10])
                        one.append(cal_ave(dict_day[key]))
                        data_all.append(one)
                    df = pd.DataFrame(columns=["kdcode", "dt", "score"], data=data_all)
                    df.to_csv(
                        self.predict_folder_save
                        + str(k)
                        + "/prediction/"
                        + days_all_last[i],
                        header=True,
                        index=False,
                        encoding="utf_8_sig",
                    )

        internal_run()
        combine()


def fun_train_pred(
    CUDA_VISIBLE_DEVICES,
    root_data_path,
    T,
    train_his,
    epoches_list,
    P,
    Is_Dyn,
    Units_GCN,
    Units_GRU,
    Units_FC,
    number_of_models,
    model_data_path,
    batch_size,
    model_dt,
    s_dt,
    e_dt,
):
    model = StockrnnBasicModel(
        model_dt=model_dt,
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        root_data_path=root_data_path,
        T=T,
        train_his=train_his,
        epoches_list=epoches_list,
        P=P,
        Is_Dyn=Is_Dyn,
        Units_GCN=Units_GCN,
        Units_GRU=Units_GRU,
        Units_FC=Units_FC,
        number_of_models=number_of_models,
        model_data_path=model_data_path,
        batch_size=batch_size,
    )
    model.train()
    model.process()


def fun_last(
    root_data_path, T, train_his, epoches_list, P, number_of_models, model_data_path
):
    batch_size = 128
    root_data_path = root_path + root_data_path + "/"
    if not os.path.exists(root_data_path):
        os.makedirs(root_data_path)
    model_data_path = root_path + model_data_path + "/"
    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)
    Units_GCN = [20, 15]
    Units_GRU = [12]
    Units_FC = [10, 1]

    Is_Dyn = False
    CUDA_VISIBLE_DEVICES = gpu_id

    model_dt, s_dt, e_dt = "2022-12-31", "2023-01-02", "2023-12-29"
    fun_train_pred(
        CUDA_VISIBLE_DEVICES,
        root_data_path,
        T,
        train_his,
        epoches_list,
        P,
        Is_Dyn,
        Units_GCN,
        Units_GRU,
        Units_FC,
        number_of_models,
        model_data_path,
        batch_size,
        model_dt,
        s_dt,
        e_dt,
    )


T = 3
P = 15
train_his = 15
data_choose = "NASDAQ100"
root_data_path = (
    data_choose + "_T_" + str(T) + "_his_" + str(train_his) + "_P_" + str(P)
)
print("root_data_path为:", root_data_path)
model_data_path = (
    data_choose + "_T_" + str(T) + "_his_" + str(train_his) + "_P_" + str(P)
)
print("model_data_path为:", model_data_path)
epoches_list = [5, 6, 8, 10, 12, 13, 14, 15]
number_of_models = [0, 8]
fun_last(
    root_data_path, T, train_his, epoches_list, P, number_of_models, model_data_path
)
