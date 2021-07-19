import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np
import itertools
from scipy import stats


def read_data(file_path, index):
    file = open(file_path)
    file_data = json.load(file)
    return file_data[index]


def time_series_images(c_list, save_name, margin_time, list_size):
    plt.figure()
    c_list = c_list[:list_size]
    x = range(0, len(c_list) * margin_time, margin_time)
    ave_size = int(len(c_list) * 0.3)
    b = np.ones(ave_size) / ave_size
    y = np.convolve(c_list, b, mode='same')
    plt.plot(x, y, label='MovingAverage')
    plt.plot(x, c_list, label='OriginalSeries')
    plt.legend()
    plt.savefig(save_name + '.png')


def save_concentration_images(x_c, y_c, save_name):
    plt.figure()
    clf = linear_model.LinearRegression()
    if len(y_c) > len(x_c):
        size = len(x_c)
    else:
        size = len(y_c)
    x_c, y_c = truncation(x_c[:size], y_c[:size])
    x_c = np.array(x_c)
    y_c = np.array(y_c)
    x_df = pd.DataFrame(x_c)
    y_df = pd.DataFrame(y_c)

    clf.fit(x_df, y_df)

    plt.scatter(x_c, y_c)
    plt.plot(x_c, clf.predict(x_df))

    plt.savefig(save_name + '.png')
    corr = np.corrcoef(x_c, y_c)
    fw = open(save_name + '.json', 'w')

    save_result = {
        "回帰係数= ": clf.coef_.tolist(),
        "切片= ": clf.intercept_.tolist(),
        "決定係数= ": clf.score(x_df, y_df),
        "相関係数= ": corr[0, 1],
        "データ数= ": len(x_c)
    }
    json.dump(save_result, fw, indent=4, ensure_ascii=False)
    return x_c, y_c


# 0から1の範囲に収まらなかった集中度を切り捨て
def truncation(x_c, y_c):
    new_x_c = []
    new_y_c = []

    for i in range(len(x_c)):

        if x_c[i] > 1 or x_c[i] < 0 or y_c[i] > 1 or y_c[i] < 0:
            continue

        new_x_c.append(x_c[i])
        new_y_c.append(y_c[i])

    return new_x_c, new_y_c


# 集中状態と＊状態の相関の画像を保存し，集中度のリストを返す
def correlation_concentration(x_path, y_path, index, save_name):
    x_c = read_data(x_path, index)
    y_c = read_data(y_path, index)

    x_c, y_c = save_concentration_images(x_c, y_c, save_name)

    return x_c.tolist(), y_c.tolist()


def outlier_calculation(c_list):
    q25, q75 = np.percentile(c_list, [25, 75])
    iqr = q75 - q25
    print(q25)
    print(q75)
    lower_bound = q25
    upper_bound = q75
    print(lower_bound)
    print(upper_bound)

    return np.array(c_list)[((c_list < upper_bound) & (c_list > lower_bound))]


if __name__ == '__main__':
    path_root = 'movie/Production'

    a_file_name = '*concentration.mp4conc*'
    b_file_name = '*watch.mp4conc*'
    c_file_name = '*game.mp4conc*'
    dic_name = 'section_concentration'

    p = Path(path_root)
    user_list = ["userA", "userB", "userD", "userE", "userG"]
    cwx_all = []
    cwy_all = []
    cgx_all = []
    cgy_all = []
    cw_name = "cw0"
    cg_name = "cg0"

    for i in list(p.glob('*')):
        for j in user_list:
            if j in str(i):
                a_file_path = list(p.glob(j + '/' + a_file_name))[0]
                b_file_path = list(p.glob(j + '/' + b_file_name))[0]
                c_file_path = list(p.glob(j + '/' + c_file_name))[0]


                a = read_data(a_file_path, dic_name)
                b = read_data(b_file_path, dic_name)
                c = read_data(c_file_path, dic_name)

                # list_size = len(a)
                # if len(a) > len(b) and len(b) < len(c):
                #     list_size = len(b)
                # if len(b) > len(c) and len(c) < len(a):
                #     list_size = len(c)

                # time_series_images(a, str(i) + '/' + 'time_series_a', 5, list_size)
                # time_series_images(b, str(i) + '/' + 'time_series_b', 5, list_size)
                # time_series_images(c, str(i) + '/' + 'time_series_c', 5, list_size)

                time_series_images(a, str(i) + '/' + 'time_series_a', 5, len(a))
                time_series_images(b, str(i) + '/' + 'time_series_b', 5, len(b))
                time_series_images(c, str(i) + '/' + 'time_series_c', 5, len(c))


    #             cwx, cwy = correlation_concentration(a_file_path, b_file_path,
    #                                                  dic_name,
    #                                                  str(i) + '/' + cw_name)
    #             cgx, cgy = correlation_concentration(a_file_path, c_file_path,
    #                                                  dic_name,
    #                                                  str(i) + '/' + cg_name)
    #
    #             cwx_all.append(cwx)
    #             cwy_all.append(cwy)
    #             cgx_all.append(cgx)
    #             cgy_all.append(cgy)
    #
    #
    # cwx_all = list(itertools.chain.from_iterable(cwx_all))
    # cwy_all = list(itertools.chain.from_iterable(cwy_all))
    # cgx_all = list(itertools.chain.from_iterable(cgx_all))
    # cgy_all = list(itertools.chain.from_iterable(cgy_all))
    #
    # cwx_all = outlier_calculation(cwx_all)
    # cwy_all = outlier_calculation(cwy_all)
    # cgx_all = outlier_calculation(cgx_all)
    # cgy_all = outlier_calculation(cgy_all)
    #
    # cw_size = len(cwx_all)
    # if len(cwx_all) > len(cwy_all):
    #     cw_size = len(cwy_all)
    #
    # cg_size = len(cgx_all)
    # if len(cgx_all) > len(cgy_all):
    #     cg_size = len(cgy_all)
    #
    # cwx_all = cwx_all[:cw_size]
    # cwy_all = cwy_all[:cw_size]
    # cgx_all = cgx_all[:cg_size]
    # cgy_all = cgy_all[:cg_size]
    #
    # cwx_all, cwy_all = save_concentration_images(cwx_all, cwy_all, path_root + '/' + cw_name)
    # cgx_all, cgy_all = save_concentration_images(cgx_all, cgy_all, path_root + '/' + cg_name)
