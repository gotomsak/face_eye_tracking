import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np
import itertools


def save_concentration_images(x_c, y_c, save_name):
    plt.figure()
    clf = linear_model.LinearRegression()
    if len(y_c) > len(x_c):
        size = len(x_c)
    else:
        size = len(y_c)
    x_c = np.array(x_c[:size])
    y_c = np.array(y_c[:size])
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
        "相関係数= ": corr[0, 1]
    }
    json.dump(save_result, fw, indent=4, ensure_ascii=False)
    return x_c, y_c


# 集中状態と＊状態の相関の画像を保存し，集中度のリストを返す
def correlation_concentration(x_path, y_path, index, save_name):
    x_file = open(x_path)
    y_file = open(y_path)
    x_data = json.load(x_file)
    y_data = json.load(y_file)

    x_c = x_data[index]
    y_c = y_data[index]
    x_c, y_c = save_concentration_images(x_c, y_c, save_name)

    return x_c, y_c


if __name__ == '__main__':
    path_root = 'movie/Production'

    # user_name = 'tomono_'
    a_file_name = '*concentration.mp4conc*'
    b_file_name = '*watch.mp4conc*'
    c_file_name = '*game.mp4conc*'

    p = Path(path_root)
    user_list = ["userA", "userB", "userD", "userE", "userG"]
    cwx_all = []
    cwy_all = []
    cgx_all = []
    cgy_all = []

    for i in list(p.glob('*')):
        for j in user_list:
            if j in str(i):
                a_file_path = list(p.glob(j + '/' + a_file_name))[0]
                b_file_path = list(p.glob(j + '/' + b_file_name))[0]
                c_file_path = list(p.glob(j + '/' + c_file_name))[0]

                cwx, cwy = correlation_concentration(a_file_path, b_file_path,
                                                     'section_concentration',
                                                     str(i) + '/' + 'cw')
                cgx, cgy = correlation_concentration(a_file_path, c_file_path,
                                                     'section_concentration',
                                                     str(i) + '/' + 'cg')

                cwx_all.append(cwx)
                cwy_all.append(cwy)
                cgx_all.append(cgx)
                cgy_all.append(cgy)

    cwx_all = list(itertools.chain.from_iterable(cwx_all))
    cwy_all = list(itertools.chain.from_iterable(cwy_all))
    cgx_all = list(itertools.chain.from_iterable(cgx_all))
    cgy_all = list(itertools.chain.from_iterable(cgy_all))

    cwx_all, cwy_all = save_concentration_images(cwx_all, cwy_all, path_root + "cw")
    cgx_all, cgy_all = save_concentration_images(cgx_all, cgy_all, path_root + "cg")

    # for i in list(p.glob('*/*concentration.mp4conc*')):
    #     print(str(i))
    #     f = open(str(i))
    #     data = json.load(f)
    #     c1_list = data['c1']
    #     c2_list = data['c2']
    #     c_list = data['section_concentration']
    #     plt.scatter(c_list, c1_list)
    #     plt.savefig(str(i) + '.png')
    #     plt.figure()
    #
    #     print(c1_list)
