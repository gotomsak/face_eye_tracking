from imutils.video import VideoStream
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
from scipy.spatial import distance
import json
import pathlib
# from . import eye_open_check
import math
# earの計算
def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)


# 目にマーカーを付ける
def eye_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


# 5秒ごとのwを返す関数
# def (all_angle_list):


# 5秒ごとの頻度を返す関数　回数 / 12
def section_frequency(section_list):
    section_frequency = []
    if np.array(section_list).ndim == 2:
        for i in section_list:
            section_frequency.append([i[0] / 12, i[1] / 12])
    else:
        for i in section_list:
            # 5/12?
            section_frequency.append(i / 12)

    return section_frequency


# 5秒おきの顔の移動量に変える
def change_5_second(all_change_list):
    change_5_second = []
    for z in all_change_list:
        x_all = 0
        y_all = 0
        for i in z:
            x_all += abs(i[0])
            y_all += abs(i[1])
        change_5_second.append([x_all, y_all])
    return change_5_second


# セクションごとの集中度を出す
def section_concentration(frequency):
    concentration_list = []
    if np.array(frequency).ndim == 2:
        for i in range(len(frequency)):
            frequency[i] = sum(frequency[i])

    for i in frequency:
        try:
            concentration_list.append(round((i - max(frequency)) / (min(frequency) - max(frequency)), 2))
        except:
            concentration_list.append(0)
    return concentration_list


# セクションごとの集中度を出す
def section_concentration_new(c1, c2, w):
    b_concentration = []
    m_concentration = []
    concentration = []
    print(len(c1))
    print(len(c2))
    print(len(w))
    for i in range(len(c1)):

        b_concentration.append(c1[i] * w[i])

    for i in range(len(c2)):
        m_concentration.append(c2[i] * (1 - w[i]))

    for i in range(len(c1)):
        concentration.append(b_concentration[i] + m_concentration[i])

    return concentration


# 5秒間のwを平均化したものを返す
# w = 0 (r = 0, p = 0, y = 0)
# w = 1 - ((y/tr + p/tp + r/ty) / 3)
def create_w_list(all_angle_list):
    angle_threshold_up = 150
    angle_threshold_down = 175
    angle_threshold_yaw = 20
    angle_threshold_roll = 15
    angle_threshold_pitch = 12.5
    w_list = []
    w = []
    for j in all_angle_list:
        section_w_list = []
        for i in j:
            print(i)
            if i[0] == 0 and i[1] == 0 and i[2] == 0:
                section_w_list.append(0)
            else:
                section_w_list.append(
                    1 - ((i[0] / angle_threshold_yaw + i[1] / angle_threshold_pitch + i[2] / angle_threshold_roll) / 3))
        w_list.append(section_w_list)

    for i in w_list:
        sum_j = 0
        for j in i:
            sum_j += j
        w.append(sum_j / len(i))

    return w


if __name__ == '__main__':
    file_path = './movie/face_eye_data/inagawa/movie1.mp4'
    json_file_path = file_path+"conc.json"
    # json_dir_path = './json_file/blink_data_/nedati/'
    right_threshold = 0.2
    left_threshold = 0.2
    load_data = open("movie/face_eye_data/inagawa/movie2.mp4cv.json", 'r')
    load_data = json.load(load_data)
    print(load_data)
    # 動画の処理をするmain関数
    all_blink_list = load_data['blink']
    all_change_list =load_data['face']
    all_angle_list = load_data['angle']


    change_5_second = change_5_second(all_change_list)

    blink_frequency = section_frequency(all_blink_list)
    change_frequency = section_frequency(change_5_second)

    c1 = section_concentration(blink_frequency)
    c2 = section_concentration(change_frequency)
    # w = 1 - ((r/tr + p/tp + y/ty) / 3))を返す
    w = create_w_list(all_angle_list)


    # print("c1:", c1)
    # print("c2:", c2)
    # print("w:", w)
    C_list = section_concentration_new(c1, c2, w)
    print("C_List:", C_list)

    C = sum(C_list) / len(C_list)
    print("C:", C)
    data = {
        'c1': c1,
        'c2': c2,
        'w': w,
        'section_concentration': C_list,
        'concentration': C,
    }

    # p = pathlib.Path(json_dir_path).mkdir(parents=True, exist_ok=True)
    with open(json_file_path, 'w')as f:
        json.dump(data, f, indent=4)
