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


# 閾値を返す関数
def eye_open(file_name):
    cap = cv2.VideoCapture(file_name)
    face_cascade = cv2.CascadeClassifier('classification_tool/haarcascade_frontalface_alt2.xml')
    face_parts_detector = dlib.shape_predictor('classification_tool/shape_predictor_68_face_landmarks.dat')

    right_eye_list = []
    left_eye_list = []

    while True:
        tick = cv2.getTickCount()

        ret, rgb = cap.read()
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except:
            break
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0, :]
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face = dlib.rectangle(x, y, x + w, y + h)
            face_parts = face_parts_detector(gray, face)
            face_parts = face_utils.shape_to_np(face_parts)

            left_eye_ear = calc_ear(face_parts[42:48])
            left_eye_list.append(left_eye_ear)
            cv2.putText(rgb, "left eye EAR:{} ".format(round(left_eye_ear, 3)),
                        (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            right_eye_ear = calc_ear(face_parts[36:42])
            cv2.putText(rgb, "right eye EAR:{} ".format(round(right_eye_ear, 3)),
                        (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            right_eye_list.append(right_eye_ear)
            leftEyeHull = cv2.convexHull(face_parts[42:48])
            rightEyeHull = cv2.convexHull(face_parts[36:42])
            cv2.drawContours(rgb, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(rgb, [rightEyeHull], -1, (0, 255, 0), 1)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)

        # cv2.imshow('frame', rgb)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    right_ave = sum(i for i in right_eye_list) / len(right_eye_list)
    left_ave = sum(i for i in left_eye_list) / len(left_eye_list)

    right_s = math.sqrt((1 / len(right_eye_list) * pow(sum(i - right_ave for i in right_eye_list), 2)))
    left_s = math.sqrt((1 / len(left_eye_list) * pow(sum(i - left_ave for i in left_eye_list), 2)))
    right_threshold = right_ave - right_s
    left_threshold = left_ave - left_s

    cap.release()
    cv2.destroyAllWindows()
    return right_threshold, left_threshold


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


# セクションごとの集中度を出す
def section_concentration(section_list, max_frequency, min_frequency):
    concentration_list = []

    for i in section_list:
        print(i)
        try:
            concentration_list.append(round((i - max_frequency) / (min_frequency - max_frequency), 2))

        except:
            concentration_list.append(0)
    return concentration_list


# 返り値: 各1分おきのlistdata(瞬き, 顔の変化量, よそ見したときのフレーム数)
# 引数: ファイルパス
def cv_main(video_path, right_t_provisional, left_t_provisional):
    # 開眼度閾値
    right_t = right_t_provisional - 0.05
    left_t = left_t_provisional - 0.05

    EYE_AR_THRESH = (right_t_provisional + left_t_provisional) / 2
    EYE_AR_CONSEC_FRAMES = 1
    # 目を閉じたときのカウンター
    COUNTER = 0

    # 目を閉じたときのトータルカウント
    TOTAL = 0

    # 5秒間の瞬き回数
    blink_5_cnt = 0
    # 60秒間の瞬き回数
    blink_60_cnt = 0

    # すべてのフレームのカウント
    all_frame_cnt = 0
    section_5_frame_cnt = 0
    section_60_frame_cnt = 0

    # 5,60秒おきのすべての瞬き
    all_5_blink_list = []



    print("[INFO] loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('classification_tool/shape_predictor_68_face_landmarks.dat')
    face_cascade = cv2.CascadeClassifier('classification_tool/haarcascade_frontalface_alt2.xml')
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(video_path)

    while True:

        all_frame_cnt += 1
        section_5_frame_cnt += 1
        section_60_frame_cnt += 1
        rect_frag = False
        ret, frame = cap.read()

        # frame = imutils.resize(frame, width=500)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
        rects = detector(gray, 0)
        # if len(rects) == 0:
        #     cnt_looking_away += 1
        #     cv2.putText(frame, "away", (10, 195), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

        for rect in rects:

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)



            if len(faces) == 1:
                x, y, w, h = faces[0, :]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_gray = gray[y:(y + h), x:(x + w)]
                scale = 480 / h
                face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

                face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
                face_parts = predictor(face_gray_resized, face)
                face_parts = face_utils.shape_to_np(face_parts)

                left_eye = face_parts[42:48]
                eye_marker(face_gray_resized, left_eye)

                left_eye_ear = calc_ear(left_eye)
                #print(left_eye_ear)

                right_eye = face_parts[36:42]
                eye_marker(face_gray_resized, right_eye)

                right_eye_ear = calc_ear(right_eye)

                if right_eye_ear < right_t and left_eye_ear < left_t:
                    # 瞬き閾値より現在のearが下回った場合(目を閉じた時)
                    COUNTER += 1
                # 瞬き閾値より現在のearが上回った場合(目を開けた時)
                else:
                    # 　目を開けた時、カウンターが一定値以上だったら
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        blink_5_cnt += 1
                        blink_60_cnt += 1
                        cv2.putText(frame, "blink", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

                    COUNTER = 0
                #cv2.imshow("gray", face_gray_resized)




        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # 5秒おきに
        if section_5_frame_cnt == 150:

            all_5_blink_list.append(blink_5_cnt)
            section_5_change_list = []

            blink_5_cnt = 0
            section_5_frame_cnt = 0


    all_5_blink_list.append(blink_5_cnt)
    section_5_cnt = 0
    frame_5_cnt = 0


    # print(change_list)
    json_file_path = "only_blink_"+video_path+"cv.json"
    # 5秒おきのx,yの変化量をすべて足してまとめた
    data = {
        'blink_5': all_5_blink_list,
    }
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    cap.release()
    cv2.destroyAllWindows()

    return all_5_blink_list


if __name__ == '__main__':
    p = pathlib.Path('.')
    movie_dir_path = './movie/face_eye_data/*/*.mp4'
    movie_list = [str(i) for i in list(p.glob(movie_dir_path))]
    # for i in movie_list:
    file_path = './movie_test/test_move5.mp4'
    print(file_path)
    json_file_path = "only_blink_"+file_path+"conc.json"
    json_file_path2 = "only_blink_"+file_path + "threshold.json"

    # json_dir_path = './json_file/blink_data_/nedati/'

    # 動画の閾値を得る
    right_threshold, left_threshold = eye_open(file_path)
    data2 = {
        'right_threshold':right_threshold,
        'left_threshold': left_threshold
    }
    with open(json_file_path2, 'w')as f:
        json.dump(data2, f, indent=4)
    # 動画の処理をするmain関数
    all_5_blink_list= cv_main(file_path, right_threshold, left_threshold)

    # # print(all_change_list)
    # load_data = open("movie_test/test_move4.mp4cv.json", 'r')
    # load_data = json.load(load_data)
    # print(load_data)
    # # 動画の処理をするmain関数
    # all_5_blink_list = load_data['blink_5']
    # all_5_change_list = load_data['face_5']
    # all_5_angle_list = load_data['angle_5']


    c1 = section_concentration(all_5_blink_list, 1.9, 1.3)
    # w = 1 - ((r/tr + p/tp + y/ty) / 3))を返す

    print("c1:", c1)
    data = {
        'c1': c1,
    }

    # p = pathlib.Path(json_dir_path).mkdir(parents=True, exist_ok=True)
    with open(json_file_path, 'w')as f:
        json.dump(data, f, indent=4)
