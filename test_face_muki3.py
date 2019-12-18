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
    blink_count = 0
    close_bool = False
    # right_t = 0.262
    # left_t = 0.255
    # EYE_AR_THRESH = (right_t+left_t)/2
    # EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0
    TOTAL = 0
    right_eye_list = []
    left_eye_list = []
    frame_cnt = 0

    def calc_ear(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

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
        # cv2.putText(rgb, "FPS:{} ".format(int(fps)),
        #     (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # cv2.imshow('frame', rgb)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    right_ave = sum(i for i in right_eye_list) / len(right_eye_list)
    left_ave = sum(i for i in left_eye_list) / len(left_eye_list)

    right_s = math.sqrt((1 / len(right_eye_list) * pow(sum(i - right_ave for i in right_eye_list), 2)))
    left_s = math.sqrt((1 / len(left_eye_list) * pow(sum(i - left_ave for i in left_eye_list), 2)))
    right_threshold = right_ave - right_s
    left_threshold = left_ave - left_s
    print(right_threshold)
    print(left_threshold)

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
    blink_cnt = 0

    # すべてのフレームのカウント
    all_frame_cnt = 0
    section_frame_cnt = 0

    # 顔の変化のリスト
    change_list = []

    # 5秒おきの瞬き回数のリスト
    section_blink_list = []

    # 5秒おきの顔の変化リスト
    section_change_list = []

    # 5秒おきの角度
    section_angle_list = []

    # 5秒おきのすべての角度
    all_angle_list = []

    # 5秒おきのすべての瞬き
    all_blink_list = []

    # 5秒おきのすべての顔の変化
    all_change_list = []

    angle_threshold_up = 150
    angle_threshold_down = 175
    angle_threshold_pitch = 12.5
    angle_threshold_yaw = 20
    angle_threshold_roll = 15

    # 1フレーム目の回転角
    fast_yaw = 0
    fast_pitch = 0
    fast_roll = 0

    # 1フレーム前の顔の位置のポイント
    old_points = None
    print("[INFO] loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('classification_tool/shape_predictor_68_face_landmarks.dat')
    face_cascade = cv2.CascadeClassifier('classification_tool/haarcascade_frontalface_alt2.xml')
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(video_path)

    while True:

        all_frame_cnt += 1
        section_frame_cnt += 1

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

            image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                     tuple(shape[48]), tuple(shape[54])])

            for (x, y) in image_points:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                     tuple(shape[48]), tuple(shape[54])], dtype='double')

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
                print(left_eye_ear)

                right_eye = face_parts[36:42]
                eye_marker(face_gray_resized, right_eye)

                right_eye_ear = calc_ear(right_eye)
                print(right_eye_ear)
                # t = (right_eye_ear + left_eye_ear) / 2.0
                # if t < EYE_AR_THRESH:
                if right_eye_ear < right_t and left_eye_ear < left_t:
                    # 瞬き閾値より現在のearが下回った場合(目を閉じた時)
                    # if right_eye_ear < right_t and left_eye_ear < left_t:
                    # close_bool = True
                    COUNTER += 1
                # 瞬き閾値より現在のearが上回った場合(目を開けた時)
                else:
                    # 　目を開けた時、カウンターが一定値以上だったら
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        blink_cnt += 1
                        cv2.putText(frame, "blink", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

                    COUNTER = 0
                #cv2.imshow("gray", face_gray_resized)

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])

            size = frame.shape

            focal_length = size[1]
            center = (size[1] // 2, size[0] // 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype='double')

            dist_coeffs = np.zeros((4, 1))

            # 物体の姿勢を求める
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            # 回転行列と回転ベクトルを相互に変換
            (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
            mat = np.hstack((rotation_matrix, translation_vector))

            # homogeneous transformation matrix (projection matrix)　射影行列を，回転行列とカメラ行列に分解します．
            (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)  # 回転を表す3つのオイラー角．
            # 顔の横の向き17で判定
            yaw = float(eulerAngles[1])
            print('yaw', yaw)
            # 顔の上下の向き　156以下170以上で判定
            pitch = float(eulerAngles[0])
            print('pitch', pitch)
            # 顔の回転 10で判定
            roll = float(eulerAngles[2])
            print('roll', roll)

            # angleのリストを作成
            if all_frame_cnt == 1:
                fast_yaw = abs(yaw)
                fast_pitch = abs(pitch)
                fast_roll = abs(roll)
                # section_angle_list.append([0, 0, 0])
            elif abs(yaw) < angle_threshold_yaw and abs(pitch) > angle_threshold_up and abs(
                    pitch) < angle_threshold_down and abs(roll) < angle_threshold_roll:
                section_angle_list.append(
                    [abs(yaw - fast_yaw), abs(pitch - (fast_pitch + angle_threshold_pitch)), abs(roll - fast_roll)])
            else:
                section_angle_list.append([angle_threshold_yaw, angle_threshold_pitch, angle_threshold_roll])

            cv2.putText(frame, 'yaw' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'pitch' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'roll' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            cv2.putText(frame, 'blink_cnt' + str(int(blink_cnt)), (20, 65), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                      translation_vector, camera_matrix, dist_coeffs)

            # 1フレーム前と今のlistの差分をframe_change_listに入れた
            #frame_change_list = []
            for p in range(len(image_points)):
                if type(old_points) == type(image_points):
                    section_change_list.append(
                        [int(image_points[p][0]) - int(old_points[p][0]),
                         int(image_points[p][1]) - int(old_points[p][1])]
                    )

                cv2.circle(frame, (int(image_points[p][0]), int(image_points[p][1])), 3, (0, 0, 255), -1)

            old_points = image_points
            # 差分のリスト
            # section_change_list.append(frame_change_list)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # show the frame
        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # 5秒おきに
        if section_frame_cnt == 150:
            # section_blink_list.append(blink_cnt)

            all_change_list.append(section_change_list)

            all_angle_list.append(section_angle_list)
            print(section_angle_list)

            all_blink_list.append(blink_cnt)
            section_change_list = []

            # section_blink_list = []
            section_angle_list = []
            blink_cnt = 0
            section_frame_cnt = 0

    all_blink_list.append(blink_cnt)
    section_cnt = 0
    frame_cnt = 0
    all_change_list.append(section_change_list)
    all_angle_list.append(section_angle_list)

    # print(change_list)
    json_file_path = video_path+"cv.json"
    # 5秒おきのx,yの変化量をすべて足してまとめた
    data = {
        'blink': all_blink_list,
        'face': all_change_list,
        'angle': all_angle_list
    }
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    cap.release()
    cv2.destroyAllWindows()

    return all_blink_list, all_change_list, all_angle_list


if __name__ == '__main__':


    file_path = './movie/face_eye_data/inagawa/movie2.mp4'
    json_file_path = file_path+"conc.json"
    # json_dir_path = './json_file/blink_data_/nedati/'
    right_threshold = 0.2
    left_threshold = 0.2
    # 動画の閾値を得る
    right_threshold, left_threshold = eye_open(file_path)

    # 動画の処理をするmain関数
    all_blink_list, all_change_list, all_angle_list = cv_main(file_path, right_threshold, left_threshold)
    change_5_second = change_5_second(all_change_list)

    blink_frequency = section_frequency(all_blink_list)
    change_frequency = section_frequency(change_5_second)

    c1 = section_concentration(blink_frequency)
    c2 = section_concentration(change_frequency)
    # w = 1 - ((r/tr + p/tp + y/ty) / 3))を返す
    w = create_w_list(all_angle_list)
    # c3 = section_concentration(list_looking_away)

    print("c1:", c1)
    print("c2:", c2)
    print("w:", w)
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
