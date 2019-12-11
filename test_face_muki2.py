from imutils.video import VideoStream
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
from scipy.spatial import distance
import json
import pathlib
#from . import eye_open_check
import math


def eye_open(file_name):

    cap = cv2.VideoCapture(file_name)
    face_cascade = cv2.CascadeClassifier('classification_tool/haarcascade_frontalface_alt2.xml')
    face_parts_detector = dlib.shape_predictor('classification_tool/shape_predictor_68_face_landmarks.dat')
    blink_count=0
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

    right_s = math.sqrt((1/len(right_eye_list)*pow(sum(i-right_ave for i in right_eye_list),2)))
    left_s = math.sqrt((1/len(left_eye_list)*pow(sum(i-left_ave for i in left_eye_list), 2)))
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


def eye_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        # cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


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


def section_concentration_new(c1, c2, c3):
    b_concentration = []
    m_concentration = []
    concentration = []
    for i in range(len(c1)):
        b_concentration.append(c1[i] * c3[i])

    for i in range(len(c2)):
        m_concentration.append(c2[i] * (1 - c3[i]))

    for i in range(len(c1)):
        concentration.append(b_concentration[i] + m_concentration[i])

    return concentration


# 返り値: 各1分おきのlistdata(瞬き, 顔の変化量, よそ見したときのフレーム数)
# 引数: ファイルパス
def cv_main(video_path, right_t_provisional, left_t_provisional):
    # 開眼度閾値
    # right_t_provisional = 0.220
    # left_t_provisional = 0.219
    right_t = right_t_provisional - 0.05
    left_t = left_t_provisional - 0.05
    EYE_AR_THRESH = (right_t_provisional + left_t_provisional) / 2
    EYE_AR_CONSEC_FRAMES = 1
    # 目を閉じたときのカウンター
    COUNTER = 0
    # 目を閉じたときのトータルカウント
    TOTAL = 0
    # 1分おきの瞬き回数のリスト
    section_list = []
    # 1分おきの瞬き回数
    section_cnt = 0
    # すべてのフレームのカウント
    frame_cnt = 0
    # 1分おきのよそ見したフレーム数
    cnt_looking_away = 0
    # 1分おきのよそ見したフレーム数のリスト
    list_looking_away = []
    # 顔の変化のリスト
    change_list = []
    # 1分おきの顔の変化リスト
    section_change_list = []

    # 1フレーム前の顔の位置のポイント
    old_points = None
    print("[INFO] loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('classification_tool/shape_predictor_68_face_landmarks.dat')
    face_cascade = cv2.CascadeClassifier('classification_tool/haarcascade_frontalface_alt2.xml')
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    # vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    cap = cv2.VideoCapture(video_path)

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('movie/test3.mp4')

    while True:
        frame_cnt += 1
        ret, frame = cap.read()

        # frame = imutils.resize(frame, width=500)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
        rects = detector(gray, 0)
        if len(rects) == 0:
            cnt_looking_away += 1
            cv2.putText(frame, "away", (10, 195), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

        # image_points = None

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
                    # print("input",COUNTER)

                # 瞬き閾値より現在のearが上回った場合(目を開けた時)
                else:
                    # 　目を開けた時、カウンターが一定値以上だったら
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        section_cnt += 1
                        cv2.putText(frame, "blink", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

                    COUNTER = 0

            # if len(rects) > 0:
            # # cv2.putText(frame, "detected", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)

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
            yaw = eulerAngles[1]
            print('yaw', yaw)
            # 顔の上下の向き　156以下170以上で判定
            pitch = eulerAngles[0]
            print('pitch',pitch)
            # 顔の回転 10で判定
            roll = eulerAngles[2]
            print('roll',roll)

            if abs(yaw) >= 17 or pitch >= 180 or pitch <= 156 or abs(roll) >= 10:
                cnt_looking_away += 1
                cv2.putText(frame, "away", (10, 195), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

            cv2.putText(frame, 'yaw' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'pitch' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'roll' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            cv2.putText(frame, 'blink_cnt' + str(int(section_cnt)), (20, 65), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                      translation_vector, camera_matrix, dist_coeffs)

            # 1フレーム前と今のlistの差分をframe_change_listに入れた
            frame_change_list = []
            for p in range(len(image_points)):
                if type(old_points) == type(image_points):
                    frame_change_list.append(
                        [int(image_points[p][0]) - int(old_points[p][0]),
                         int(image_points[p][1]) - int(old_points[p][1])]
                    )

                cv2.circle(frame, (int(image_points[p][0]), int(image_points[p][1])), 3, (0, 0, 255), -1)

            old_points = image_points
            section_change_list.append(frame_change_list)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if frame_cnt == 1800:
            section_list.append(section_cnt)
            section_cnt = 0
            frame_cnt = 0

            change_list.append(section_change_list)
            section_change_list = []

            list_looking_away.append(cnt_looking_away)
            cnt_looking_away = 0

    section_list.append(section_cnt)
    section_cnt = 0
    frame_cnt = 0
    change_list.append(section_change_list)
    list_looking_away.append(cnt_looking_away)
    print(section_list)
    # print(change_list)

    # 1分おきのx,yの変化量をすべて足してまとめた
    change_minute = []
    for i in change_list:
        x_all = 0
        y_all = 0
        for j in i:
            for k in j:
                x_all += abs(k[0])
                y_all += abs(k[1])
        change_minute.append([x_all, y_all])
    print(change_minute)

    print(list_looking_away)

    cap.release()
    cv2.destroyAllWindows()

    return section_list, change_minute, list_looking_away, change_list


if __name__ == '__main__':
    file_path = './movie_test/testNew.mp4'
    json_file_path = './movie_test/testNew.json'
    # json_dir_path = './json_file/blink_data_/nedati/'
    right_threshold = 0.22
    left_threshold = 0.22
    # right_threshold, left_threshold = eye_open(file_path)
    section_list, change_minute, list_looking_away, change_list = cv_main(0, right_threshold, left_threshold)
    c1 = section_concentration(section_list)
    c2 = section_concentration(change_minute)
    c3 = section_concentration(list_looking_away)
    print("c1:", c1)
    print("c2:", c2)
    print("c3:", c3)
    C_list = section_concentration_new(c1, c2, c3)
    print("C_List:", C_list)

    C = sum(C_list) / len(C_list)
    print("C:", C)
    data = {
        'blink': section_list,
        'face': change_minute,
        'away': list_looking_away,
        'c1': c1,
        'c2': c2,
        'c3': c3,
        'section_concentration': C_list,
        'concentration': C,
        'face_raw': change_list,
    }

    # p = pathlib.Path(json_dir_path).mkdir(parents=True, exist_ok=True)
    with open(json_file_path, 'w')as f:
        json.dump(data, f, indent=4)

