from imutils.video import VideoStream
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
from scipy.spatial import distance

right_t_provisional = 0.210
left_t_provisional = 0.210
right_t = right_t_provisional - 0.05
left_t = left_t_provisional - 0.05
EYE_AR_THRESH = (right_t_provisional + left_t_provisional) / 2
EYE_AR_CONSEC_FRAMES = 1
COUNTER = 0
TOTAL = 0
section_list = []
section_cnt = 0
frame_cnt = 0
cnt_looking_away = 0
list_looking_away = []
change_list = []
section_change_list = []

old_points = None
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('classification_tool/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('classification_tool/haarcascade_frontalface_alt2.xml')
# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
#cap = cv2.VideoCapture('movie/blink_data_/nedati/omosiro.mp4')
cap = cv2.VideoCapture(0)

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)


def eye_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        #cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


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


    image_points = None

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
        # 顔の上下の向き　156以下170以上で判定
        pitch = eulerAngles[0]
        # 顔の回転 10で判定
        roll = eulerAngles[2]

        if abs(yaw) >= 17 or pitch >= 180 or pitch<=156 or abs(roll)>=10:
            cnt_looking_away += 1
            cv2.putText(frame, "away", (10, 195), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)


        t = (right_eye_ear + left_eye_ear) / 2.0
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

        cv2.putText(frame, 'yaw' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'pitch' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'roll' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.putText(frame, 'blink_cnt' + str(int(section_cnt)), (20, 65), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                  translation_vector, camera_matrix, dist_coeffs)
        frame_change_list = []
        for p in range(len(image_points)):

            if type(old_points) == type(image_points):
                frame_change_list.append([int(image_points[p][0]) - int(old_points[p][0]), int(image_points[p][1]) - int(old_points[p][1])])
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

change_minute = []
for i in change_list:
    x_all = 0
    y_all = 0
    for j in i:
        for k in j:
            x_all += abs(k[0])
            y_all += abs(k[1])
    change_minute.append([x_all,y_all])
print(change_minute)
print(list_looking_away)

cap.release()
cv2.destroyAllWindows()
