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



# 返り値: 各1分おきのlistdata(瞬き, 顔の変化量, よそ見したときのフレーム数)
# 引数: ファイルパス
def cv_main(video_path, right_t_provisional, left_t_provisional):


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
            rect_frag = True
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # for (x, y) in shape:
            #     cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                     tuple(shape[48]), tuple(shape[54])])
            for (x, y) in image_points:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            if len(faces) == 1:
                x, y, w, h = faces[0, :]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


                face = dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])
                face_parts = predictor(frame, face)
                face_parts = face_utils.shape_to_np(face_parts)





        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()

    return all_blink_list, all_change_list, all_angle_list


if __name__ == '__main__':
    p = pathlib.Path('.')
    movie_dir_path = './movie/face_eye_data/*/*.mp4'
    movie_list = [str(i) for i in list(p.glob(movie_dir_path))]
    # for i in movie_list:
    file_path = './movie_test/testgoto.mp4'
    print(file_path)
    json_file_path = file_path+"conc.json"
    # json_dir_path = './json_file/blink_data_/nedati/'
    right_threshold = 0.2
    left_threshold = 0.2
    # 動画の閾値を得る
    # right_threshold, left_threshold = eye_open(file_path)

    # 動画の処理をするmain関数
    all_blink_list, all_change_list, all_angle_list = cv_main(0, right_threshold, left_threshold)
