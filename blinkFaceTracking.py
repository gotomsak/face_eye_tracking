import os, sys
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
import shutil

file_name = "newMovie/goto/goto9"
cap = cv2.VideoCapture(file_name + ".mp4")
#cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
blink_count = 0
close_bool = False
right_t_provisional = 0.240
left_t_provisional = 0.246
right_t = right_t_provisional - 0.1
left_t = left_t_provisional - 0.1
EYE_AR_THRESH = (right_t_provisional + left_t_provisional) / 2
EYE_AR_CONSEC_FRAMES = 1
COUNTER = 0
TOTAL = 0
section_list = []
section_cnt = 0
frame_cnt = 0

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)


# if (right_t_provisional + left_t_provisional) / 2 <= 0.25 and (right_t_provisional + left_t_provisional) / 2 > 0.23:
#     EYE_AR_CONSEC_FRAMES = 2
# elif (right_t_provisional + left_t_provisional) / 2 <= 0.23:
#     EYE_AR_CONSEC_FRAMES = 1
if (right_t_provisional + left_t_provisional) / 2 <= 0.25:
    EYE_AR_CONSEC_FRAMES = 1
else:
    EYE_AR_CONSEC_FRAMES = 3

print(EYE_AR_CONSEC_FRAMES)
print(cap.get(cv2.CAP_PROP_FPS))
print(len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))))

while True:
    frame_cnt+=1

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
        cv2.putText(rgb, "left eye EAR:{} ".format(round(left_eye_ear, 3)),
                    (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        right_eye_ear = calc_ear(face_parts[36:42])
        cv2.putText(rgb, "right eye EAR:{} ".format(round(right_eye_ear, 3)),
                    (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        leftEyeHull = cv2.convexHull(face_parts[42:48])
        rightEyeHull = cv2.convexHull(face_parts[36:42])
        cv2.drawContours(rgb, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(rgb, [rightEyeHull], -1, (0, 255, 0), 1)

        # right_t = 0.262
        # left_t = 0.255
        t = (right_eye_ear + left_eye_ear) / 2.0

        print('face ok')
        if t < EYE_AR_THRESH:
        # 瞬き閾値より現在のearが下回った場合(目を閉じた時)
        #if right_eye_ear < right_t and left_eye_ear < left_t:
            # close_bool = True
            COUNTER += 1
            # print("input",COUNTER)

        # 瞬き閾値より現在のearが上回った場合(目を開けた時)
        else:
            #　目を開けた時、カウンターが一定値以上だったら
            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                TOTAL += 1
                section_cnt += 1
                cv2.putText(rgb, "blink", (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)

            COUNTER = 0
        # if (left_eye_ear + right_eye_ear)/2 < 0.3:
        #     COUNTER += 1
        #     #close_bool = True
        #     if TOTAL == 0:
        #         os.mkdir(file_name)
        #
        #     cv2.imwrite(file_name + '/{}.{}'.format(str(blink_count), 'png'), rgb)
        #
        #     cv2.putText(rgb,"Sleepy eyes. Wake up!",
        #         (10,180), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)
        # else:
        #     if close_bool == True:
        #         blink_count += 1
        #         close_bool = False

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(rgb, "FPS:{} ".format(int(fps)),
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', rgb)

    if cv2.waitKey(1) == 27:
        break  # esc to quit
    print(TOTAL)
    print(frame_cnt)


    if frame_cnt == 1800:
        section_list.append(section_cnt)
        section_cnt = 0
        frame_cnt = 0


section_list.append(section_cnt)
section_cnt = 0
frame_cnt = 0

print(section_list)
print(len(section_list))

cap.release()
cv2.destroyAllWindows()
