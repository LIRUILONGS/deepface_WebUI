
from retinaface import RetinaFace  # this is not a must dependency
from retinaface.commons import postprocess
import  cv2
import numpy as np
import math


def build(img):
    face_detector = RetinaFace.build_model()

    obj = RetinaFace.detect_faces(img, model=face_detector, threshold=0.9)
    for i in obj:
        print(obj[i])
        face = obj[i]
        x1, y1, x2, y2 = face['facial_area']
        landmarks = face['landmarks']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        nose = landmarks['nose']
        mouth_left = landmarks['mouth_left']
        mouth_right = landmarks['mouth_right']
        print(left_eye[0],left_eye[1])
        cv2.circle(img, (int(left_eye[0]),int(left_eye[1])), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(right_eye[0]),int(right_eye[1])), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(nose[0]),int(nose[1])), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(mouth_left[0]),int(mouth_left[1])), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(mouth_right[0]),int(mouth_right[1])), 2, (0, 255, 0), -1)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = -1 * math.atan2(dy, dx) * 180 / math.pi
        angle = cv2.minAreaRect(np.array([left_eye, right_eye, nose]))[-1]

        # 判断人脸是否为正面朝向
        print(f"计算后的角度为：{angle}")
        if abs(angle) < 5:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print('This face is facing forward.')
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            print('This face is not facing forward.')


        ## 判断人脸是否为正面朝向，并在人脸框上标注
        #if is_frontal_face(landmarks[i]):
        #    cv2.putText(img, 'Frontal', (int(face[0]), int(face[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
        #                2)
        #else:
        #    cv2.putText(img, 'Not Frontal', (int(face[0]), int(face[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                (0, 0, 255), 2)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    print(obj)


if __name__ == '__main__':
    # 加载图片
    img_path = '20230528185610943_TIMING.jpg'
    img = cv2.imread(img_path)

    # 转换为RGB格式
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    build(img)
    cv2.imwrite('new_' + img_path, img)


