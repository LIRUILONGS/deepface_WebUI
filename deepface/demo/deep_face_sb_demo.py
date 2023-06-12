import os
import glob
import tensorflow as tf
from deepface import DeepFace
import cv2
import shutil
import time
import subprocess

# 指定输入照片文件夹路径
input_folder_path = '/tmp/photo'
# 已知人的文件夹
know_images_dir_url = './know_images_dir/'
# 未知人的文件夹
unknow_images_dir_url = './unknow_images_dir/'
db_path = "./id_card_images_dir"

f_n_img = "./flag_iamges/"

print(tf.config.list_physical_devices('GPU'))
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace", "Facenet512"]
# 人脸检测器
detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
# 相似性
metrics = ["cosine", "euclidean", "euclidean_l2"]




def init():
    try:
        subprocess.check_output(['rm', '-rf', f_n_img + '*/*'])
    except:
        pass

def exe():
    # 任务锁
    isRun = True
    while isRun:
        # 获取文件夹下所有文件的路径
        file_paths = glob.glob(os.path.join(input_folder_path, '*.jpg'))
        # 对文件路径进行降序排列
        # file_paths = sorted(file_paths, reverse=False)
        if len(file_paths) > 0:
            for file_path in file_paths:
                mark_1, mark_2, mark_3 = 0, 0, 0
                # 取出第一个文件
                out_bytes = subprocess.check_output(['stat', '--format', '%W %Z', file_path])
                out_text = out_bytes.decode('UTF-8')
                out_array = out_text.split()
                if out_array[1] == out_array[0] or time.time() - int(out_array[1]) < 10:
                    continue
                print('有文件开始执行任务:', time.time())
                # 打开图片
                img = cv2.imread(file_path)
                # FTP文件服务器可能正在被摄像头读写，如果没写完就会是空对象就会报错，此时执行下次循环尝试读写非空照片
                if img is None:
                    print('检测出:', file_path, "是空对象，跳出循环")
                    continue
                # mark
                img_r = cv2.imread(file_path)
                faces = DeepFace.extract_faces(img, detector_backend=detectors[4], enforce_detection=False)
                for face_index, face in enumerate(faces):
                    x = int(face['facial_area']['x'])
                    y = int(face['facial_area']['y'])
                    w = int(face['facial_area']['w'])
                    h = int(face['facial_area']['h'])
                    face_filename = os.path.basename(file_path) + "_" + str(face_index) + ".jpg"

                    if face['confidence'] <= 0.99:
                        # mark
                        cv2.rectangle(img_r, (x, y), (x + w, y + h), [255, 0, 255], 2)
                        cv2.putText(img_r, "N_P:" + format(face['confidence'], '0.4f'), (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 0, 255), 2)
                        mark_1 += 1
                        continue
                    # 裁剪坐标为[y0:y1, x0:x1]
                    cropped = img[y:y + h, x:x + w]
                    cv2.imwrite('./headshot_images_dir/' + face_filename, cropped)
                    # 移动文件
                    shutil.move('./headshot_images_dir/' + face_filename,
                                './headshot_images_moved_dir/' + face_filename)

                    results = DeepFace.find(img_path='./headshot_images_moved_dir/' + face_filename,
                                            db_path=db_path,
                                            # actions=["age", "gender", "emotion", "race"],
                                            model_name=models[6],
                                            detector_backend=detectors[4],
                                            distance_metric=metrics[0],
                                            enforce_detection=False,
                                            align=True,
                                            normalization="ArcFace")
                    # 找到人了
                    if len(results[0]) > 0:
                        file_name = os.path.basename(results[0].identity[0])
                        # find true 找到人 mark
                        cv2.rectangle(img_r, (x, y), (x + w, y + h), [0, 255, 0], 2)
                        cv2.putText(img_r, file_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        mark_2 += 1
                    else:
                        print('没找到人:', face_filename, time.time())
                        cv2.rectangle(img_r, (x, y), (x + w, y + h), [0, 165, 255], 2)
                        cv2.putText(img_r, " U_P", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        mark_3 += 1

                    os.remove('./headshot_images_moved_dir/' + face_filename)


                if mark_2 > 0:
                    ip, file_name = os.path.basename(file_path).split('_01_')
                    cv2.imwrite(f_n_img + ip +'/'+ os.path.basename(file_path) , img_r)

                # 任务完成后删除改文件
                os.remove(file_path)
                print(file_path, "@@@@@@@@@@@@@@@@@@@@@@@", mark_1, mark_2, mark_3)

        else:
            # 没有文件，执行下次任务
            print('没有文件了',time.time())
            isRun = True
            #time.sleep(0.5)
            continue


if __name__ == "__main__":
    init()
    exe()