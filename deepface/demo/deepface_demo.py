#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   deepface_demo.py
@Time    :   2023/05/07 21:17:29
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   None
"""

# here put the import lib
# pip install deepface==0.0.79


from deepface import DeepFace

import yaml_util

config = yaml_util.get_yaml_config()

#  视频监控相关配置
video_surveillance = config['video_surveillance']

# deepface 相关配置
deepface = config['deepface'][0]

# 静态文件位置,人脸库配置
data = config['data']


# 教室人脸识别
# verification = DeepFace.verify(img1_path = "hg1.png", img2_path = "hg2.png",model_name=models[6],detector_backend = detectors[4] )



# 测试可用的
# ArcFace，VGG-Face，Facenet，Facenet512

# 视频流识别

def camera():
    """
    @Time    :   2023/05/21 02:33:48
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   None
                 Args:
                    - db_path：人脸数据库路径。您应该在此文件夹中存储一些.jpg文件。
                    - model_name：人脸识别模型的名称，例如VGG-Face、Facenet、Facenet512、OpenFace、DeepFace、DeepID、Dlib、ArcFace、SFace。
                    - detector_backend：人脸检测后端可以是retinaface、mtcnn、opencv、ssd或dlib。
                    - distance_metric：距离度量方法可以是cosine、euclidean或euclideanl2。
                    - enable_face_analysis：将其设置为False以仅运行人脸识别。
                    - source：将其设置为0以访问网络摄像头。否则，传递确切的视频路径。
                    - time_threshold：分析的图像将显示多少秒。
                    - frame_threshold：需要多少帧才能聚焦于人脸。
                 Returns:
                   void
    """
    DeepFace.stream(db_path=data['db_path'], model_name=deepface['model'], detector_backend=deepface['detector'],
                    source=video_surveillance['rtsp_url'], time_threshold=10, frame_threshold=2)


def find():
    """
    @Time    :   2023/05/21 02:35:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   DeepFace.find方法是DeepFace库中的一个方法，用于在人脸数据库中查找与给定人脸最相似的人脸。
                 Args:
                    - img_path：要查找的人脸图像路径、numpy数组（BGR）或base64编码的图像。
                    - db_path：人脸数据库路径。您应该在此文件夹中存储一些.jpg文件。
                    - model_name：人脸识别模型的名称，例如VGG-Face、Facenet、Facenet512、OpenFace、DeepFace、DeepID、Dlib、ArcFace、SFace。
                    - distance_metric：距离度量方法可以是cosine、euclidean或euclideanl2。
                    - enforce_detection 参数是一个布尔值，指定如果无法检测到人脸，则该函数是否应引发异常。如果不想得到异常并仍要运行该函数，则将其设置为False。这对于低分辨率图像可能很方便。
                    - detector_backend 参数指定要使用的人脸检测器后端，可以是opencv、retinaface、mtcnn、ssd、dlib或mediapipe。
                    - align 参数是一个布尔值，指定是否应对人脸进行对齐。
                    - normalization参数指定要使用的归一化方法，可以是base、l2或tanh。这些方法在不同的情况下可能会产生不同的效果，例如，
                        + 如果您的数据集中存在大量噪声或异常值，那么使用tanh方法可能会更好，因为它可以将数据缩放到-1到1之间，从而更好地处理异常值。
                        + 如果您的数据集中没有太多噪声或异常值，那么使用l2方法可能会更好，因为它可以将数据缩放到单位圆上，从而更好地进行比较和匹配。
                    - silent参数是一个布尔值，用于禁用一些日志记录和进度条。
                 Returns:
                   void
    """

    dfs = DeepFace.find(
        img_path="W:\python_code\deepface\\temp\\new_44fb0993cb14346b7b3252d3a6642e2bf.jpg",
        db_path="W:\python_code\db",
        model_name="ArcFace",
        distance_metric="cosine",
        enforce_detection=False,
        detector_backend="retinaface",
        align=True,
        normalization="base",
        silent=False,
    )
    img_paths = dfs[0]._values
    for img in img_paths:
        print(" 查询到的文件信息：", img)


def verify():
    dfs = DeepFace.verify("database\liubiao\liubiao1.jpg", "database\liruilong\liruilong2.png",
                          model_name=deepface['model'],
                          detector_backend=deepface['detector'])
    print(dfs)


def img_analyze():
    dfs = DeepFace.analyze("database\liubiao\yz_W_4.jpg.jpg",detector_backend=deepface['detector'])
    
    for i in dfs:
        print(i['dominant_emotion'])
        print(i['dominant_gender'])
        print(i['dominant_race'])
        print(i['age'])
        




# camera()
# find()

# dfs = DeepFace.represent(img_path="W:\python_code\deepface\database\zhangru.png")



#print(dfs)

find()

# find()
