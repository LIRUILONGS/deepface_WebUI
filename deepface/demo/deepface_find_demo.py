#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   deepface_find_demo.py
@Time    :   2023/05/21 02:56:15
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   find 方法Demo 
"""

# here put the import lib
# pip install deepface==0.0.79



from deepface import DeepFace



if __name__ == '__main__':
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
                        - normalization 参数指定要使用的归一化方法，可以是base VGGFace Facenet raw 等
                        - silent参数是一个布尔值，用于禁用一些日志记录和进度条。
                     Returns:
                       void
        """

    dfs = DeepFace.find(
        img_path="huge_1.jpg",
        db_path="W:\python_code\db",
        model_name="DeepID",
        distance_metric="cosine",
        enforce_detection=True,
        detector_backend="retinaface",
        align=False,
        normalization="ArcFace",
        silent=False,
    )
    print(dfs)
    img_paths = dfs[0]._values
    #print(img_paths)
    for img in img_paths:
        print(" 查询到的文件信息：", img)