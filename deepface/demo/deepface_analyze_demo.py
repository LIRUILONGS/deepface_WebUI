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
    @Time    :   2023/05/31 01:44:27
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   analyze 方法是 DeepFace 库中的一个函数，用于分析人脸属性，包括年龄、性别、情绪和种族。
                 在后台，分析函数构建卷积神经网络模型，以对输入图像中的人脸进行年龄、性别、情绪和种族分类。
                 Args:
                   - img_path:图像路径、numpy 数组（BGR）或 base64 编码的图像。如果源图像中有多个人脸，则结果将是出现在图像中的人脸数量大小的列表。
                   - actions: 参数是一个元组，其中默认值为 ('age', 'gender', 'emotion', 'race')，您可以删除其中的一些属性。
                   - enforce_detection :参数默认为 True，如果未检测到人脸，则函数会抛出异常。如果您不想得到异常，则可以将其设置为 False。这对于低分辨率图像可能很方便。
                   - detector_backend: 参数指定要使用的人脸检测器的后端，例如 OpenCV、RetinaFace、MTCNN 等。
                   - align: 参数是一个布尔值，表示是否根据眼睛位置进行对齐。
                   - silent :参数是一个布尔值，表示是否禁用（某些）日志消息。
                 Returns:
                    - "region"：表示人脸在图像中的位置和大小。
                    - "age"：表示人脸的年龄。
                    - "dominant_gender"：表示人脸的主要性别。
                    - "gender"：表示人脸的性别及其置信度。
                    - "dominant_emotion"：表示人脸的主要情绪。
                    - "emotion"：表示人脸的情绪及其置信度。
                    - "dominant_race"：表示人脸的主要种族。
                    - "race"：表示人脸的种族及其置信度。
    """
    dfs = DeepFace.analyze("database\yz_W.jpg",detector_backend="retinaface")
    
    for i in dfs:
        print(i)
        print(i['dominant_emotion'])
        print(i['dominant_gender'])
        print(i['dominant_race'])
        print(i['age'])
        print(i['region'])
   