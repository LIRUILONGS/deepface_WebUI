#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   face_check_demo.py
@Time    :   2023/06/06 00:15:30
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   对识别后的照片匹配加权，筛选最终的结果
"""

# here put the import lib

import os
import glob
import tensorflow as tf
from deepface import DeepFace
import cv2
import shutil
import time
import subprocess
import combination_algorithm



if __name__ == "__main__":
    path = "W:\python_code\deepface\\temp\watgih"
    from imutils import paths
    image_paths =  paths.list_images(path)
    size =  len(list(image_paths)) 
    dict_img =  {i+1:0 for i in range(size)}
    print(dict_img)
    combinations =  combination_algorithm.build_(size)
    print(combinations)
    for i in paths.list_images(path):
        print(i)
        cv2.imwrite(i)
        

     