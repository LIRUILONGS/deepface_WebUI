#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   deepface_find_demo.py
@Time    :   2023/05/21 02:56:15
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   verify 方法Demo 
"""

# here put the import lib
# pip install deepface==0.0.79



from deepface import DeepFace
import os 
import cv2

def vef(m1,m2):
    dfs = DeepFace.verify(
        img1_path=m1,
        img2_path =m2,
        model_name="SFace",
        detector_backend="mtcnn",
        enforce_detection=False)
    return dfs['verified'],dfs['distance']
        

def vefify(path):
    image_paths = None  
    from imutils import paths
    image_paths = list(paths.list_images(path))
    size =  len(image_paths) 
    dict_img =  {i:0 for i in range(size)}
    import combination_algorithm
    combinations =  combination_algorithm.build_(size)
    print(combinations)
    for f_idx in combinations:
        a,b =  f_idx
        print("💟💌💢💥💥💥",a,b)
        img_a = image_paths[a - 1]
        img_b = image_paths[b - 1]
        m1 = cv2.imread(img_a)
        m2 = cv2.imread(img_b)
        print("比较的数据：",img_a,img_b)
        print("对应的路径索引：",a-1,b-1)
        new_dir ="W:\\python_code\\deepface\\temp\\cf_s\\" + os.path.basename(img_a)
        if not os.path.exists(new_dir):
              os.makedirs(new_dir)
              file_name =  os.path.basename(img_a)
              cv2.imwrite(new_dir+"\\"+"_bz_"+file_name, m1)
        verified, distance = vef(m1,m2)  
        if verified:   
          print("🚜🚜🚜🚜🚜🚜🚜🚜🚜🚜  保存：",img_b)     
          if distance <= 0.35:
            print(f"存在相同的💥💥💥💥💥💝💝💝💝💝💝💝💟💌💢💥💥💥💥💥💥💥💤💦💨🕳️💥💥💥{img_a}，{img_b}")
            file_name = os.path.basename(img_b)
            cv2.imwrite(new_dir+"\\"+str(distance)+".jpg", m2)
            print("🚜🚜🚜🚜🚜🚜🚜🚜🚜🚜  保存：",img_b)     
    return 0          
        

if __name__ == '__main__':
    """
    @Time    :   2023/06/13 00:34:17
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸验证:用于验证图像对是否为同一人或不同人，验证函数将面部图像表示为向量，然后计算这些向量之间的相似度。
                  同一人图像的向量应具有更高的相似度（或更小的距离）比不同人的向量。

                 Args:
                   + img1_path (str): 第一张图像的路径
                   + img2_path (str): 第二张图像的路径
                   + model_name (str): 要使用的人脸识别模型的名称（默认为“VGG-Face”）
                   + detector_backend (str): 要使用的人脸检测后端（默认为“opencv”）
                   + distance_metric (str): 用于比较面部嵌入的距离度量（默认为“cosine”）
                   + enforce_detection (bool): 是否在图像中未检测到人脸时引发异常（默认为True）
                   + align (bool): 是否在生成嵌入之前执行面部对齐（默认为True）
                   + normalization (str): 用于预处理图像的归一化技术（默认为“base”）
                 Returns:
                   + verified(核实):True
                   + distance(距离):0.4439834803806296
                   + threshold(阈值):0.593
                   + model:SFace
                   + detector_backend:mtcnn
                   + similarity_metric(相似性指标):cosine
                   + facial_areas(人脸位置):{'img1': {'x': 0, 'y': 0, 'w': 200, 'h': 255}, 'img2': {'x': 2, 'y': 13, 'w': 194, 'h': 231}}
                   + time:1.95
                   void
    """
    
    #dfs = DeepFace.verify(
    #    img1_path="W:\\python_code\\deepface\\temp\\cf\\cf_6dd3a0638cf4f4006aa1f455cac65577d.jpg.png",
    #    img2_path ="W:\\python_code\\deepface\\temp\\cf\\cf_8a96787835b5d4677a56ad6db0c610958.jpg.png",
    #    model_name="SFace",
    #    detector_backend="mtcnn",
    #    enforce_detection=False)
    
    #for k in dfs:
    #    print(k+':'+ str(dfs[k]))
    vefify("W:\\python_code\\deepface\\temp\\cf\\") 
    print("😍🛼🤣😂🚆🦽😊😊🚛")   


        
           
            
   