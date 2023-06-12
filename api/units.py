
from urllib.parse import urlparse
import uuid
import requests
import base64
import numpy as np
import cv2 
from io import BytesIO
import zipfile

from flask import send_file




def is_valid_url(url):
    """
    @Time    :   2023/05/29 21:49:19
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   url  字符串 校验
                 Args:
                   url
                 Returns:
                   booler
    """
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_uuid():
    """
    @Time    :   2023/05/29 21:50:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成 UUID
                 Args:
                   
                 Returns:
                   string
    """
    
    return str(uuid.uuid4()).replace('-', '')


def get_img_url_base64(url):
    """
    @Time    :   2023/05/29 21:50:42
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   图片 url 解析为 base64 编码
                 Args:
                   url
                 Returns:
                   base64_bytes
    """
    response = requests.get(url)
    image_bytes = response.content
    base64_bytes = base64.b64encode(image_bytes)
    return base64_bytes.decode('utf-8')

def get_base64_to_img(base64_str):
    """
    @Time    :   2023/05/29 21:51:23
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   base64 编码转化为 opencv img 对象
                 Args:
                   
                 Returns:
                   void
    """
    
    # 从 base64 编码的字符串中解码图像数据
    img_data = base64.b64decode(base64_str)
    # 将图像数据转换为 NumPy 数组
    nparr = np.frombuffer(img_data, np.uint8)
    # 解码图像数组
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_img_to_base64(img):
    """
    @Time    :   2023/05/29 21:54:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   opencv img 对象转化为 base64 编码
                 Args:
                   
                 Returns:
                   void
    """
    img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

    return img_b64

def get_b64s_and_make_to_zip(b64s_mark,img_id):
    # 创建一个名为 'images.zip' 的 zip 文件
    with zipfile.ZipFile(img_id+'_images.zip', 'w') as zip_file:
    # 遍历字典中的每个图像
        for img_b64,img_name in b64s_mark:
            # 将 base64 编码的数据解码为二进制数据
            img_data_binary = base64.b64decode(img_b64)
            # 将图像数据写入 zip 文件中
            zip_file.writestr(img_name+"_" +get_uuid()+ '.jpg', img_data_binary)

    return send_file("..\\"+img_id+'_images.zip', as_attachment=True)


def  build_img_text_marge(img_,text):
    """
    @Time    :   2023/06/01 05:29:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成文字图片拼接到 img 对象
                 Args:
                   
                 Returns:
                   void
    """
    # 创建一个空白的图片
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # 设置字体和字号
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2

    # 在图片上写入文字
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=2)
    text_x = (500 - text_size[0]) // 2
    text_y = (500 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=2)
    montage_size = (300, 400)
    montages = cv2.build_montages([img_,img], montage_size, (1, 2))



    # 保存图片
    return montages

    
