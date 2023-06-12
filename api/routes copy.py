from flask import Blueprint, request, Flask, jsonify, request, render_template,send_file
import service
import base64
import numpy as np
import cv2
import imutils
import units
import os
from io import BytesIO



blueprint = Blueprint("routes", __name__)

# 检测字典
img_data = dict()

# 分析字典
img_analyze = dict()

@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"

@blueprint.route("/initdate",methods=["GET"])
def initdate():
    """
    @Time    :   2023/05/29 05:16:21
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   初始化检测的数据
                 Args:
                   
                 Returns:
                   void
    """
    if img_data :
        return jsonify({"status":200,"data": list(img_data.values())})
    else:
        return jsonify({"status":200,"message": "数据不存在"})


@blueprint.route("/initanalyze",methods=["GET"])
def initanalyze():
    """
    @Time    :   2023/05/30 21:43:07
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   初始化解析的数据
                 Args:
                 Returns:
                   void
    """
    
    if img_analyze :
        return jsonify({"status":200,"data": list(img_analyze.values())})
    else:
        return jsonify({"status":200,"message": "数据不存在"})




@blueprint.route("/handle_clear", methods=["GET"])
def handle_clear():
    img_data.clear()
    return jsonify({"status":200,"message": "数据已清理"})



@blueprint.route("/handle_detectors_all", methods=["GET"])
def handle_detectors_all():
    """
    @Time    :   2023/06/02 03:43:34
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   批量人脸检测
                 Args:
                   
                 Returns:
                   void
    """
    
    for img_id in img_data:
        if img_id in img_data and 'img_b64_new' in img_data[img_id] :
            continue
        img_b64 = img_data[img_id]['img_b64']
        imgs = units.get_base64_to_img(img_b64)
        img,res,to_deepfaces = service.extract_faces_all(imgs)
        img_b64 =  units.get_img_to_base64(img)
        img_data[img_id]['img_b64_new'] =img_b64
        img_data[img_id]['res'] =str(res)
        img_data[img_id]['imgs_details'] =to_deepfaces

    return jsonify({"status":200,"message": "检测已完成"})


@blueprint.route("/extract_faces_all", methods=["POST"])
def extract_faces_all():
    """
    @Time    :   2023/05/29 22:12:48
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸检测
                 Args:
                   
                 Returns:
                   void
    """
    
    if request.args is None:
        return jsonify({"status":100,"message": "empty input set passed"})
    img_id =request.args.get("img_id")
    if img_id is None:
        return jsonify({"status":100,"message": "you must pass img_id input"})
    # 如果发生过解析直接返回
    if img_id in img_data and 'img_b64_new' in img_data[img_id] :
        return jsonify({"img_b64":img_data[img_id]['img_b64_new'],"res":img_data[img_id]['res'],"message":"以解析"})
    img_b64 = img_data[img_id]['img_b64']
    imgs = units.get_base64_to_img(img_b64)
    img,res,to_deepfaces = service.extract_faces_all(imgs)
    img_b64 =  units.get_img_to_base64(img)
    img_data[img_id]['img_b64_new'] =img_b64
    img_data[img_id]['res'] =str(res)
    img_data[img_id]['imgs_details'] =to_deepfaces


    return jsonify({"img_b64":img_b64,"res":str(res),"message":"解析成功"})


@blueprint.route("/analyze", methods=["GET"])
def analyze():
    """
    @Time    :   2023/05/30 22:51:05
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸分析
                 Args:
                   
                 Returns:
                   void
    """
    

    if request.args is None:
        return {"message": "empty input set passed"}

    img_id = request.args.get("img_id")
    if img_id is None:
        return {"message": "you must pass img_id input"}
    detector_backend = request.args.get("detector_backend", "retinaface")
    enforce_detection = request.args.get("enforce_detection", False)
    align = request.args.get("align", True)
    actions = request.args.get("actions", ["age", "gender", "emotion", "race"])
    if img_id in img_analyze and 'img_b64' in img_analyze[img_id] :
        img_b64 = img_analyze[img_id]['img_b64']
        imgs = units.get_base64_to_img(img_b64)
        demographies,img = service.analyze(
            image=imgs,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        
        img_b64 =  units.get_img_to_base64(img)
        img_analyze[img_id]['analyze'] = demographies
        img_analyze[img_id]['img_b64_new'] = img_b64

        return  jsonify({"status":200,"data": demographies,"message":"分析成功"})

    else:
        return  jsonify({"status":100,"message":"分析数据不存在" })





@blueprint.route('/download')
def download():
    """
    @Time    :   2023/05/30 03:55:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   下载标记后的图片
                 Args:
                   
                 Returns:
                   void
    """
    
    if request.args is None:
        return jsonify({"status":100,"message": "empty input set passed"})
    img_id =request.args.get("img_id")
    if img_id is None:
        return jsonify({"status":100,"message": "you must pass img_id input"})
    if img_id in img_data  and 'img_b64_new' in img_data[img_id] :
        img_b64 = img_data[img_id]['img_b64_new']
        image_data = base64.b64decode(img_b64)        # 将图片发送给客户端
        file_like = BytesIO(image_data)
        return send_file(file_like, mimetype='image/jpeg', attachment_filename='image.jpg', as_attachment=True)
    else:
       return jsonify({"status":100,"message": "下载失败"})


@blueprint.route('/handle_download_slicing')
def handle_download_slicing():
    """
    @Time    :   2023/05/30 03:55:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   下载人脸切片图片
                 Args:
                 Returns:
                   void
    """
    
    if request.args is None:
        return jsonify({"status":100,"message": "empty input set passed"})
    img_id =request.args.get("img_id")
    if img_id is None:
        return jsonify({"status":100,"message": "you must pass img_id input"})
    if img_id in img_data and 'imgs_details' in img_data[img_id] :
        imgs_details = img_data[img_id]['imgs_details']
        img_b64s =  [(i['img_b64'],i['confidence'])  for i in imgs_details]
        return units.get_b64s_and_make_to_zip(img_b64s,img_id)
    else:
       return jsonify({"status":100,"message": "下载失败"})


@blueprint.route("/upload_file", methods=["POST"])
def upload_file():
    """
    @Time    :   2023/05/29 02:21:05
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   检测上传图片
                 Args:
                   file
                 Returns:
                   void
    """

    req = None
    print("上传图片，把图片转化为 base64 和 Numpy 数组")
    if request.method == 'POST':
        try:
            f = request.files['file']
            print("上传的文件名:===", f.filename)
            file_content = f.read()
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
            nparr = np.fromstring(base64.b64decode(file_content_b64), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_id = units.get_uuid()
            req = {'img_name': f.filename, 'img_b64': file_content_b64, 'img_nparr': str(img),'id': img_id}
            img_data[img_id] =req
        except Exception as e:
            print("图片上传失败", e)

    return jsonify(req)


@blueprint.route("/upload_file/analyze", methods=["POST"])
def upload_file_analyze():
    """
    @Time    :   2023/05/30 21:32:48
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸剖析上传
                 Args:
                   
                 Returns:
                   void
    """
    req = None
    print("上传图片，把图片转化为 base64 和 Numpy 数组")
    if request.method == 'POST':
        try:
            f = request.files['file']
            print("上传的文件名:===", f.filename)
            file_content = f.read()
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
            nparr = np.fromstring(base64.b64decode(file_content_b64), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_id = units.get_uuid()
            req = {'img_name': f.filename, 'img_b64': file_content_b64, 'img_nparr': str(img),'id': img_id}
            img_analyze[img_id] =req
            print(f"上传图片成功{f.filename}")
        except Exception as e:
            print("图片上传失败", e)

    return jsonify(req)



@blueprint.route("/upload_file_url", methods=["GET"])
def upload_file_url():
    """
    @Time    :   2023/05/29 21:47:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   检测：通过图片url 上传图片
                 Args:
                   
                 Returns:
                   void
    """
    
    if request.args is None:
        return jsonify({"message": "empty input set passed"})
    img_url =request.args.get("img_url")
    if img_url is None:
        return jsonify({"message": "you must pass img_url input"})
    if units.is_valid_url(img_url):
        img = imutils.url_to_image(img_url)
        file_content_b64 =  units.get_img_url_base64(img_url)
        img_id = units.get_uuid()
        img_name = os.path.basename(img_url)
        if len(img_name) > 20:
            img_name=  img_name[0:20]
        req = {'img_name': img_name, 'img_b64': file_content_b64, 'img_nparr': str(img),'id': img_id}
        img_data[img_id] = req
    else:
       return jsonify({"status":100,"message": "非法的URL"})

    return jsonify({"status":200,"message": "图片上传成功"})


@blueprint.route("/upload_file_url/analyze", methods=["GET"])
def upload_file_url_analyze():
    """
    @Time    :   2023/05/29 21:47:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   分析：通过图片url 上传图片
                 Args:
                   
                 Returns:
                   void
    """
    
    if request.args is None:
        return jsonify({"message": "empty input set passed"})
    img_url =request.args.get("img_url")
    if img_url is None:
        return jsonify({"message": "you must pass img_url input"})
    if units.is_valid_url(img_url):
        img = imutils.url_to_image(img_url)
        img = imutils.resize(img,width=500)
        
        file_content_b64 =  units.get_img_to_base64(img)
        img_id = units.get_uuid()
        img_name = os.path.basename(img_url)
        if len(img_name) > 20:
            img_name=  img_name[0:20]
        req = {'img_name': img_name, 'img_b64': file_content_b64, 'img_nparr': str(img),'id': img_id}
        img_analyze[img_id] = req
    else:
       return jsonify({"status":100,"message": "非法的URL"})

    return jsonify({"status":200,"message": "图片上传成功"})


@blueprint.route("/handle_del", methods=["GET"])
def handle_del():
    if request.args is None:
        return {"status":100,"message": "empty input set passed"}

    img_id = request.args.get("img_id")
    if img_id is None:
        return {"status":100,"message": "you must pass img_id input"}
    del img_data[img_id]
    return jsonify({"status":200,"message": "删除成功"})

@blueprint.route("/handle_del_analyze", methods=["GET"])
def handle_del_analyze():
    if request.args is None:
        return {"status":100,"message": "empty input set passed"}

    img_id = request.args.get("img_id")
    if img_id is None:
        return {"status":100,"message": "you must pass img_id input"}
    del img_analyze[img_id]
    return jsonify({"status":200,"message": "删除成功"})

@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_url = input_args.get("img_url")
    if img_url is None:
        return {"message": "you must pass img_url input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

    obj = service.represent(
        img_path=img_url,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1_path")
    img2_path = input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    verification["verified"] = str(verification["verified"])

    return verification


