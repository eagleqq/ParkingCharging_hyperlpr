import os
import cv2
import numpy as np
import datetime
from hyperlpr_py3 import pipline as pp
from PIL import ImageFont,ImageDraw,Image

font = ImageFont.truetype(r'./Font/platech.ttf',80)

def SimpleRecognizePlateWithGui(image):
    images = pp.detect.detectPlateRough(
        image, image.shape[0], top_bottom_padding_rate=0.1)
    res_set = []
    for j, plate in enumerate(images):
        plate, rect, origin_plate = plate
        plate = cv2.resize(plate, (136, 36 * 2))
        plate_color = "蓝"
        plate_type = pp.td.SimplePredict(plate)
        if (plate_type > 0) and (plate_type < 5):
            plate = cv2.bitwise_not(plate)
            plate_color = "黄"
        image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)
        image_rgb = pp.fv.finemappingVertical(image_rgb)
        pp.cache.verticalMappingToFolder(image_rgb)
        e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
        res_set.append([
                        rect,
                        plate_color,
                        e2e_plate,
                        e2e_confidence
                        ])

    return image, res_set

def recognize_and_show_one_image(image):

    # image = cv2.imdecode(image, -1)
    image,res_set = SimpleRecognizePlateWithGui(image)
    image = draw_dialog(image,res_set)
    return image

def draw_dialog(image,res_set):
    if len(res_set) > 0:
        for res in res_set:
            curr_rect = res[0]
            y = int(curr_rect[1])
            x = int(curr_rect[0])
            h = int(curr_rect[1] + curr_rect[3])
            w = int(curr_rect[0] + curr_rect[2])
            image = cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 7)
            img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            draw.text((x,y-100),res[2], font=font, fill=(0,255,0))
            image = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        return image

def video(path):
    vcapture = cv2.VideoCapture(path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    # 保存路径
    path = path.replace('/', '\\')
    dir = path.rsplit('\\', 1)[0] + '\\temp'
    creat_dir(dir)
    file_name = dir+"/plate_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))
    count = 0
    success = True
    while success:
        # print("frame: ", count)
        # yield count
        # Read next image
        success, image = vcapture.read()
        if success:
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            crop_image = image[..., ::-1]
            image = np.array(crop_image)
            # 获取对象结果
            image = recognize_and_show_one_image(image)
            if image is not None:
                splash = image[..., ::-1]

            else:
                splash = crop_image
            # Add image to video writer
            vwriter.write(splash)
            # count += 1
    vwriter.release()
    print("Saved to ", file_name)
    return file_name

def creat_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == '__main__':
    video(r'D:\workspace\EasyPR-python\data\video\test2.mp4')