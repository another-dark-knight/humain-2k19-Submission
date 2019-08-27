import darknet
import math
import os
import cv2
import numpy as np


def get_box(x, y, w, h):

    left = int(round(x - (w / 2)))
    right = int(round(x + (w / 2)))
    up = int(round(y - (h / 2)))
    down = int(round(y + (h / 2)))

    return ((left, up), (right, down))

def draw_boxes(detections, img):

    preds = {}
    for i, detection in enumerate(detections):

        x, y, w, h = detection[2][0], detection[2][1], \
                     detection[2][2], detection[2][3]

        box = get_box(x, y, w, h)

        up_left, bottom_right = box
        preds[i+1]={'box': box,
                    'label':detection[0],
                    'confidence': detection[1]}
        cv2.rectangle(img, up_left, bottom_right, (0,255,0), 5)

    return img, preds

def get_model(config, weights, data):

    if not os.path.exists(config):
        raise ValueError("Invalid config path")
    if not os.path.exists(weights):
        raise ValueError("Invalid weights path")
    if not os.path.exists(data):
        raise ValueError("Invalid data path")

    net = darknet.load_net_custom(config.encode("ascii"),
                                  weights.encode("ascii"), 0, 1)
    meta = darknet.load_meta(data.encode("ascii"))
    altNames=None
    try:
        with open(data) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass

    return net, meta, altNames

def predict(config, weights, data, img_paths):

    net, meta, altNames = get_model(config, weights, data)
    os.makedirs(os.path.join('predictions','plates'), exist_ok=True)
    for index, img_path in enumerate(img_paths):
        print(img_path)
        detections = darknet.detect( net, meta, img_path.encode("ascii"))
        img = cv2.imread(img_path)
        img, preds = draw_boxes(detections, img)
        picname = img_path.split('/')[-1]
        print("Picname-"+picname)
        cv2.imwrite(os.path.join('predictions',picname), img)
        print(img_path)
        for i, (k,v) in enumerate(preds.items()):
            print(k)
            print(v)
            print('')
            box = v['box']
            img_crop = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            cv2.imwrite(os.path.join('predictions','plates',picname[:-4]+"_1"+".png"), img_crop)

if __name__=='__main__':
    imdir = os.path.join('data','obj')
    imgs = [os.path.join(imdir,x) for x in os.listdir(imdir) if x[-3:]=='png' or x[-3:]=='jpg']
    imgs.sort()
    predict('cfg/yolo-voc.2.0.cfg', './backup/yolo-voc_final.weights',
            './cfg/obj.data', imgs)

