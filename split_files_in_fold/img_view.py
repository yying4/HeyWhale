import cv2
import os
import sys
from xml.dom.minidom import parse


img_path = './test-A-image'
label_path = './test-A-box-New'
roi_path = './test-A-ROI'


def listdir_nohidden(path):
    for f in os.listdir(path):
        if f[-4:] == '.jpg':
            yield f


img_list = list(listdir_nohidden(img_path))
for img in img_list:
    label = parse(label_path + '/' + img[:-4] + '.xml')
    roi = parse(roi_path + '/' + img[:-4] + '.xml')
    ob = label.documentElement.getElementsByTagName('bndbox')
    roi_box = roi.documentElement.getElementsByTagName('roi')
    pic = cv2.imread(img_path + '/' + img)
    height, width = pic.shape[:2]  # 获取原图像的水平方向尺寸和垂直方向尺寸。
    pic = cv2.resize(pic, (int(0.5*width), int(0.5*height)),
                     interpolation=cv2.INTER_CUBIC)

    for o in ob:
        xmin = int(o.getElementsByTagName('xmin')[0].firstChild.data)
        xmax = int(o.getElementsByTagName('xmax')[0].firstChild.data)
        ymin = int(o.getElementsByTagName('ymin')[0].firstChild.data)
        ymax = int(o.getElementsByTagName('ymax')[0].firstChild.data)
        cv2.rectangle(img=pic, pt1=(int(xmin/2), int(ymin/2)),
                      pt2=(int(xmax/2), int(ymax/2)), color=(255, 255, 255), thickness=3)
    for r in roi_box:
        xmin = int(r.getElementsByTagName('xmin')[0].firstChild.data)
        ymin = int(r.getElementsByTagName('ymin')[0].firstChild.data)
        xmax = int(r.getElementsByTagName('xmax')[0].firstChild.data)
        ymax = int(r.getElementsByTagName('ymax')[0].firstChild.data)
        cv2.rectangle(img=pic, pt1=(int(xmin/2), int(ymin/2)),
                      pt2=(int(xmax/2), int(ymax/2)), color=(255, 255, 255), thickness=3)

    cv2.imshow(img, pic)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyWindow(img)
        continue
    if cv2.waitKey(0) & 0xFF == ord('e'):
        break
