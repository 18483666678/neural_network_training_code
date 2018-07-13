import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

anno_src = r"/media/tensorflow01/My Passport/CelebA/Anno/list_bbox_celeba.txt"
img_dir = r"/media/tensorflow01/My Passport/CelebA/Img/img_celeba.7z/img_celeba"

save_path = r"/media/tensorflow01/My Passport/Result"

for face_size in [24]:
    print("gen %i image"%face_size)
    #样本图片存储路径
    positive_image_dir = os.path.join(save_path,str(face_size),"positive")
    negative_image_dir = os.path.join(save_path,str(face_size),"negative")
    part_image_dir = os.path.join(save_path,str(face_size),"part")

    for dir_path in [positive_image_dir,negative_image_dir,part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    #样本描述路径
    positive_anno_filename = os.path.join(save_path,str(face_size),"positive.txt")
    negative_anno_filename = os.path.join(save_path,str(face_size),"negative.txt")
    part_anno_filename = os.path.join(save_path,str(face_size),"part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename,"w")
        negative_anno_file = open(negative_anno_filename,"w")
        part_anno_file = open(part_anno_filename,"w")

        for i, line in enumerate(open(anno_src)):
            if i<2:
                continue
                #  i<2时
            try:
                strs = line.strip().split(" ")
                strs = list(filter(bool,strs))
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir,image_filename)
                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    # #五个关键点
                    # px1 = 0  # float(strs[5].strip())
                    # py1 = 0  # float(strs[6].strip())
                    # px2 = 0  # float(strs[7].strip())
                    # py2 = 0  # float(strs[8].strip())
                    # px3 = 0  # float(strs[9].strip())
                    # py3 = 0  # float(strs[10].strip())
                    # px4 = 0  # float(strs[11].strip())
                    # py4 = 0  # float(strs[12].strip())
                    # px5 = 0  # float(strs[13].strip())
                    # py5 = 0  # float(strs[14].strip())

                    if max(w,h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue
                        #丢失掉的太小的label
                        #
                        #
                    boxes = [[x1,y2,x2,y2]]

                    #计算出人脸中心点的位置
                    cx = x1 + w/2
                    cy = y1 + h/2

                    #使正样本和部分的样本数量翻倍
                    for _ in range(20):
                        #让人脸中心点有少许的偏移
                        w_ = np.random.randint(-0.7*w, 0.7*w)
                        h_ = np.random.randint(-0.7*h, 0.7*h)
                        cx_ =cx + w_
                        cy_ =cy + h_

                        #让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w,h)*0.8),np.ceil(1.25*max(w,h)))
                        #
                        #robustness 生成的一个正方形
                        #
                        x1_ = np.max(cx_ - side_len/2,0)
                        y1_ = np.max(cy_ - side_len/2,0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_ ,y2_])

                        #计算坐标的偏移值
                        offset_x1 = (x1-x1_)/side_len
                        offset_y1 = (y1-y1_)/side_len
                        offset_x2 = (x2-x2_)/side_len
                        offset_y2 = (y2-y2_)/side_len

                        offset_px1 = 0  # (px1 - x1_) / side_len
                        offset_py1 = 0  # (py1 - y1_) / side_len
                        offset_px2 = 0  # (px2 - x1_) / side_len
                        offset_py2 = 0  # (py2 - y1_) / side_len
                        offset_px3 = 0  # (px3 - x1_) / side_len
                        offset_py3 = 0  # (py3 - y1_) / side_len
                        offset_px4 = 0  # (px4 - x1_) / side_len
                        offset_py4 = 0  # (py4 - y1_) / side_len
                        offset_px5 = 0  # (px5 - x1_) / side_len
                        offset_py5 = 0  # (py5 - y1_) / side_len

                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size,face_size),Image.ANTIALIAS)

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.65:
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir,"{0}.jpg".format(positive_count)))
                            positive_count+=1
                        elif iou > 0.4:
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.3:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    negative_count, 0, offset_x1, offset_y1, offset_x2, offset_y2, offset_px1, offset_py1,
                                    offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count+= 1
            except Exception as e:
                traceback.print_exc()
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

