import numpy as np

def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0])*(box[3] - box[1])
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3] - boxes[:,1])
    xx1 = np.maximum(box[0],boxes[:,0])
    yy1 = np.maximum(box[1],boxes[:,1])
    xx2 = np.maximum(box[2],boxes[:,2])
    yy2 = np.maximum(box[3],boxes[:,3])

    w = np.maximum(0,xx2 - xx1)
    h = np.maximum(0,yy2 - yy1)

    inter = w*h

    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area,area))
    else:
        ovr = np.true_divide(inter,(box_area+area-inter))
    return ovr
#return the rate of the iou

# boxes.shape = batch_num * [V]

def nms(boxes, thresh=0.3, isMin = False):

    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:,4]).argsort()]
    # -从大到小 按照iou大小
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        # index of the boxes maxest and others

        r_boxes.append(a_box)

        #print(iou(a_box,b_boxes))
        # caculate the iou  to filter the boxes

        index = np.where(iou(a_box,b_boxes,isMin)< thresh)
        # 1: np.where(conditions,x,y)
        # 2: np.where(conditions)
        # return index satisfy the conditions
        _boxes = b_boxes[index]
        #
        #
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)
    # default np.stack(vector,axis=0,out=None)

def convert_to_square(bbox):
    square_bbox = bbox.copy()
    #
    if bbox.shape[0] == 0:
        return np.array([])

    h = bbox[:, 3]- bbox[:, 1]
    w = bbox[:, 2]- bbox[:, 0]
    #长方形的宽和高
    max_side = np.maximum(h, w)
    #正方形的边长
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    #确定正方形的x1，y1
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)

    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
