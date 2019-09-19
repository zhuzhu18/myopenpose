#-*-coding:utf-8-*-
import cv2
import numpy as np
import copy

# ['nose', 'neck', 'left shoulder', 'left elbow', 'left wrist',
#  'right shoulder', 'right elbow', 'right wrist', 'left ankle',
#  'left knee', 'left leg', 'right ankle', 'right knee', 'right leg',
#  'left eye', 'right eye', 'left ear', 'right ear']
limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17], [2, 16], [5, 17]]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
def transfer(model, model_dict):
    transfered_model_dict = dict()
    for state_name in model.state_dict().keys():
        transfered_model_dict[state_name] = model_dict['.'.join(state_name.split('.')[1:])]
    return transfered_model_dict

def padding_right_down_corner(img, stride):
    h, w, c = img.shape
    pad_rows = 0 if h % stride == 0 else stride - h % stride
    pad_columns = 0 if w % stride == 0 else stride - w % stride
    img_padded = np.concatenate([img, np.zeros((h, pad_columns, c))], axis=1).astype('uint8')
    img_padded = np.concatenate([img_padded, np.zeros((pad_rows, img_padded.shape[1], c))], axis=0).astype('uint8')

    return img_padded, (pad_rows, pad_columns)

def draw_keypoints(canvas, candidate, subset):
    for i in range(18):
        for person in subset:
            if int(person[i]) == -1:
                continue
            x, y = candidate[int(person[i])][:2]
            cv2.circle(canvas, center=(int(x), int(y)), radius=4, color=colors[i], thickness=-1)

    return canvas

def draw_skeleton(canvas, candidate, subset):
    for i in range(17):
        for person in subset:
            partA, partB = person[limbSeq[i]]
            if partA == -1 or partB == -1:
                continue
            cur_canvas = copy.deepcopy(canvas)
            x1, x2 = candidate[[int(partA), int(partB)], 0]
            y1, y2 = candidate[[int(partA), int(partB)], 1]
            median_x = np.mean([x1, x2])
            median_y = np.mean([y1, y2])
            length = np.sqrt(np.sum((x1-x2)**2+(y1-y2)**2))
            angle = np.degrees(np.arctan2(y1-y2, x1-x2))
            polygon = cv2.ellipse2Poly((int(median_x), int(median_y)), (int(length/2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas
