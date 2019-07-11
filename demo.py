import cv2
import util

from body import Body
import copy
import os

body_estimation = Body('/home/zhuzhu/Desktop/pytorch-openpose/model/body_pose_model.pth')

test_dir = '/media/zhuzhu/6684B82784B7F81F/debug/src'
dest_dir = '/media/zhuzhu/6684B82784B7F81F/debug/dst'

for img in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img)
    oriImg = cv2.imread(img_path)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    cv2.imwrite(os.path.join(dest_dir, img), canvas)
# import numpy as np
# from scipy.ndimage.filters import gaussian_filter1d
#
# a = np.arange(7.0, step=1)
# print(a)
# b = gaussian_filter1d(a, sigma=1)
# print(b)