#-*-coding:utf-8-*-
import cv2
import os
from body import Body
from util import draw_keypoints, draw_skeleton

if __name__ == '__main__':
    weight_path = '/home/tl/Desktop/pytorch-openpose (copy)/model/body_pose_model.pth'
    test_dir = '/media/zhuzhu/6684B82784B7F81F/debug/src'
    dest_dir = '/media/zhuzhu/6684B82784B7F81F/debug/dst'
    model = Body(weight_path)
    for img in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img)
        oriImg = cv2.imread(img_path)  # B,G,R order
        candidate, subset = model(ori_img)
        canvas = draw_keypoints(ori_img, candidate, subset)
        canvas = draw_skeleton(canvas, candidate, subset)
        cv2.imwrite(os.path.join(dest_dir, img), canvas)
