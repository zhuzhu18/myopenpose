#-*-coding:utf-8-*-
import torch
from model import bodypose_model
from util import transfer, padding_right_down_corner
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# paf map对应的肢体
mapIdx = [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3],
          [4, 5], [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33],
          [36, 37], [18, 19], [26, 27]]
# heatmap对应的肢体
limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17], [2, 16], [5, 17]]

class Body:
    def __init__(self, model_state_dict):
        self.model = bodypose_model()
        self.model.load_state_dict(transfer(self.model, torch.load(model_state_dict)))
        self.model.eval()

    def __call__(self, oriImg):
        base_size = 368
        stride = 8
        thre1 = 0.1
        thre2 = 0.05
        h, w = oriImg.shape[:-1]
        scale_search = [0.5, 1, 2, 2.5]
        multiplier = [base_size*scale/h for scale in scale_search]
        pafmap_avg = np.zeros((h, w, 38))
        heatmap_avg = np.zeros((h, w, 19))
        for scale in multiplier:
            img = cv2.resize(oriImg, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            _h, _w = img.shape[:-1]
            img, (pad_rows, pad_columns) = padding_right_down_corner(img, stride)
            img = np.transpose(img / 255 - 0.5, axes=[2,0,1])[np.newaxis, :, :, :]
            img = np.ascontiguousarray(img, dtype=np.float32)
            img = torch.from_numpy(img).to(device)

            with torch.no_grad():
                paf_map, heatmap = self.model(img)
            paf_map = paf_map.cpu().numpy()
            heatmap = heatmap.cpu().numpy()
            paf_map = np.transpose(np.squeeze(paf_map), axes=[1,2,0])
            heatmap = np.transpose(np.squeeze(heatmap), axes=[1,2,0])
            paf_map = cv2.resize(paf_map, dsize=(0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.resize(heatmap, dsize=(0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf_map = paf_map[:(_h-pad_rows), :(_w-pad_columns), :]
            heatmap = heatmap[:(_h-pad_rows), :(_w-pad_columns), :]
            paf_map = cv2.resize(paf_map, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.resize(heatmap, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            pafmap_avg += (pafmap_avg + paf_map / len(scale_search))
            heatmap_avg += (heatmap_avg + heatmap / len(scale_search))
        peak_counter = 0        # 计数检测到的关键点数
        all_peaks = []
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            heatmap = gaussian_filter(map_ori, sigma=3)
            map_up = np.zeros_like(heatmap)
            map_up[:-1, :] = heatmap[1:, :]
            map_down = np.zeros_like(heatmap)
            map_down[1:, :] = heatmap[:-1, :]
            map_left = np.zeros_like(heatmap)
            map_left[:, :-1] = heatmap[:, 1:]
            map_right = np.zeros_like(heatmap)
            map_right[:, 1:] = heatmap[:, :-1]
            peaks_binary = np.logical_and.reduce([heatmap > map_up, heatmap > map_down,
                            heatmap > map_left, heatmap > map_right, heatmap > thre1])    # 一类关键点的峰值二值图
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))   # [(x0, y0), (x1, y1), ...], 所有人的同一类关键点, 也可能为一个空列表
            # print(peaks)
            peak_with_score = [peak+(map_ori[peak[1], peak[0]], ) for peak in peaks]                # [(x0, y0, score0), (x1, y1, score1), ...], 也可能为一个空列表
            peak_id = list(range(peak_counter, peak_counter + len(peaks)))
            peak_with_score_and_id = [peak_with_score[i] + (peak_id[i], ) for i in range(len(peak_id))]     # [(x0, y0, score0, id0), (x1, y1, score1, id1), ...], 也可能为一个空列表
            peak_counter += len(peaks)
            all_peaks.append(peak_with_score_and_id)        # [[(x0, y0, score0, id0), (x1, y1, score1, id1), ...],...], 18类关键点, 内部元素也可能为一个空列表

        num_points = 10
        connection_all = []
        missing_limb = []
        for k in range(len(mapIdx)):
            paf_score = pafmap_avg[:, :, mapIdx[k]]

            candA = all_peaks[limbSeq[k][0]]     # [(x0, y0, score0, id0), (x1, y1, score1, id1), ...], 也可能为一个空列表
            candB = all_peaks[limbSeq[k][1]]
            nA = len(candA)
            nB = len(candB)

            if nA > 0 and nB > 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec_AB = np.subtract(candB[j][:2], candA[i][:2])   # 由A指向B的一个向量, array[a, b]
                        norm = np.sqrt(np.sum(np.power(vec_AB, 2)))
                        vec_AB = vec_AB / norm
                        sample_points = list(zip(np.linspace(candA[i][0], candB[j][0], num_points),
                                                 np.linspace(candA[i][1], candB[j][1], num_points)))    # A与B之间的采样点, [(x0, y0),...,(x9, y9)]
                        vec_x = np.array([paf_score[int(round(point[1])), int(round(point[0])), 0] for point in sample_points])      # array([u0, u1, ...,u[9]])
                        vec_y = np.array([paf_score[int(round(point[1])), int(round(point[0])), 1] for point in sample_points])      # array([v0, v1, ...,v[9]])

                        score_paf = vec_x*vec_AB[0] + vec_y*vec_AB[1]         # array[score0, score1, ..., score9]
                        score_with_dist_prior = score_paf.mean() +\
                                            min(0, 0.5*h/norm - 1)

                        if np.count_nonzero(score_paf > thre2) > 0.8*num_points and score_with_dist_prior > 0:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])      # [[i, j, score1, score2],...], 一条肢体初步筛选出的候选连接

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)       # [[i, j, score1, score2],...], 分数从高到低排序
                connection = np.zeros((0, 5))
                for i, j, score, _ in connection_candidate:
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], score, i, j]])      # array([[idA, idB, score, i, j], ... ])
                        if len(connection) > min(nA, nB):        # A和B之间连接的肢体数不能超过A或B的关键点数最小值
                            break
                connection_all.append(connection)
            else:
                missing_limb.append(k)
                connection_all.append([])            # [array([[idA, idB, score, i, j], ... ]), ... ], 19个元素, 可能有空列表
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])   # array[[x0, y0, score0, id0],...], 所有检测到的关键点组成一个二维列表
        for k in range(len(mapIdx)):
            if k not in missing_limb:
                partAs = connection_all[k][:, 0]     # 肢体k的关键点A的id, array([id0_A, id1_A,...]), 注意这里的id是总id
                partBs = connection_all[k][:, 1]     # 肢体k的关键点B的id, array([id0_B, id1_B,...])
                indexA, indexB = limbSeq[k]          # 肢体k的关键点A和关键点B分别在heatmap中的索引

                for i in range(len(connection_all[k])):       # 遍历第k条肢体的每一对连接
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:   # 这条肢体的关键点A或者关键点B在在某个人身上出现过
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]        # 如果只出现过一次, 先找出这是哪个人
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += connection_all[k][i, 2] + candidate[partBs[i].astype('int'), 2]

                    elif found == 2:            # 如果出现过两次
                        j1, j2 = subset_idx
                        membership = (subset[j1][:2] > 0).astype('int') + (subset[j2][:2] > 0).astype('int')
                        if len(np.nonzero(membership == 2)[0]) > 0:
                            subset[j1][:-2] += (subset[j2][:2] + 1)     # +1是为了消除subset中的-1
                            subset[j1][-2] += connection_all[k][i][2] + subset[j2][-2]
                            subset[j1][-1] += subset[j2][-1]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype('int'), 2] + connection_all[k][i][2]
                    elif found == 0:
                        row = -1 * np.ones(20)      # 创建一个新的人
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-2] = connection_all[k][i, 2] + candidate[[partAs[i].astype('int'), partBs[i].astype('int')], 2].sum()
                        subset = np.vstack([subset, row])
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                subset = np.delete(subset, i, 0)

        return candidate, subset
