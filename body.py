import cv2
import numpy as np
import util
import math

from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from model import bodypose_model
import torch


class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        ori_h, ori_w, _ = oriImg.shape
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / ori_h for x in scale_search]
        heatmap_avg = np.zeros((ori_h, ori_w, 19))
        paf_avg = np.zeros((ori_h, ori_w, 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)        # 填充右下角,使长宽都变成stride的整数倍
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5    # 使图片的维度变成1*3*H*W
            im = np.ascontiguousarray(im)       # 此时的im数组在内存中不连续,将其转换成连续的

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()         # pafs: 1*38*46*46
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()         # heatmaps: 1*19*46*46

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))           # 转换成H*W*19维的
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)       # 将长和宽放大8倍,变成与原图的0.5倍填充后一样大小
            heatmap = heatmap[:(imageToTest_padded.shape[0] - pad[2]), :(imageToTest_padded.shape[1] - pad[3]), :]    # 去掉右下角的填充,变成与原图的0.5倍未填充一样大小
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)       # 放大成原图一样大小


            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))          # 转换成H*W*38维的
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)            # 将长和宽放大8倍,变成与原图的0.5倍填充后一样大小
            paf = paf[:(imageToTest_padded.shape[0] - pad[2]), :(imageToTest_padded.shape[1] - pad[3]), :]     # 去掉右下角的填充,变成与原图的0.5倍未填充一样大小
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)           # 放大成原图一样大小

            heatmap_avg += heatmap / len(multiplier)
            paf_avg += paf / len(multiplier)

        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]          # 每个关键点的heatmap, H * W
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))

            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # 峰值点的[(x, y),...]坐标
            peaks_with_score = [peak + (map_ori[peak[1], peak[0]],) for peak in peaks]   # 将峰值点坐标和分数绑定在一起, [(x, y, score),...]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]   # 将带分数的峰值点和它的id绑定在一起, [(x, y, score, id),...]
            # 注意peaks_with_score_and_id这里得到的是图片中所有人的同一个关键点, 如所有人的手腕
            all_peaks.append(peaks_with_score_and_id)      # 18类关节的坐标, 分数和id
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence (19)
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]       # H * W * 2, 预测的一条肢体的亲和场
            candA = all_peaks[limbSeq[k][0] - 1]       # 某条肢体(如手臂)的一类关键点(手腕), [(x,y,score,id),...]
            candB = all_peaks[limbSeq[k][1] - 1]       # 这条肢体的另一类关键点, [(x,y,score,id),...]
            nA = len(candA)              # 这一类关键点的总数
            nB = len(candB)              # 这一类关键点的总数
            indexA, indexB = limbSeq[k]       # 第k条肢体
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])         # 得到一个由candA[i]指向candB[j]的向量, 如array([x, y])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])   # 该向量的范数, 即向量的幅值
                        if norm == 0:              # 防止向量的范数为0
                            norm = 0.1
                        vec = np.divide(vec, norm)         # 将该向量标准化为单位向量

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))    # 候选点A和B之间的一些采样点, [(x, y),...],10个采样点, 注意为dtype为float型

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])              # 位于10个候选点的亲和场的x坐标, array([x0, x1, x2,...,x9])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])              # 位于候选点的亲和场的y坐标, array([y0, y1, y2,...,y9])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])     # 将10个亲和场向量的(x, y)坐标与候选点candA[i]和candB[j]构成的单位向量的(x, y)坐标做内积


                        ############################# 这一部分暂时没看懂  #############################

                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)      # 得到平均分数

                        ############################# 这一部分暂时没看懂  #############################
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)      # 分数大于0.05的个数多于80%
                        criterion2 = score_with_dist_prior > 0                 # 平均分数大于0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])   # 将要连接的候选点, [[i, j, 亲和场的平均分数, 总分数],...]
                            # i和j分别表示candA和candB中符合标准的关键点的id

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)      # 按亲和场的平均分数从高到低排序, [[i, j, 亲和场的平均分数, 总分数],...]
                connection = np.zeros((0, 5))

                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])      # [符合标准的candA的id, candB的id, 亲和场的平均分数,]
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset

if __name__ == "__main__":
    body_estimation = Body('../model/body_pose_model.pth')

    test_image = '../images/ski.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
