#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from collections import OrderedDict

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layers.append((layer_name, nn.MaxPool2d(v[0], v[1], v[2])))
        else:
            layers.append((layer_name, nn.Conv2d(v[0], v[1], v[2], v[3], v[4])))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        block0 = OrderedDict({
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1_stage1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2_stage1': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3_stage1': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3_CPM': [512, 256, 3, 1, 1],
            'conv4_4_CPM': [256, 128, 3, 1, 1]
        })
        block1_1 = OrderedDict({
            'conv5_1_CPM_L1': [128, 128, 3, 1, 1],
            'conv5_2_CPM_L1': [128, 128, 3, 1, 1],
            'conv5_3_CPM_L1': [128, 128, 3, 1, 1],
            'conv5_4_CPM_L1': [128, 512, 1, 1, 0],
            'conv5_5_CPM_L1': [512, 38, 1, 1, 0]
        })
        block1_2 = OrderedDict({
            'conv5_1_CPM_L2': [128, 128, 3, 1, 1],
            'conv5_2_CPM_L2': [128, 128, 3, 1, 1],
            'conv5_3_CPM_L2': [128, 128, 3, 1, 1],
            'conv5_4_CPM_L2': [128, 512, 1, 1, 0],
            'conv5_5_CPM_L2': [512, 19, 1, 1, 0]
        })
        blocks = dict()
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2
        for i in range(2, 7):
            blocks['block%d_1'%i] = OrderedDict({
                'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]
            })
            blocks['block%d_2'%i] = OrderedDict({
                'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]
            })
        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)
        self.model0 = make_layers(block0, no_relu_layers)
        self.make_stage(blocks)

    def make_stage(self, blocks):
        for i in range(1, 7):
            setattr(self, 'model%d_1'%i, blocks['block%d_1'%i])
            setattr(self, 'model%d_2'%i, blocks['block%d_2'%i])

    def forward(self, x):
        x = self.model0(x)
        vgg_feature = x
        for i in range(1, 6):
            x1 = getattr(self, 'model%d_1'%i)(x)
            x2 = getattr(self, 'model%d_2'%i)(x)
            x = torch.cat([x1, x2, vgg_feature], dim=1)
        out1 = self.model6_1(x)
        out2 = self.model6_2(x)

        return out1, out2
