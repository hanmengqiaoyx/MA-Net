import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import ViT
from layer import Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, Convolution6, Convolution7, Convolution8, Convolution9, \
    Convolution10, Convolution11, Convolution12, Convolution13, Convolution14, Convolution15, Convolution16, Convolution17, Fully_Connection


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class ResNet18(nn.Module):
    def __init__(self, in_channel=3, c_in0=64, c_in1=128, c_in2=256, c_in3=512, f_in=512, num_classes=10, expansion=1):
        super(ResNet18, self).__init__()
        self.c_in0 = c_in0
        self.c_in1 = c_in1
        self.c_in2 = c_in2
        self.c_in3 = c_in3
        self.f_in = f_in
        self.c_layer1 = Convolution1(in_channel, c_in0)
        self.c_layer2 = Convolution2(c_in0, c_in0)
        self.c_layer3 = Convolution3(c_in0, c_in0)
        self.c_layer4 = Convolution4(c_in0, c_in0)
        self.c_layer5 = Convolution5(c_in0, c_in0)
        self.c_layer6 = Convolution6(c_in0, c_in1)
        self.c_layer7 = Convolution7(c_in1, c_in1)
        self.c_layer8 = Convolution8(c_in1, c_in1)
        self.c_layer9 = Convolution9(c_in1, c_in1)
        self.c_layer10 = Convolution10(c_in1, c_in2)
        self.c_layer11 = Convolution11(c_in2, c_in2)
        self.c_layer12 = Convolution12(c_in2, c_in2)
        self.c_layer13 = Convolution13(c_in2, c_in2)
        self.c_layer14 = Convolution14(c_in2, c_in3)
        self.c_layer15 = Convolution15(c_in3, c_in3)
        self.c_layer16 = Convolution16(c_in3, c_in3)
        self.c_layer17 = Convolution17(c_in3, c_in3)
        self.f_layer18 = Fully_Connection(f_in, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4, 4)

        self.trans1 = nn.Sequential(
            SepConv(
                channel_in=64 * expansion,
                channel_out=128 * expansion
            ),
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.trans2 = nn.Sequential(
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion,
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.trans3 = nn.Sequential(
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.trans4 = nn.AvgPool2d(4, 4)

    def forward(self, input, pr_1, pr_2, pr_3, pr_4, pattern, i):
        feature_list = []
        feature_list1 = []
        if pattern == 0:
            out = self.c_layer1(input, 0, 0, pattern)
            out, out0 = self.c_layer2(out, 0, 0, pattern)
            out = self.c_layer3(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, 0, 0, pattern)
            out = self.c_layer5(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer6(out, 0, 0, 0, pattern)
            out = self.c_layer7(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, 0, 0, pattern)
            out = self.c_layer9(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer10(out, 0, 0, 0, pattern)
            out = self.c_layer11(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, 0, 0, pattern)
            out = self.c_layer13(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer14(out, 0, 0, 0, pattern)
            out = self.c_layer15(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, 0, 0, pattern)
            out = self.c_layer17(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)

            out1_feature = self.trans1(feature_list[0]).view(input.size(0), -1)
            out2_feature = self.trans2(feature_list[1]).view(input.size(0), -1)
            out3_feature = self.trans3(feature_list[2]).view(input.size(0), -1)
            out4_feature = self.trans4(feature_list[3]).view(input.size(0), -1)
            out = self.f_layer18(out4_feature, 0, 0, pattern)

            feat_list = [out4_feature, out3_feature, out2_feature, out1_feature]
            for index in range(len(feat_list)):
                feat_list[index] = F.normalize(feat_list[index], dim=1)
            if self.training:
                return out, feat_list
            else:
                return out

        elif pattern == 1:
            weight_avg1 = self.c_layer1(input, 0, 0, pattern)
            weight_avg2 = self.c_layer2(input, 0, 0, pattern)
            weight_avg3 = self.c_layer3(input, 0, 0, pattern)
            weight_avg4 = self.c_layer4(input, 0, 0, pattern)
            weight_avg5 = self.c_layer5(input, 0, 0, pattern)
            weight_avg_1 = torch.cat((weight_avg1, weight_avg2, weight_avg3, weight_avg4, weight_avg5), dim=0).view(1, -1)  # [5, 64]
            weight_avg6, shortcut_weight_avg6 = self.c_layer6(input, 0, 0, 0, pattern)
            weight_avg7 = self.c_layer7(input, 0, 0, pattern)
            weight_avg8 = self.c_layer8(input, 0, 0, pattern)
            weight_avg9 = self.c_layer9(input, 0, 0, pattern)
            weight_avg_2 = torch.cat((weight_avg6, shortcut_weight_avg6, weight_avg7, weight_avg8, weight_avg9), dim=0).view(1, -1)  # [5, 128]
            weight_avg10, shortcut_weight_avg10 = self.c_layer10(input, 0, 0, 0, pattern)
            weight_avg11 = self.c_layer11(input, 0, 0, pattern)
            weight_avg12 = self.c_layer12(input, 0, 0, pattern)
            weight_avg13 = self.c_layer13(input, 0, 0, pattern)
            weight_avg_3 = torch.cat((weight_avg10, shortcut_weight_avg10, weight_avg11, weight_avg12, weight_avg13), dim=0).view(1, -1)  # [5, 256]
            weight_avg14, shortcut_weight_avg14 = self.c_layer14(input, 0, 0, 0, pattern)
            weight_avg15 = self.c_layer15(input, 0, 0, pattern)
            weight_avg16 = self.c_layer16(input, 0, 0, pattern)
            weight_avg17 = self.c_layer17(input, 0, 0, pattern)
            weight_avg18 = self.f_layer18(input, 0, 0, pattern)
            weight_avg_4 = torch.cat((weight_avg14, shortcut_weight_avg14, weight_avg15, weight_avg16, weight_avg17, weight_avg18), dim=0).view(1, -1)  # [6, 512]
            return weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4

        elif pattern == 2:
            if i == 1:
                pr_1_1 = pr_1[0][0:1, :, :].view(5, 64)
                pr_1_2 = pr_1[0][1:2, :, :].view(5, 64)
                pr_1_3 = pr_1[1][0:1, :, :].view(5, 64)
                pr_1_4 = pr_1[1][1:2, :, :].view(5, 64)
                pr1_1 = pr_1_1[0:1, :]
                pr2_1 = pr_1_1[1:2, :]
                pr3_1 = pr_1_1[2:3, :]
                pr4_1 = pr_1_1[3:4, :]
                pr5_1 = pr_1_1[4:5, :]
                pr1_2 = pr_1_2[0:1, :]
                pr2_2 = pr_1_2[1:2, :]
                pr3_2 = pr_1_2[2:3, :]
                pr4_2 = pr_1_2[3:4, :]
                pr5_2 = pr_1_2[4:5, :]
                pr1_3 = pr_1_3[0:1, :]
                pr2_3 = pr_1_3[1:2, :]
                pr3_3 = pr_1_3[2:3, :]
                pr4_3 = pr_1_3[3:4, :]
                pr5_3 = pr_1_3[4:5, :]
                pr1_4 = pr_1_4[0:1, :]
                pr2_4 = pr_1_4[1:2, :]
                pr3_4 = pr_1_4[2:3, :]
                pr4_4 = pr_1_4[3:4, :]
                pr5_4 = pr_1_4[4:5, :]
                pr6 = pr_2[0:1, :]
                c_shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
                pr10 = pr_3[0:1, :]
                c_shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
                pr14 = pr_4[0:1, :]
                c_shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, pr1_1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2_1, 1, pattern)
                out = self.c_layer3(out, pr3_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4_1, 1, pattern)
                out = self.c_layer5(out, pr5_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                out = self.c_layer1(input, pr1_2, 1, pattern)
                out, out0 = self.c_layer2(out, pr2_2, 1, pattern)
                out = self.c_layer3(out, pr3_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4_2, 1, pattern)
                out = self.c_layer5(out, pr5_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                out = self.c_layer1(input, pr1_3, 1, pattern)
                out, out0 = self.c_layer2(out, pr2_3, 1, pattern)
                out = self.c_layer3(out, pr3_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4_3, 1, pattern)
                out = self.c_layer5(out, pr5_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 4
                out = self.c_layer1(input, pr1_4, 1, pattern)
                out, out0 = self.c_layer2(out, pr2_4, 1, pattern)
                out = self.c_layer3(out, pr3_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4_4, 1, pattern)
                out = self.c_layer5(out, pr5_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)

                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                feature_list1.append(torch.cat((feature_list[2], feature_list[3]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out, feature_list1
                else:
                    return out

            elif i == 2:
                pr_2_1 = pr_2[0][0:1, :, :].view(5, 128)
                pr_2_2 = pr_2[0][1:2, :, :].view(5, 128)
                pr_2_3 = pr_2[1][0:1, :, :].view(5, 128)
                pr_2_4 = pr_2[1][1:2, :, :].view(5, 128)
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
                pr6_1 = pr_2_1[0:1, :]
                c_shortcut_pr6_1 = pr_2_1[1:2, :]
                pr7_1 = pr_2_1[2:3, :]
                pr8_1 = pr_2_1[3:4, :]
                pr9_1 = pr_2_1[4:5, :]
                pr6_2 = pr_2_2[0:1, :]
                c_shortcut_pr6_2 = pr_2_2[1:2, :]
                pr7_2 = pr_2_2[2:3, :]
                pr8_2 = pr_2_2[3:4, :]
                pr9_2 = pr_2_2[4:5, :]
                pr6_3 = pr_2_3[0:1, :]
                c_shortcut_pr6_3 = pr_2_3[1:2, :]
                pr7_3 = pr_2_3[2:3, :]
                pr8_3 = pr_2_3[3:4, :]
                pr9_3 = pr_2_3[4:5, :]
                pr6_4 = pr_2_4[0:1, :]
                c_shortcut_pr6_4 = pr_2_4[1:2, :]
                pr7_4 = pr_2_4[2:3, :]
                pr8_4 = pr_2_4[3:4, :]
                pr9_4 = pr_2_4[4:5, :]
                pr10 = pr_3[0:1, :]
                c_shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
                pr14 = pr_4[0:1, :]
                c_shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6_1, c_shortcut_pr6_1, 1, pattern)
                out = self.c_layer7(out, pr7_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8_1, 1, pattern)
                out = self.c_layer9(out, pr9_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6_2, c_shortcut_pr6_2, 1, pattern)
                out = self.c_layer7(out, pr7_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8_2, 1, pattern)
                out = self.c_layer9(out, pr9_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6_3, c_shortcut_pr6_3, 1, pattern)
                out = self.c_layer7(out, pr7_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8_3, 1, pattern)
                out = self.c_layer9(out, pr9_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 4
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6_4, c_shortcut_pr6_4, 1, pattern)
                out = self.c_layer7(out, pr7_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8_4, 1, pattern)
                out = self.c_layer9(out, pr9_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                feature_list1.append(torch.cat((feature_list[2], feature_list[3]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out, feature_list1
                else:
                    return out

            elif i == 3:
                pr_3_1 = pr_3[0][0:1, :, :].view(5, 256)
                pr_3_2 = pr_3[0][1:2, :, :].view(5, 256)
                pr_3_3 = pr_3[1][0:1, :, :].view(5, 256)
                pr_3_4 = pr_3[1][1:2, :, :].view(5, 256)
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
                pr6 = pr_2[0:1, :]
                c_shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
                pr10_1 = pr_3_1[0:1, :]
                c_shortcut_pr10_1 = pr_3_1[1:2, :]
                pr11_1 = pr_3_1[2:3, :]
                pr12_1 = pr_3_1[3:4, :]
                pr13_1 = pr_3_1[4:5, :]
                pr10_2 = pr_3_2[0:1, :]
                c_shortcut_pr10_2 = pr_3_2[1:2, :]
                pr11_2 = pr_3_2[2:3, :]
                pr12_2 = pr_3_2[3:4, :]
                pr13_2 = pr_3_2[4:5, :]
                pr10_3 = pr_3_3[0:1, :]
                c_shortcut_pr10_3 = pr_3_3[1:2, :]
                pr11_3 = pr_3_3[2:3, :]
                pr12_3 = pr_3_3[3:4, :]
                pr13_3 = pr_3_3[4:5, :]
                pr10_4 = pr_3_4[0:1, :]
                c_shortcut_pr10_4 = pr_3_4[1:2, :]
                pr11_4 = pr_3_4[2:3, :]
                pr12_4 = pr_3_4[3:4, :]
                pr13_4 = pr_3_4[4:5, :]

                pr14 = pr_4[0:1, :]
                c_shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10_1, c_shortcut_pr10_1, 1, pattern)
                out = self.c_layer11(out, pr11_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12_1, 1, pattern)
                out = self.c_layer13(out, pr13_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10_2, c_shortcut_pr10_2, 1, pattern)
                out = self.c_layer11(out, pr11_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12_2, 1, pattern)
                out = self.c_layer13(out, pr13_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10_3, c_shortcut_pr10_3, 1, pattern)
                out = self.c_layer11(out, pr11_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12_3, 1, pattern)
                out = self.c_layer13(out, pr13_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 4
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10_4, c_shortcut_pr10_4, 1, pattern)
                out = self.c_layer11(out, pr11_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12_4, 1, pattern)
                out = self.c_layer13(out, pr13_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16, 1, pattern)
                out = self.c_layer17(out, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                feature_list1.append(torch.cat((feature_list[2], feature_list[3]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out, feature_list1
                else:
                    return out

            elif i == 4:
                pr_4_1 = pr_4[0][0:1, :, :].view(6, 512)
                pr_4_2 = pr_4[0][1:2, :, :].view(6, 512)
                pr_4_3 = pr_4[1][0:1, :, :].view(6, 512)
                pr_4_4 = pr_4[1][1:2, :, :].view(6, 512)
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
                pr6 = pr_2[0:1, :]
                c_shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
                pr10 = pr_3[0:1, :]
                c_shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
                pr14_1 = pr_4_1[0:1, :]
                c_shortcut_pr14_1 = pr_4_1[1:2, :]
                pr15_1 = pr_4_1[2:3, :]
                pr16_1 = pr_4_1[3:4, :]
                pr17_1 = pr_4_1[4:5, :]
                pr18_1 = pr_4_1[5:6, :]
                pr14_2 = pr_4_2[0:1, :]
                c_shortcut_pr14_2 = pr_4_2[1:2, :]
                pr15_2 = pr_4_2[2:3, :]
                pr16_2 = pr_4_2[3:4, :]
                pr17_2 = pr_4_2[4:5, :]
                pr18_2 = pr_4_2[5:6, :]
                pr14_3 = pr_4_3[0:1, :]
                c_shortcut_pr14_3 = pr_4_3[1:2, :]
                pr15_3 = pr_4_3[2:3, :]
                pr16_3 = pr_4_3[3:4, :]
                pr17_3 = pr_4_3[4:5, :]
                pr18_3 = pr_4_3[5:6, :]
                pr14_4 = pr_4_4[0:1, :]
                c_shortcut_pr14_4 = pr_4_4[1:2, :]
                pr15_4 = pr_4_4[2:3, :]
                pr16_4 = pr_4_4[3:4, :]
                pr17_4 = pr_4_4[4:5, :]
                pr18_4 = pr_4_4[5:6, :]
                # 1
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14_1, c_shortcut_pr14_1, 1, pattern)
                out = self.c_layer15(out, pr15_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16_1, 1, pattern)
                out = self.c_layer17(out, pr17_1, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18_1, 1, pattern)
                feature_list.append(out)
                # 2
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14_2, c_shortcut_pr14_2, 1, pattern)
                out = self.c_layer15(out, pr15_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16_2, 1, pattern)
                out = self.c_layer17(out, pr17_2, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18_2, 1, pattern)
                feature_list.append(out)
                # 3
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14_3, c_shortcut_pr14_3, 1, pattern)
                out = self.c_layer15(out, pr15_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16_3, 1, pattern)
                out = self.c_layer17(out, pr17_3, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18_3, 1, pattern)
                feature_list.append(out)
                # 4
                out = self.c_layer1(input, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, pr2, 1, pattern)
                out = self.c_layer3(out, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, pr4, 1, pattern)
                out = self.c_layer5(out, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, pr8, 1, pattern)
                out = self.c_layer9(out, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, pr12, 1, pattern)
                out = self.c_layer13(out, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, pr14_4, c_shortcut_pr14_4, 1, pattern)
                out = self.c_layer15(out, pr15_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, pr16_4, 1, pattern)
                out = self.c_layer17(out, pr17_4, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18_4, 1, pattern)
                feature_list.append(out)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                feature_list1.append(torch.cat((feature_list[2], feature_list[3]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out, feature_list1
                else:
                    return out

        elif pattern == 3:
            pr1 = pr_1[0:1, :]
            pr2 = pr_1[1:2, :]
            pr3 = pr_1[2:3, :]
            pr4 = pr_1[3:4, :]
            pr5 = pr_1[4:5, :]
            pr6 = pr_2[0:1, :]
            c_shortcut_pr6 = pr_2[1:2, :]
            pr7 = pr_2[2:3, :]
            pr8 = pr_2[3:4, :]
            pr9 = pr_2[4:5, :]
            pr10 = pr_3[0:1, :]
            c_shortcut_pr10 = pr_3[1:2, :]
            pr11 = pr_3[2:3, :]
            pr12 = pr_3[3:4, :]
            pr13 = pr_3[4:5, :]
            pr14 = pr_4[0:1, :]
            c_shortcut_pr14 = pr_4[1:2, :]
            pr15 = pr_4[2:3, :]
            pr16 = pr_4[3:4, :]
            pr17 = pr_4[4:5, :]
            pr18 = pr_4[5:6, :]
            out = self.c_layer1(input, pr1, 1, pattern)
            out, out0 = self.c_layer2(out, pr2, 1, pattern)
            out = self.c_layer3(out, pr3, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, pr4, 1, pattern)
            out = self.c_layer5(out, pr5, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer6(out, pr6, c_shortcut_pr6, 1, pattern)
            out = self.c_layer7(out, pr7, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, pr8, 1, pattern)
            out = self.c_layer9(out, pr9, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer10(out, pr10, c_shortcut_pr10, 1, pattern)
            out = self.c_layer11(out, pr11, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, pr12, 1, pattern)
            out = self.c_layer13(out, pr13, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer14(out, pr14, c_shortcut_pr14, 1, pattern)
            out = self.c_layer15(out, pr15, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, pr16, 1, pattern)
            out = self.c_layer17(out, pr17, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)

            out1_feature = self.trans1(feature_list[0]).view(input.size(0), -1)
            out2_feature = self.trans2(feature_list[1]).view(input.size(0), -1)
            out3_feature = self.trans3(feature_list[2]).view(input.size(0), -1)
            out4_feature = self.trans4(feature_list[3]).view(input.size(0), -1)
            out = self.f_layer18(out4_feature, pr18, 1, pattern)

            feat_list = [out4_feature, out3_feature, out2_feature, out1_feature]
            for index in range(len(feat_list)):
                feat_list[index] = F.normalize(feat_list[index], dim=1)
            if self.training:
                return out, feat_list
            else:
                return out