import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import ViT


class MLP_Encoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class MLP_Decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Astrocyte_Network_head(nn.Module):
    def __init__(self, image_height=21, image_width=512, patch_height=1, patch_width=512, depth=2, heads=8, dim=64, dim_head=64, mlp_dim=128, channels=1, num_classes=10752):
        super(Astrocyte_Network_head, self).__init__()
        self.c_1 = 64
        self.c_2 = 128
        self.c_3 = 256
        self.c_4 = 512
        self.encoder1 = MLP_Encoder(self.c_1, self.c_4)
        self.encoder2 = MLP_Encoder(self.c_2, self.c_4)
        self.encoder3 = MLP_Encoder(self.c_3, self.c_4)
        self.encoder4 = MLP_Encoder(self.c_4, self.c_4)

        self.vit = ViT(image_height, image_width, patch_height, patch_width, depth, heads, dim, dim_head, mlp_dim, channels, num_classes)

        self.decoder1 = MLP_Decoder(self.c_4, self.c_1)
        self.decoder2 = MLP_Decoder(self.c_4, self.c_2)
        self.decoder3 = MLP_Decoder(self.c_4, self.c_3)
        self.decoder4 = MLP_Decoder(self.c_4, self.c_4)

    def forward(self, weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4):
        weight_avg1 = self.encoder1(weight_avg_1[0:1, 0:self.c_1])
        weight_avg2 = self.encoder1(weight_avg_1[1:2, self.c_1:2*self.c_1])
        weight_avg3 = self.encoder1(weight_avg_1[2:3, 2*self.c_1:3*self.c_1])
        weight_avg4 = self.encoder1(weight_avg_1[3:4, 3*self.c_1:4*self.c_1])
        weight_avg5 = self.encoder1(weight_avg_1[4:5, 4*self.c_1:5*self.c_1])
        weight_avg6 = self.encoder2(weight_avg_2[0:1, 0:self.c_2])
        shortcut_weight_avg6 = self.encoder2(weight_avg_2[1:2, self.c_2:2*self.c_2])
        weight_avg7 = self.encoder2(weight_avg_2[2:3:, 2*self.c_2:3*self.c_2])
        weight_avg8 = self.encoder2(weight_avg_2[3:4:, 3*self.c_2:4*self.c_2])
        weight_avg9 = self.encoder2(weight_avg_2[4:5:, 4*self.c_2:5*self.c_2])
        weight_avg10 = self.encoder3(weight_avg_3[0:1, 0:self.c_3])
        shortcut_weight_avg10 = self.encoder3(weight_avg_3[1:2, self.c_3:2*self.c_3])
        weight_avg11 = self.encoder3(weight_avg_3[2:3:, 2*self.c_3:3*self.c_3])
        weight_avg12 = self.encoder3(weight_avg_3[3:4:, 3*self.c_3:4*self.c_3])
        weight_avg13 = self.encoder3(weight_avg_3[4:5:, 4*self.c_3:5*self.c_3])
        weight_avg14 = self.encoder4(weight_avg_4[0:1:, 0:self.c_4])
        shortcut_weight_avg14 = self.encoder4(weight_avg_4[1:2:, self.c_4:2*self.c_4])
        weight_avg15 = self.encoder4(weight_avg_4[2:3, 2*self.c_4:3*self.c_4])
        weight_avg16 = self.encoder4(weight_avg_4[3:4, 3*self.c_4:4*self.c_4])
        weight_avg17 = self.encoder4(weight_avg_4[4:5, 4*self.c_4:5*self.c_4])
        weight_avg18 = self.encoder4(weight_avg_4[6:7, 5*self.c_4:6*self.c_4])
        weight_avg = torch.cat((weight_avg1, weight_avg2, weight_avg3, weight_avg4, weight_avg5, weight_avg6, shortcut_weight_avg6, weight_avg7, weight_avg8,
                                weight_avg9, weight_avg10, shortcut_weight_avg10, weight_avg11, weight_avg12, weight_avg13, weight_avg14, shortcut_weight_avg14,
                                weight_avg15, weight_avg16, weight_avg17, weight_avg18), dim=0).view(1, 1, -1, self.c_4)  # [1, 1, 21, 512]

        preds = self.vit(weight_avg).view(-1, 512)

        weight_avg1 = self.decoder1(preds[0:1, :])
        weight_avg2 = self.decoder1(preds[1:2, :])
        weight_avg3 = self.decoder1(preds[2:3, :])
        weight_avg4 = self.decoder1(preds[3:4, :])
        weight_avg5 = self.decoder1(preds[4:5, :])
        weight_avg6 = self.decoder2(preds[5:6, :])
        shortcut_weight_avg6 = self.decoder2(preds[6:7, :])
        weight_avg7 = self.decoder2(preds[7:8, :])
        weight_avg8 = self.decoder2(preds[8:9, :])
        weight_avg9 = self.decoder2(preds[9:10, :])
        weight_avg10 = self.decoder3(preds[10:11, :])
        shortcut_weight_avg10 = self.decoder3(preds[11:12, :])
        weight_avg11 = self.decoder3(preds[12:13, :])
        weight_avg12 = self.decoder3(preds[13:14, :])
        weight_avg13 = self.decoder3(preds[14:15, :])
        weight_avg14 = self.decoder4(preds[15:16, :])
        shortcut_weight_avg14 = self.decoder4(preds[16:17, :])
        weight_avg15 = self.decoder4(preds[17:18, :])
        weight_avg16 = self.decoder4(preds[18:19, :])
        weight_avg17 = self.decoder4(preds[19:20, :])
        weight_avg18 = self.decoder4(preds[20:21, :])
        weight_avg_1 = torch.cat((weight_avg1, weight_avg2, weight_avg3, weight_avg4, weight_avg5), dim=1).view(-1, 64)  # [5, 64]
        weight_avg_2 = torch.cat((weight_avg6, shortcut_weight_avg6, weight_avg7, weight_avg8, weight_avg9), dim=1).view(-1, 128)  # [5, 128]
        weight_avg_3 = torch.cat((weight_avg10, shortcut_weight_avg10, weight_avg11, weight_avg12, weight_avg13), dim=1).view(-1, 256)  # [5, 256]
        weight_avg_4 = torch.cat((weight_avg14, shortcut_weight_avg14, weight_avg15, weight_avg16, weight_avg17, weight_avg18), dim=1).view(-1, 512)  # [6, 512]

        return weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4


class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=False,
                            batch_first=True, dropout=0, bidirectional=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
        setattr(self, '_flattened', True)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        return self.sigmoid(out)


class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=False,
                            batch_first=True, dropout=0, bidirectional=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
        setattr(self, '_flattened', True)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        return self.sigmoid(out)


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class Astrocyte_Network_1(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, expansion=1):
        super(Astrocyte_Network_1, self).__init__()
        self.lstm1 = RNN1(input_size, hidden_size, num_layers)
        self.lstm2 = RNN2(input_size, hidden_size, num_layers)

        self.trans1 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

        self.trans2 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 0:
            weight_avg = weight_avg.view(1, 5, 64)
            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            return pr.view(-1, 64)
        else:
            weight_avg = weight_avg.view(1, 1, 5, 64)
            weight_avg1 = self.trans1(weight_avg).view(1, 5, 64)
            weight_avg2 = self.trans2(weight_avg).view(1, 5, 64)
            weight_avg = torch.cat((weight_avg1, weight_avg2), dim=0)  # [2, 5, 64]

            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            feat_list = [pr, pr1]
            return feat_list


class Astrocyte_Network_2(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, expansion=1):
        super(Astrocyte_Network_2, self).__init__()
        self.lstm1 = RNN1(input_size, hidden_size, num_layers)
        self.lstm2 = RNN2(input_size, hidden_size, num_layers)

        self.trans1 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

        self.trans2 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 0:
            weight_avg = weight_avg.view(1, 5, 128)
            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            return pr.view(-1, 128)
        else:
            weight_avg = weight_avg.view(1, 1, 5, 128)
            weight_avg1 = self.trans1(weight_avg).view(1, 5, 128)
            weight_avg2 = self.trans2(weight_avg).view(1, 5, 128)
            weight_avg = torch.cat((weight_avg1, weight_avg2), dim=0)  # [2, 5, 128]

            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            feat_list = [pr, pr1]
            return feat_list


class Astrocyte_Network_3(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2, expansion=1):
        super(Astrocyte_Network_3, self).__init__()
        self.lstm1 = RNN1(input_size, hidden_size, num_layers)
        self.lstm2 = RNN2(input_size, hidden_size, num_layers)

        self.trans1 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

        self.trans2 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 0:
            weight_avg = weight_avg.view(1, 5, 256)
            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            return pr.view(-1, 256)
        else:
            weight_avg = weight_avg.view(1, 1, 5, 256)
            weight_avg1 = self.trans1(weight_avg).view(1, 5, 256)
            weight_avg2 = self.trans2(weight_avg).view(1, 5, 256)
            weight_avg = torch.cat((weight_avg1, weight_avg2), dim=0)  # [2, 5, 256]

            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            feat_list = [pr, pr1]
            return feat_list


class Astrocyte_Network_4(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2, expansion=1):
        super(Astrocyte_Network_4, self).__init__()
        self.lstm1 = RNN1(input_size, hidden_size, num_layers)
        self.lstm2 = RNN2(input_size, hidden_size, num_layers)

        self.trans1 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

        self.trans2 = nn.Sequential(
            SepConv(
                channel_in=1 * expansion,
                channel_out=16 * expansion
            ),
            SepConv(
                channel_in=16 * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 0:
            weight_avg = weight_avg.view(1, 6, 512)
            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            return pr.view(-1, 512)
        else:
            weight_avg = weight_avg.view(1, 1, 6, 512)
            weight_avg1 = self.trans1(weight_avg).view(1, 6, 512)
            weight_avg2 = self.trans2(weight_avg).view(1, 6, 512)
            weight_avg = torch.cat((weight_avg1, weight_avg2), dim=0)  # [2, 6, 512]

            pr1 = self.lstm1(weight_avg)
            pr = self.lstm2(pr1)
            feat_list = [pr, pr1]
            return feat_list