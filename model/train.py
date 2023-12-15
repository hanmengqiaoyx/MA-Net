import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import progress_bar
from loss import *
from cutout import *
from model import ResNet18
from Astrocyte_Network import Astrocyte_Network_head, Astrocyte_Network_1, Astrocyte_Network_2, Astrocyte_Network_3, Astrocyte_Network_4


parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr_NN', default=1e-1, type=float, help='learning rate')
parser.add_argument('--lr_AN', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--epoch', default=1000, type=int, help='max epoch')
args = parser.parse_args()
gpu = "0,6"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1
train_loss_NN = 1000
train_loss_AN_1 = 1000
train_loss_AN_2 = 1000
train_loss_AN_3 = 1000
train_loss_AN = 1000


# Data Loading
def data_prepare():
    transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../../../../../../../data/hanmq/CIFAR10', train=True, download=False, transform=TwoCropTransform(transform1, transform2))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='../../../../../../../../data/hanmq/CIFAR10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


# Model Loading
def model_prepare():
    NN = ResNet18()
    NN.to(device)
    AN_head = Astrocyte_Network_head()
    AN_head.to(device)
    AN_1 = Astrocyte_Network_1()
    AN_1.to(device)
    AN_2 = Astrocyte_Network_2()
    AN_2.to(device)
    AN_3 = Astrocyte_Network_3()
    AN_3.to(device)
    AN_4 = Astrocyte_Network_4()
    AN_4.to(device)
    NN = torch.nn.DataParallel(NN)
    AN_head = torch.nn.DataParallel(AN_head)
    AN_1 = torch.nn.DataParallel(AN_1)
    AN_2 = torch.nn.DataParallel(AN_2)
    AN_3 = torch.nn.DataParallel(AN_3)
    AN_4 = torch.nn.DataParallel(AN_4)

    optimizer_NN = optim.SGD(NN.parameters(), lr=args.lr_NN, weight_decay=5e-4, momentum=0.9)
    scheduler_NN = optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    optimizer_AN_head = optim.SGD(AN_head.parameters(), lr=args.lr_AN, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_head = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_head, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    optimizer_AN_1 = optim.SGD(AN_1.parameters(), lr=args.lr_AN, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_1, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    optimizer_AN_2 = optim.SGD(AN_2.parameters(), lr=args.lr_AN, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_2, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    optimizer_AN_3 = optim.SGD(AN_3.parameters(), lr=args.lr_AN, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_3, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    optimizer_AN_4 = optim.SGD(AN_4.parameters(), lr=args.lr_AN, weight_decay=5e-4, momentum=0.9)
    scheduler_AN_4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AN_4, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion = nn.CrossEntropyLoss()
    contra_criterion = SupConLoss()
    return NN, AN_head, AN_1, AN_2, AN_3, AN_4, optimizer_AN_head, scheduler_AN_head, optimizer_NN, scheduler_NN, optimizer_AN_1, scheduler_AN_1, optimizer_AN_2, scheduler_AN_2, \
           optimizer_AN_3, scheduler_AN_3, optimizer_AN_4, scheduler_AN_4, criterion, contra_criterion


# Train
def train(epoch, dataloader, NN, AN_head, AN_1, AN_2, AN_3, AN_4, optimizer_NN, optimizer_AN_head, optimizer_AN_1,
          optimizer_AN_2, optimizer_AN_3, optimizer_AN_4, criterion, contra_criterion, vali=True):
    print('\nEpoch: %d' % epoch)
    global train_loss_NN, train_loss_AN_1, train_loss_AN_2, train_loss_AN_3, train_loss_AN
    NN.train()
    AN_head.train()
    AN_1.train()
    AN_2.train()
    AN_3.train()
    AN_4.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    train_loss0 = 0
    correct0 = 0
    total0 = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    for batch_id, (img, labels) in enumerate(dataloader):
        # if batch_id < (256 / args.batch_size):
            batch = labels.size(0)
            inputs = torch.cat([img[0], img[1]], dim=0)
            inputs, labels = inputs.to(device), labels.to(device)
            if epoch < 6:
                num_id += 1
                pattern = 0
                i = 0
                outputs, feat_list = NN(inputs, 0, 0, 0, 0, pattern, i)
                outputs = outputs[:batch]
                loss = criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_NN.zero_grad()
                loss.backward()
                optimizer_NN.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss / num_id, 100. * correct / total, correct, total))
            elif epoch >= 6:
                inputs_0 = img[0]
                inputs_0, labels = inputs_0.to(device), labels.to(device)
                num_id += 1
                for params_nn in NN.parameters():
                    params_nn.requires_grad = False
                for params_AN_head in AN_head.parameters():
                    params_AN_head.requires_grad = True
                for params_an1 in AN_1.parameters():
                    params_an1.requires_grad = True
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = False
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = False
                for params_an4 in AN_4.parameters():
                    params_an4.requires_grad = False
                pattern = 1
                i = 1
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = NN(inputs_0, 0, 0, 0, 0, pattern, i)
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = AN_head(weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4)
                feat_list_1 = AN_1(weight_avg_1, 1)
                pr_2 = AN_2(weight_avg_2, 0)
                pr_3 = AN_3(weight_avg_3, 0)
                pr_4 = AN_4(weight_avg_4, 0)
                pattern = 2
                outputs, feat_list = NN(inputs_0, feat_list_1, pr_2, pr_3, pr_4, pattern, i)
                outputs = outputs[:batch]
                loss = criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_head.zero_grad()
                optimizer_AN_1.zero_grad()
                loss.backward()
                optimizer_AN_head.step()
                optimizer_AN_1.step()
                train_loss1 += loss.item()
                for params_AN_head in AN_head.parameters():
                    params_AN_head.requires_grad = False
                for params_an1 in AN_1.parameters():
                    params_an1.requires_grad = False
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = True
                pattern = 1
                i = 2
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = NN(inputs_0, 0, 0, 0, 0, pattern, i)
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = AN_head(weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4)
                pr_1 = AN_1(weight_avg_1, 0)
                feat_list_2 = AN_2(weight_avg_2, 1)
                pr_3 = AN_3(weight_avg_3, 0)
                pr_4 = AN_4(weight_avg_4, 0)
                pattern = 2
                outputs, feat_list = NN(inputs_0, pr_1, feat_list_2, pr_3, pr_4, pattern, i)
                outputs = outputs[:batch]
                loss = criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_head.zero_grad()
                optimizer_AN_2.zero_grad()
                loss.backward()
                optimizer_AN_head.step()
                optimizer_AN_2.step()
                train_loss2 += loss.item()
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = False
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = True
                pattern = 1
                i = 3
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = NN(inputs_0, 0, 0, 0, 0, pattern, i)
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = AN_head(weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4)
                pr_1 = AN_1(weight_avg_1, 0)
                pr_2 = AN_2(weight_avg_2, 0)
                feat_list_3 = AN_3(weight_avg_3, 1)
                pr_4 = AN_4(weight_avg_4, 0)
                pattern = 2
                outputs, feat_list = NN(inputs_0, pr_1, pr_2, feat_list_3, pr_4, pattern, i)
                outputs = outputs[:batch]
                loss = criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_head.zero_grad()
                optimizer_AN_3.zero_grad()
                loss.backward()
                optimizer_AN_head.step()
                optimizer_AN_3.step()
                train_loss3 += loss.item()
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = False
                for params_an4 in AN_4.parameters():
                    params_an4.requires_grad = True
                pattern = 1
                i = 4
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = NN(inputs_0, 0, 0, 0, 0, pattern, i)
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = AN_head(weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4)
                pr_1 = AN_1(weight_avg_1, 0)
                pr_2 = AN_2(weight_avg_2, 0)
                pr_3 = AN_3(weight_avg_3, 0)
                feat_list_4 = AN_4(weight_avg_4, 1)
                pattern = 2
                outputs, feat_list = NN(inputs_0, pr_1, pr_2, pr_3, feat_list_4, pattern, i)
                outputs = outputs[:batch]
                loss = criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_AN_head.zero_grad()
                optimizer_AN_4.zero_grad()
                loss.backward()
                optimizer_AN_head.step()
                optimizer_AN_4.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss / num_id, 100. * correct / total, correct, total))

                for params_nn in NN.parameters():
                    params_nn.requires_grad = True
                for params_AN_head in AN_head.parameters():
                    params_AN_head.requires_grad = False
                for params_an1 in AN_1.parameters():
                    params_an1.requires_grad = False
                for params_an2 in AN_2.parameters():
                    params_an2.requires_grad = False
                for params_an3 in AN_3.parameters():
                    params_an3.requires_grad = False
                for params_an4 in AN_4.parameters():
                    params_an4.requires_grad = False
                pattern = 1
                i = 0
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = NN(inputs, 0, 0, 0, 0, pattern, i)
                weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = AN_head(weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4)
                pr_1 = AN_1(weight_avg_1, 0)
                pr_2 = AN_2(weight_avg_2, 0)
                pr_3 = AN_3(weight_avg_3, 0)
                pr_4 = AN_4(weight_avg_4, 0)
                pattern = 3
                outputs, feat_list = NN(inputs, pr_1, pr_2, pr_3, pr_4, pattern, i)
                outputs = outputs[:batch]
                loss = criterion(outputs, labels.long())
                contra_loss = 0
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [batch, batch], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    contra_loss += contra_criterion(features, labels) * 1e-1
                loss += contra_loss
                optimizer_NN.zero_grad()
                loss.backward()
                optimizer_NN.step()
                train_loss0 += loss.item()
                _, predicted = outputs.max(1)
                total0 += labels.size(0)
                correct0 += predicted.eq(labels).sum().item()
                progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                             % (train_loss0 / num_id, 100. * correct0 / total0, correct0, total0))
        # else:
        #     print('End of the train')
        #     break
    if vali is True:
        if epoch >= 6:
            train_loss_AN_1 = train_loss1 / num_id
            train_loss_AN_2 = train_loss2 / num_id
            train_loss_AN_3 = train_loss3 / num_id
            train_loss_AN = train_loss / num_id
            train_loss_NN = train_loss0 / num_id
    if epoch < 6:
        return train_loss / num_id, 100. * correct / total, train_loss / num_id, 100. * correct / total
    if epoch >= 6:
        return train_loss / num_id, 100. * correct / total, train_loss0 / num_id, 100. * correct0 / total


# Test
def test(epoch, dataloader, NN, AN_head, AN_1, AN_2, AN_3, AN_4, criterion):
    NN.eval()
    AN_head.eval()
    AN_1.eval()
    AN_2.eval()
    AN_3.eval()
    AN_4.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(dataloader):
            # if batch_id < (256 / args.batch_size):
                inputs, labels = inputs.to(device), labels.to(device)
                if epoch < 6:
                    num_id += 1
                    pattern = 0
                    i = 0
                    outputs = NN(inputs, 0, 0, 0, 0, pattern, i)
                    loss = criterion(outputs, labels.long())

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss / num_id, 100. * correct / total, correct, total))
                elif epoch >= 6:
                    num_id += 1
                    pattern = 1
                    i = 0
                    weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = NN(inputs, 0, 0, 0, 0, pattern, i)
                    weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4 = AN_head(weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4)
                    pr_1 = AN_1(weight_avg_1, 0)
                    pr_2 = AN_2(weight_avg_2, 0)
                    pr_3 = AN_3(weight_avg_3, 0)
                    pr_4 = AN_4(weight_avg_4, 0)
                    pattern = 3
                    outputs = NN(inputs, pr_1, pr_2, pr_3, pr_4, pattern, i)
                    loss = criterion(outputs, labels.long())

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss / num_id, 100. * correct / total, correct, total))
            # else:
            #     print('End of the test')
            # break
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    print('==> Preparing data..')
    trainloader, testloader = data_prepare()

    print('==> Building model..')
    NN, AN_head, AN_1, AN_2, AN_3, AN_4, optimizer_AN_head, scheduler_AN_head, optimizer_NN, scheduler_NN, optimizer_AN_1, scheduler_AN_1, optimizer_AN_2, \
    scheduler_AN_2, optimizer_AN_3, scheduler_AN_3, optimizer_AN_4, scheduler_AN_4, criterion, contra_criterion = model_prepare()

    print('==> Training..')
    train_AN_loss_lst, train_AN_acc_lst, train_NN_loss_lst, train_NN_acc_lst, test_loss_lst, test_acc_lst = [], [], [], [], [], []
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train_AN_loss, train_AN_acc, train_NN_loss, train_NN_acc = train(epoch, trainloader, NN, AN_head, AN_1, AN_2, AN_3, AN_4, optimizer_NN, optimizer_AN_head,
                                      optimizer_AN_1, optimizer_AN_2, optimizer_AN_3, optimizer_AN_4, criterion, contra_criterion)
        test_loss, test_acc = test(epoch, testloader, NN, AN_head, AN_1, AN_2, AN_3, AN_4, criterion)
        if epoch < 6:
            pass
        elif epoch >= 6:
            scheduler_AN_1.step(train_loss_AN_1)
            scheduler_AN_2.step(train_loss_AN_2)
            scheduler_AN_3.step(train_loss_AN_3)
            scheduler_AN_4.step(train_loss_AN)
            scheduler_NN.step(train_loss_NN)
            lr_AN_1 = optimizer_AN_1.param_groups[0]['lr']
            lr_AN_2 = optimizer_AN_2.param_groups[0]['lr']
            lr_AN_3 = optimizer_AN_3.param_groups[0]['lr']
            lr_AN_4 = optimizer_AN_4.param_groups[0]['lr']
            lr_NN = optimizer_NN.param_groups[0]['lr']
            train_AN_loss_lst.append(train_AN_loss)
            train_AN_acc_lst.append(train_AN_acc)
            train_NN_loss_lst.append(train_NN_loss)
            train_NN_acc_lst.append(train_NN_acc)
            test_loss_lst.append(test_loss)
            test_acc_lst.append(test_acc)

            print('Saving:')
            plt.figure(num=1, dpi=200)
            plt.subplot(2, 3, 1)
            picture1, = plt.plot(np.arange(0, len(train_AN_loss_lst)), train_AN_loss_lst, color='red', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture1], labels=['train_AN_loss'], loc='best')
            plt.subplot(2, 3, 2)
            picture2, = plt.plot(np.arange(0, len(train_AN_acc_lst)), train_AN_acc_lst, color='red', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture2], labels=['train_AN_acc'], loc='best')
            plt.subplot(2, 3, 3)
            picture3, = plt.plot(np.arange(0, len(train_NN_loss_lst)), train_NN_loss_lst, color='blue', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture3], labels=['train_NN_loss'], loc='best')
            plt.subplot(2, 3, 4)
            picture4, = plt.plot(np.arange(0, len(train_NN_acc_lst)), train_NN_acc_lst, color='blue', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture4], labels=['train_NN_acc'], loc='best')
            plt.subplot(2, 3, 5)
            picture3, = plt.plot(np.arange(0, len(test_loss_lst)), test_loss_lst, color='green', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture3], labels=['test_loss'], loc='best')
            plt.subplot(2, 3, 6)
            picture4, = plt.plot(np.arange(0, len(test_acc_lst)), test_acc_lst, color='green', linewidth=1.0, linestyle='-')
            plt.legend(handles=[picture4], labels=['test_acc'], loc='best')
            plt.savefig('./AstroNet.jpg')

            if lr_NN > 5e-5 or lr_AN_1 > 5e-5 or lr_AN_2 > 5e-5 or lr_AN_3 > 5e-5 or lr_AN_4 > 5e-5:
                print('Saving:')
                state1 = {
                    'net': NN.state_dict()
                }
                state2 = {
                    'net': AN_head.state_dict()
                }
                state3 = {
                    'net': AN_1.state_dict()
                }
                state4 = {
                    'net': AN_2.state_dict()
                }
                state5 = {
                    'net': AN_3.state_dict()
                }
                state6 = {
                    'net': AN_4.state_dict()
                }
                if not os.path.isdir('Model'):
                    os.mkdir('Model')
                torch.save(state1, './Model/NN''.t7')
                torch.save(state2, './Model/AN_head''.t7')
                torch.save(state3, './Model/AN_1''.t7')
                torch.save(state4, './Model/AN_2''.t7')
                torch.save(state5, './Model/AN_3''.t7')
                torch.save(state6, './Model/AN_4''.t7')
                acc = open('./AstroNet.txt', 'w')
                acc.write(str(test_acc))
                acc.close()
            else:
                break