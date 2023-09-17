from torch.utils.tensorboard import SummaryWriter
import os, glob, losses, utils
import losses as losseses
import sys
# sys.path.append(r'D:/code/TransMorph_Transformer_for_Medical_Image_Registration-main/utils')
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_Bayes import CONFIGS as CONFIGS_TM
import models.TransMorph_Bayes as TransMorph_Bayes

from datagenerators import Dataset
import torch.utils.data as Data
from metric import MSE
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
import torch.nn.functional as F
from utilses.losses import neg_Jdet_loss
import logging
import time
from utilses.config import get_args

args = get_args()
from utilses.losses import NCC as NCC_new
from utilses.scheduler import StopCriterion
from utilses.utilize import save_model


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


# class Logger(object):
#     def __init__(self, save_dir):
#         self.terminal = sys.stdout
#         self.log = open(save_dir+"logfile.log", "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def main():
    batch_size = 1

    train_dir = 'D:/datasets/train_val_test/train_160_small'
    train_fixed_folder = os.path.join(train_dir, 'fixed')
    train_moving_folder = os.path.join(train_dir, 'moving')
    train_f_img_file_list = sorted(
        [os.path.join(train_fixed_folder, file_name) for file_name in os.listdir(train_fixed_folder) if
         file_name.lower().endswith('.gz')])
    train_m_img_file_list = sorted(
        [os.path.join(train_moving_folder, file_name) for file_name in os.listdir(train_moving_folder) if
         file_name.lower().endswith('.gz')])

    val_dir = 'D:\datasets/train_val_test/val_160_small'
    val_fixed_folder = os.path.join(val_dir, 'fixed')
    val_moving_folder = os.path.join(val_dir, 'moving')
    val_f_img_file_list = sorted(
        [os.path.join(val_fixed_folder, file_name) for file_name in os.listdir(val_fixed_folder) if
         file_name.lower().endswith('.gz')])
    val_m_img_file_list = sorted(
        [os.path.join(val_moving_folder, file_name) for file_name in os.listdir(val_moving_folder) if
         file_name.lower().endswith('.gz')])

    weights = [1, 0.02]  # loss weights


    lr = 1e-6  # learning rate
    epoch_start = 0
    max_epoch = 500  # max traning epoch
    cont_training = False  # if continue training

    image_loss_func_NCC = NCC_new(win=args.win_size)
    # loss_Jdet = neg_Jdet_loss
    imgshape = (160, 160, 160)
    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorphBayes']
    model = TransMorph_Bayes.TransMorphBayes(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.cuda()
    # reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    # reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:

        epoch_start = 394
        # model_dir = 'experiments/'+save_dir

        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)

        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training 
    '''

    train_dataset = Dataset(moving_files=train_m_img_file_list, fixed_files=train_f_img_file_list)
    val_dataset = Dataset(moving_files=val_m_img_file_list, fixed_files=val_f_img_file_list)

    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)


    # optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = NCC_new(win=args.win_size)
    criterions = [criterion]
    criterions += [losseses.Grad3d(
        penalty='l2')]  # criterions 被定义为一个列表，首先包含了一个均方误差损失函数 nn.MSELoss()，然后又添加了一个基于梯度的正则化损失函数 losses.Grad3d()。这个列表中的损失函数将在模型训练过程中被同时使用，以帮助模型学习更好的特征表示和更稳定的模型。
    best_MSE = 0
    stop_criterion = StopCriterion()

    writer = SummaryWriter(log_dir='logs/')
    best_loss = 99.
    # writer = SummaryWriter(log_dir = save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')

        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        loss_total = []
        # for data in train_loader:
        for batch, (moving, fixed) in enumerate(train_loader):
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]
            x = moving[0].to(device).float()
            y = fixed[0].to(device).float()

            x_in = torch.cat((x, y), dim=1)
            output = model(x_in,True)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss

            # print("F_X_Y_norm:  ",F_X_Y_norm)
            # print("grad:  ",grid)
            # loss_Jacobian = loss_Jdet(output[1].permute(0, 2, 3, 4, 1), grid)
            # print("loss_Jacobian: ",loss_Jacobian)

            # loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output

            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in,True)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))
            sys.stdout.write(
                "\r" + 'epoch:"{0}"step:batch "{1}:{2}" -> training loss "{3:.6f}" - sim "{4:.6f}"  -reg "{5:.6f}"'.format(
                    epoch, idx, len(train_loader), loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("TM, epoch: %d  iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                epoch,idx, len(train_loader), loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Train: Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''

        val_Loss = []
        # eval_DSC = utils.AverageMeter()
        with torch.no_grad():
            # for data in val_loader:
            losses = []
            for batch, (moving, fixed) in enumerate(val_loader):
                model.eval()
                # data = [t.cuda() for t in data]
                # x = data[0]
                # y = data[1]
                x = moving[0].to('cuda').float()
                y = fixed[0].to('cuda').float()

                # x_seg = data[2]
                # y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                # grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in,True)  # [warped,DVF]

                # loss_Jacobian = loss_Jdet(output[1].permute(0, 2, 3, 4, 1), grid)
                loss_Jacobian = criterions[1](output[1],y)
                ncc_loss_ori = image_loss_func_NCC(output[0], y)

                mse_loss = MSE(output[0], y)
                loss_sum = ncc_loss_ori + weights[1] * loss_Jacobian
                losses.append([ncc_loss_ori.item(), mse_loss.item(), loss_Jacobian.item(), loss_sum.item()])

            val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = np.mean(losses, 0)
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        # mean_loss = np.mean(np.array(loss_total), 0)
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                loss_all.avg, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        loss_all.reset()
    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()  # GPU_num=1

    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):  # GPU_idx=0
        GPU_name = torch.cuda.get_device_name(GPU_idx)  # GPU_name='NVIDIA GeForce RTX 3060'
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)  # 单卡  使用指定的卡
    GPU_avai = torch.cuda.is_available()  # GPU_avai=true
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    model_dir = 'model'
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_TM_".format(train_time)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])
    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    main()