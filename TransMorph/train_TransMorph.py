from torch.utils.tensorboard import SummaryWriter
import os, glob, losses, utils
from TransMorph.losses import Grad3d
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
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

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

from utilses.config import get_args
from utilses.datagenerators import PatientDataset, DirLabDataset
import torch.utils.data as Data
from utilses.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utilses.utilize import save_image, load_landmarks
from layers import SpatialTransformer


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
    # train_dir = 'D:/DATA/JHUBrain/Train/'
    # val_dir = 'D:/DATA/JHUBrain/Val/'
    train_dir = r'D:\datasets\dirlab_160_ small_small\dirlab_160_train'
    train_fixed_folder = os.path.join(train_dir, 'fixed')
    train_moving_folder = os.path.join(train_dir, 'moving')
    train_f_img_file_list = sorted(
        [os.path.join(train_fixed_folder, file_name) for file_name in os.listdir(train_fixed_folder) if
         file_name.lower().endswith('.gz')])
    train_m_img_file_list = sorted(
        [os.path.join(train_moving_folder, file_name) for file_name in os.listdir(train_moving_folder) if
         file_name.lower().endswith('.gz')])

    val_dir = r'D:\datasets\dirlab_160_ small_small\dirlab_160_val'
    val_fixed_folder = os.path.join(val_dir, 'fixed')
    val_moving_folder = os.path.join(val_dir, 'moving')
    val_f_img_file_list = sorted(
        [os.path.join(val_fixed_folder, file_name) for file_name in os.listdir(val_fixed_folder) if
         file_name.lower().endswith('.gz')])
    val_m_img_file_list = sorted(
        [os.path.join(val_moving_folder, file_name) for file_name in os.listdir(val_moving_folder) if
         file_name.lower().endswith('.gz')])

    landmark_list = load_landmarks(args.landmark_dir)
    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')

    f_dir_file_list = sorted(
        [os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
         file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])
    test_dataset_dirlab = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                        landmark_files=landmark_list)
    test_loader_dirlab = Data.DataLoader(test_dataset_dirlab, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0)

    weights = [1, 0.02]  # loss weights

    lr = 1e-4 # learning rate
    epoch_start = 0
    max_epoch = 500  # max traning epoch
    cont_training = True  # if continue training

    image_loss_func_NCC = NCC_new(win=args.win_size)

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    Initialize training 
    '''

    train_dataset = Dataset(moving_files=train_m_img_file_list, fixed_files=train_f_img_file_list)
    val_dataset = Dataset(moving_files=val_m_img_file_list, fixed_files=val_f_img_file_list)

    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = image_loss_func_NCC
    criterions = [criterion]
    criterions += [Grad3d(penalty='l2')]  # criterions 被定义为一个列表，首先包含了一个均方误差损失函数 nn.MSELoss()，然后又添加了一个基于梯度的正则化损失函数 losses.Grad3d()。这个列表中的损失函数将在模型训练过程中被同时使用，以帮助模型学习更好的特征表示和更稳定的模型。

    stop_criterion = StopCriterion()

    writer = SummaryWriter(log_dir='logs/')
    '''
       If continue from previous training
       '''
    # if cont_training:
    #     # checkpoint = r'D:\TransMorph_Transformer_for_Medical_Image_Registration_main\TransMorph\model\2023-05-07-15-12-52_TM__133_1.0887best.pth'
    #     # model.load_state_dict(torch.load(checkpoint)['model'])
    #     # optimizer.load_state_dict(torch.load(checkpoint)['o'
    #     #                                                  ''
    #     #                                                  'ptimizer'])
    #     # model_path = "../model/2023-07-09-18-24-26_TM__048_-0.6769best.pth"
    #     model_path = r'D:\code\TransMorph_Transformer_for_Medical_Image_Registration_main\TransMorph\model\2023-07-16-22-02-22_TM__051_-0.6649.pth'
    #     print("Loading weight: ", model_path)
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     epoch_start = 52

    # writer = SummaryWriter(log_dir = save_dir)
    best_loss = 99.
    for epoch in range(epoch_start, max_epoch):
        print('\n Training Starts')




        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        model.train()
        # for data in train_loader:
        for batch, (moving, fixed) in enumerate(train_loader):
            idx += 1

            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            x = moving[0].to(device).float()
            y = fixed[0].to(device).float()

            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # del x_in
            # del output
            #
            # # flip fixed and moving images
            # loss = 0
            # x_in = torch.cat((y, x), dim=1)
            # output = model(x_in)
            # for n, loss_function in enumerate(criterions):
            #     curr_loss = loss_function(output[n], x) * weights[n]
            #     loss_vals[n] += curr_loss
            #     loss += curr_loss
            # loss_all.update(loss.item(), y.numel())
            # # compute gradient and do SGD step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            loss_all.update(loss.item())
            sys.stdout.write(
                "\r" + 'epoch:"{0}"step:batch "{1}:{2}" -> training loss "{3:.6f}" - sim "{4:.6f}"  -reg "{5:.6f}" '.format(
                    epoch, idx, len(train_loader), loss.item(), loss_vals[0].item() , loss_vals[1].item() ))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("TM, epoch: %d  iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                epoch, idx, len(train_loader), loss.item(), loss_vals[0].item() , loss_vals[1].item() ))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Train: Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''

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
                output = model(x_in)  # [warped,DVF]

                # loss_Jacobian = loss_Jdet(output[1].permute(0, 2, 3, 4, 1), grid)
                loss_Jacobian = criterions[1](output[1], y)
                ncc_loss_ori = image_loss_func_NCC(output[0], y)

                mse_loss = MSE(output[0], y)
                loss_sum = ncc_loss_ori + weights[1] * loss_Jacobian
                losses.append([ncc_loss_ori.item(), mse_loss.item(), loss_Jacobian.item(), loss_sum.item()])

                """
                这段代码使用另一个空间变换模型 reg_model_bilin 对一个网格图像 grid_img 进行空间变换操作，并将变换后的结果赋值给变量 def_grid。
                具体地，代码将 grid_img 和 output[1]（表示光流场信息）分别转移到 GPU 上，并将它们作为输入传递给 reg_model_bilin 模型。reg_model_bilin 模型会根据给定的光流场信息对输入的网格图像进行空间变换操作，从而得到变换后的网格图像。
                需要注意的是，这里使用的是另一个空间变换模型 reg_model_bilin，而不是之前提到的 reg_model。reg_model_bilin 模型与 reg_model 模型的主要区别是，在进行空间变换操作时，reg_model_bilin 模型使用的是双线性插值方法，而 reg_model 模型使用的是最近邻插值方法。双线性插值方法可以产生更平滑的变换结果，但计算代价也更高。
                """
                # def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                # def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                # dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                # eval_dsc.update(dsc.item(), x.size(0))

                # mse_loss = MSE(output[0], x)

                # ncc_loss_ori=loss_similarity(output[0], x)
                # loss_Jacobian = neg_Jdet_loss(flow.permute(0, 2, 3, 4, 1), grid)
                # val_Loss[0]=torch.mean(torch.mean(val_Loss[0], dim=0))#range (expected to be in range of [-1, 0], but got 1)
                # eval_MSE.update(val_Loss[0].item(), x.size(0))
                #
                # print('Iter {} of {} eval_MSE.avg {:.4f}'.format(batch,len(val_loader),eval_MSE.avg))

            mean_loss = np.mean(losses, 0)
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = mean_loss
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
            "one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                loss_all.avg, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        if (epoch + 1) % 1 == 0:

            with torch.no_grad():

                losses = []
                model.eval()
                # for batch, (moving, fixed, img_name) in enumerate(test_loader_dirlab):
                for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
                    x = moving.to(args.device).float()
                    y = fixed.to(args.device).float()
                    landmarks00 = landmarks['landmark_00'].squeeze().cuda()
                    landmarks50 = landmarks['landmark_50'].squeeze().cuda()
                    x_in = torch.cat((x, y), dim=1)
                    flow = model(x_in, False)  # warped,DVF

                    crop_range = args.dirlab_cfg[batch + 1]['crop_range']
                    # TRE
                    _mean, _std = landmark_loss(flow[0],
                                                landmarks00 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                                landmarks50 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                                args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                                y.cpu().detach().numpy()[0, 0], False)
                    losses.append([_mean.item(), _std.item()])
                    # print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                    #     batch + 1, _mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()))

                    # # flip moving and fixed images
                    # y_in = torch.cat((y, x), dim=1)
                    # flow = model(y_in, False)
                    #
                    # crop_range = args.dirlab_cfg[batch + 1]['crop_range']
                    # # TRE
                    # _mean, _std = landmark_loss(flow[0], landmarks50 - torch.tensor(
                    #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                    #                             landmarks00 - torch.tensor(
                    #                                 [crop_range[2].start, crop_range[1].start,
                    #                                  crop_range[0].start]).view(1,
                    #                                                             3).cuda(),
                    #                             args.dirlab_cfg[batch + 1]['pixel_spacing'],
                    #                             y.cpu().detach().numpy()[0, 0], False)
                    # losses.append([_mean.item(), _std.item()])
                    # # print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                    # #     batch + 1, _mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()))

                mean_total = np.mean(losses, 0)
                mean_tre = mean_total[0]
                mean_std = mean_total[1]

                print('mean TRE=%.2f+-%.2f' % (
                    mean_tre, mean_std))

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
    # train_time = '2023-07-16-22-02-22'
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
