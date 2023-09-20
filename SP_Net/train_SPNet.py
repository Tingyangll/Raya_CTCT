from torch.utils.tensorboard import SummaryWriter
import os, glob, utils
from TransMorph.losses import Grad3d
import sys

import numpy as np
import torch
from torch import optim
from torch import nn

import matplotlib.pyplot as plt
from natsort import natsorted

from CPCA.CPCANet import CPCANet
from CPCA.CPCA_config import ACDC_224


from datagenerators import Dataset
import torch.nn.functional as F
import logging
import time
from utilses.config import get_args
from layers import SpatialTransformer

args = get_args()

# from utilses.losses import NCC as NCC_new
from losses import ncc_loss, gradient_loss
from model import SPnet
from utilses.scheduler import StopCriterion
from utilses.utilize import save_model
from utilses.datagenerators import PatientDataset, DirLabDataset
import torch.utils.data as Data
from utilses.metric import MSE, SSIM,jacobian_determinant, landmark_loss
from utilses.utilize import save_image, load_landmarks



def make_dirs():
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # if not os.path.exists(args.result_dir):
    #     os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

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
    train_dir = r'D:\datasets\datasets_raya\dirlab_train_val_144_144_96\dirlab_train_144_144_96'
    train_fixed_folder = os.path.join(train_dir, 'fixed')
    train_moving_folder = os.path.join(train_dir, 'moving')
    train_f_img_file_list = sorted(
        [os.path.join(train_fixed_folder, file_name) for file_name in os.listdir(train_fixed_folder) if
         file_name.lower().endswith('.gz')])
    train_m_img_file_list = sorted(
        [os.path.join(train_moving_folder, file_name) for file_name in os.listdir(train_moving_folder) if
         file_name.lower().endswith('.gz')])

    val_dir = r'D:\datasets\datasets_raya\dirlab_train_val_144_144_96\dirlab_val_144_144_96'
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

    lr =1e-4   # learning rate
    epoch_start = 0
    max_epoch = 2000  # max traning epoch
    cont_training = True  # if continue training
    STN = SpatialTransformer()
    # image_loss_func_NCC = NCC_new(win=args.win_size)
    '''
    Initialize model
    '''
    #[160,160,160],[96,144,144]
    model = SPnet([160,160,160])
    model.cuda()

    '''
    Initialize training 
    '''

    train_dataset = Dataset(moving_files=train_m_img_file_list, fixed_files=train_f_img_file_list)
    val_dataset = Dataset(moving_files=val_m_img_file_list, fixed_files=val_f_img_file_list)

    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = image_loss_func_NCC
    # criterions = [criterion]
    # criterions += [Grad3d(penalty='l2')]

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
    #     model_path = r'D:\code\TransMorph_Transformer_for_Medical_Image_Registration_main\CPCA\model\2023-09-15-10-46-41_CPCA__111_-0.6167.pth'
    #     print("Loading weight: ", model_path)
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     epoch_start =112

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
        for batch, (moving, fixed) in enumerate(train_loader):
            idx += 1
            moving_ori = moving
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            input_moving = moving[0].to(device).float()
            input_fixed = fixed[0].to(device).float()
            # utilize.show_slice(x.cpu().detach().numpy())
            # utilize.show_slice(y.cpu().detach().numpy())
            ([wraped, wraped1, wraped2, wraped3], [fixed, fixed1, fixed2, fixed3],
             [sym_wraped, sym_wraped1, sym_wraped2, sym_wraped3], [moving, moving1, moving2, moving3],
             [flow0, flow1, flow2, flow3], [sym_flow0, sym_flow1, sym_flow2, sym_flow3],
             [vec0, vec1, vec2, vec3]) = model(input_fixed, input_moving)

            sim_loss0, sim_loss1, sim_loss2, sim_loss3 = (ncc_loss(fixed, wraped),
                                                          ncc_loss(fixed1, wraped1, [7, 7, 7]),
                                                          ncc_loss(fixed2, wraped2, [5, 5, 5]),
                                                          ncc_loss(fixed3, wraped3, [3, 3, 3]))
            sym_sim_loss0, sym_sim_loss1, sym_sim_loss2, sym_sim_loss3 = (ncc_loss(moving, sym_wraped),
                                                                          ncc_loss(moving1, sym_wraped1, [7, 7, 7]),
                                                                          ncc_loss(moving2, sym_wraped2, [5, 5, 5]),
                                                                          ncc_loss(moving3, sym_wraped3, [3, 3, 3]))
            grad_loss0, grad_loss1, grad_loss2, grad_loss3 = (gradient_loss(vec0),
                                                              gradient_loss(vec1),
                                                              gradient_loss(vec2),
                                                              gradient_loss(vec3))

            loss0 = (sim_loss0 + sym_sim_loss0) / 2.0 + grad_loss0
            loss1 = (sim_loss1 + sym_sim_loss1) / 2.0 + grad_loss1
            loss2 = (sim_loss2 + sym_sim_loss2) / 2.0 + grad_loss2
            loss3 = (sim_loss3 + sym_sim_loss3) / 2.0 + grad_loss3

            loss = loss0 + 0.5 * loss1 + 0.25 * loss2 + 0.125 * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item())
            sys.stdout.write(
                "\r" + 'epoch:"{0}"step:batch "{1}:{2}" -> training loss "{3:.6f}" - sim "{4:.6f}" -sym sim loss "{5: .6f}" -reg "{6:.6f}" '.format(
                    epoch, idx, len(train_loader),  loss0.item(), sim_loss0.item(), sym_sim_loss0.item(), grad_loss0.item() ))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving_ori[1][0]))
            logging.info("SP-Net, epoch: %d  iter: %d batch: %d  loss: %.4f  sim: %.4f sym sim loss: %.4f  grad: %.4f" % (
                epoch, idx, len(train_loader), loss.item(), sim_loss0.item() , sym_sim_loss0.item(), grad_loss0.item() ))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Train: Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''

        eval_DSC = utils.AverageMeter()
        with torch.no_grad():
            # for data in val_loader:
            losses = []
            for batch, (moving, fixed) in enumerate(val_loader):
                model.eval()
                input_moving = moving[0].to(device).float()
                input_fixed = fixed[0].to(device).float()
                # utilize.show_slice(x.cpu().detach().numpy())
                # utilize.show_slice(y.cpu().detach().numpy())
                ([wraped, wraped1, wraped2, wraped3], [fixed, fixed1, fixed2, fixed3],
                 [sym_wraped, sym_wraped1, sym_wraped2, sym_wraped3], [moving, moving1, moving2, moving3],
                 [flow0, flow1, flow2, flow3], [sym_flow0, sym_flow1, sym_flow2, sym_flow3],
                 [vec0, vec1, vec2, vec3]) = model(input_fixed, input_moving)

                sim_loss0, sim_loss1, sim_loss2, sim_loss3 = (ncc_loss(fixed, wraped),
                                                              ncc_loss(fixed1, wraped1, [7, 7, 7]),
                                                              ncc_loss(fixed2, wraped2, [5, 5, 5]),
                                                              ncc_loss(fixed3, wraped3, [3, 3, 3]))
                sym_sim_loss0, sym_sim_loss1, sym_sim_loss2, sym_sim_loss3 = (ncc_loss(moving, sym_wraped),
                                                                              ncc_loss(moving1, sym_wraped1, [7, 7, 7]),
                                                                              ncc_loss(moving2, sym_wraped2, [5, 5, 5]),
                                                                              ncc_loss(moving3, sym_wraped3, [3, 3, 3]))
                grad_loss0, grad_loss1, grad_loss2, grad_loss3 = (gradient_loss(vec0),
                                                                  gradient_loss(vec1),
                                                                  gradient_loss(vec2),
                                                                  gradient_loss(vec3))

                loss0 = (sim_loss0 + sym_sim_loss0) / 2.0 + grad_loss0
                loss1 = (sim_loss1 + sym_sim_loss1) / 2.0 + grad_loss1
                loss2 = (sim_loss2 + sym_sim_loss2) / 2.0 + grad_loss2
                loss3 = (sim_loss3 + sym_sim_loss3) / 2.0 + grad_loss3

                loss = loss0 + 0.5 * loss1 + 0.25 * loss2 + 0.125 * loss3

                loss_Jacobian = jacobian_determinant(flow0[0])
                sym_loss_Jacobian = jacobian_determinant(sym_flow0[0])
                mse_loss = MSE(wraped[0], input_fixed)
                sym_mse_loss = MSE(sym_wraped[0], input_moving)
                losses.append([sim_loss0.cpu(), sym_sim_loss0.cpu(), mse_loss.item(), sym_mse_loss.item(), loss_Jacobian, sym_loss_Jacobian, loss.cpu()])

            # mean_loss = torch.mean(losses)
            mean_loss = np.mean(losses, 0)
            val_ncc_loss, val_sym_ncc_loss, val_mse_loss, val_sym_mse_loss, val_jac_loss, val_sym_jac_loss, val_total_loss = mean_loss
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}.pth'.format( val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        print("one epoch pass. train loss %.4f . val sym_ncc loss %.4f . val ncc loss %.4f . val mse loss %.4f .  val sym_mse loss %.4f . val_jac_loss %.6f . val_sym_jac_loss %.6f . val_total loss %.4f" % (
                loss_all.avg, val_ncc_loss, val_sym_ncc_loss, val_mse_loss, val_sym_mse_loss, val_jac_loss, sym_loss_Jacobian,  val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)
        if stop_criterion.stop():
            break

        if (epoch + 1) % 1 == 0:

            with torch.no_grad():
                is_save = True
                losses = []
                model.eval()
                # for batch, (moving, fixed, img_name) in enumerate(test_loader_dirlab):
                for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
                    input_moving = moving.to(args.device).float()
                    input_fixed = fixed.to(args.device).float()
                    landmarks00 = landmarks['landmark_00'].squeeze().cuda()
                    landmarks50 = landmarks['landmark_50'].squeeze().cuda()
                    ([wraped, wraped1, wraped2, wraped3], [fixed, fixed1, fixed2, fixed3],
                     [sym_wraped, sym_wraped1, sym_wraped2, sym_wraped3], [moving, moving1, moving2, moving3],
                     [flow0, flow1, flow2, flow3], [sym_flow0, sym_flow1, sym_flow2, sym_flow3],
                     [vec0, vec1, vec2, vec3]) = model(input_fixed, input_moving)
                    x_def = STN(input_fixed, flow0)

                    crop_range = args.dirlab_cfg[batch + 1]['crop_range']
                    # TRE
                    _mean, _std = landmark_loss(flow0[0],
                                                landmarks00 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                                landmarks50 - torch.tensor([crop_range[2].start, crop_range[1].start,crop_range[0].start]).view(1, 3).cuda(),
                                                args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                                input_fixed.cpu().detach().numpy()[0, 0], False)
                    losses.append([_mean.item(), _std.item()])

                mean_total = np.mean(losses, 0)
                mean_tre = mean_total[0]
                mean_std = mean_total[1]

                print('mean TRE=%.2f+-%.2f' % ( mean_tre, mean_std))

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
    # train_time = '2023-09-15-10-46-41'
    model_name = "{}_CPCA_".format(train_time)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])
    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


    main()
