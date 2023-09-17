import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch

from utilses.config import get_args
from utilses.datagenerators import PatientDataset, DirLabDataset
import torch.utils.data as Data
from utilses.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utilses.utilize import save_image, load_landmarks
from layers import SpatialTransformer

def main(args,is_save=False):


    losses = []

    for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
        x = moving.to(args.device).float()






        y = fixed.to(args.device).float()
        landmarks00 = landmarks['landmark_00'].squeeze().cuda()
        landmarks50 = landmarks['landmark_50'].squeeze().cuda()

        flow =torch.zeros(x.shape, device=x.device) #warped,DVF

        crop_range = args.dirlab_cfg[batch + 1]['crop_range']
        # TRE
        _mean, _std = landmark_loss(flow[0],
                                    landmarks00 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                    landmarks50 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,3).cuda(),
                                    args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                    y.cpu().detach().numpy()[0, 0], is_save)
        losses.append([_mean.item(), _std.item()])
        print('case=%d after warped, TRE=%.2f+-%.2f ' % (batch + 1, _mean.item(), _std.item()))

    mean_total = np.mean(losses, 0)
    mean_tre = mean_total[0]
    mean_std = mean_total[1]
    print('mean TRE=%.2f+-%.2f' % (mean_tre, mean_std))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden-1)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden-1))
    print('If the GPU is available? ' + str(GPU_avai))

    args = get_args()
    device = args.device


    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # pa_fixed_folder = r'D:\xxf\test_patient\fixed'
    # pa_moving_folder = r'D:\xxf\test_patient\moving'
    # f_patient_file_list = sorted(
    #     [os.path.join(pa_fixed_folder, file_name) for file_name in os.listdir(pa_fixed_folder) if
    #      file_name.lower().endswith('.gz')])
    # m_patient_file_list = sorted(
    #     [os.path.join(pa_moving_folder, file_name) for file_name in os.listdir(pa_moving_folder) if
    #      file_name.lower().endswith('.gz')])
    #
    # test_dataset_patient = PatientDataset(moving_files=m_patient_file_list, fixed_files=f_patient_file_list)
    # test_loader_patient = Data.DataLoader(test_dataset_patient, batch_size=args.batch_size, shuffle=False,
    #                                       num_workers=0)

    landmark_list = load_landmarks(args.landmark_dir)
    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')

    f_dir_file_list = sorted([os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])
    test_dataset_dirlab = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                        landmark_files=landmark_list)
    test_loader_dirlab = Data.DataLoader(test_dataset_dirlab, batch_size=args.batch_size, shuffle=False, num_workers=0)


    # test_dirlab(args)
    main(args)

