import os, losses, utils, glob
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_Bayes import CONFIGS as CONFIGS_TM
import models.TransMorph_Bayes as TransMorph_Bayes

from utilses.config import get_args
from utilses.datagenerators import PatientDataset, DirLabDataset
import torch.utils.data as Data
from utilses.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utilses.utilize import save_image, load_landmarks

def main():
    # test_dir = 'D:/DATA/JHUBrain/Test/'
    # model_idx = -1
    # weights = [1, 1]
    # model_folder = 'TransMorphBayes_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    # model_dir = 'experiments/' + model_folder
    # dict = utils.process_label()
    # if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
    #     os.remove('experiments/'+model_folder[:-1]+'.csv')
    # csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    # line = ''
    # for i in range(46):
    #     line = line + ',' + dict[i]
    # csv_writter(line, 'experiments/' + model_folder[:-1])
    # config = CONFIGS_TM['TransMorphBayes']
    # model = TransMorph_Bayes.TransMorphBayes(config)
    model.load_state_dict(torch.load(checkpoint)['model'])
    model.cuda()
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # model.load_state_dict(best_model)
    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.cuda()
    # test_composed = transforms.Compose([trans.Seg_norm(),
    #                                     trans.NumpyType((np.float32, np.int16)),
    #                                     ])
    # test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # eval_dsc_def = utils.AverageMeter()
    # eval_dsc_raw = utils.AverageMeter()
    # eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        losses = []
        model.eval()
        for batch, (moving, fixed, img_name) in enumerate(test_loader_patient):
            x = moving.to(args.device).float()
            y = fixed.to(args.device).float()
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]
            # x_seg = data[2]
            # y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            #outputs, flows = utils.get_mc_preds(model, x_in)
            #flow = torch.mean(torch.cat(flows, dim=0)[:], dim=0, keepdim=True)
            outputs, flows, errs = utils.get_mc_preds_w_errors(model, x_in, y)
            min_err_idx = np.argmin(errs)
            flow = flows[min_err_idx]
            x_def = outputs[min_err_idx]

            ncc = NCC(y.cpu().detach().numpy(), x_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.cpu().detach().numpy())
            _mse = MSE(y, x_def)
            _ssim = SSIM(y.cpu().detach().numpy()[0, 0], x_def.cpu().detach().numpy()[0, 0])
            losses.append([_mse.item(), jac, _ssim.item(), ncc.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.8f, SSIM=%.5f, NCC=%.5f' % (
                batch + 1, _mse.item(), jac, _ssim.item(), ncc.item()))
            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            # line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            #outputs, flows = utils.get_mc_preds(model, y_in)
            #flow = torch.mean(torch.cat(flows, dim=0)[:], dim=0, keepdim=True)
            outputs, flows, errs = utils.get_mc_preds_w_errors(model, y_in, y)
            min_err_idx = np.argmin(errs)
            flow = flows[min_err_idx]
            y_def = outputs[min_err_idx]
            ncc = NCC(x.cpu().detach().numpy(), y_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.cpu().detach().numpy())
            _mse = MSE(x, y_def)
            _ssim = SSIM(x.cpu().detach().numpy()[0, 0], y_def.cpu().detach().numpy()[0, 0])
            losses.append([_mse.item(), jac, _ssim.item(), ncc.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.8f, SSIM=%.5f, NCC=%.5f' % (
                batch + 1, _mse.item(), jac, _ssim.item(), ncc.item()))
            # def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            # tar = x.detach().cpu().numpy()[0, 0, :, :, :]

            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            # line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            # out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            # print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            #
            # dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            # dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1
    mean_total = np.mean(losses, 0)
    mean_mse = mean_total[0]
    mean_jac = mean_total[1]
    mean_ssim = mean_total[2]
    mean_ncc = mean_total[3]
    print('mean SSIM=%.5f Jac=%.8f MSE=%.5f NCC=%.5f' % (mean_ssim, mean_jac, mean_mse, mean_ncc))
        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    args = get_args()
    device = args.device

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    pa_fixed_folder = r'D:\datasets\train_val_test\test_ori\fixed'
    pa_moving_folder = r'D:\datasets\train_val_test\test_ori\moving'
    f_patient_file_list = sorted(
        [os.path.join(pa_fixed_folder, file_name) for file_name in os.listdir(pa_fixed_folder) if
         file_name.lower().endswith('.gz')])
    m_patient_file_list = sorted(
        [os.path.join(pa_moving_folder, file_name) for file_name in os.listdir(pa_moving_folder) if
         file_name.lower().endswith('.gz')])

    test_dataset_patient = PatientDataset(moving_files=m_patient_file_list, fixed_files=f_patient_file_list)
    test_loader_patient = Data.DataLoader(test_dataset_patient, batch_size=args.batch_size, shuffle=False,
                                          num_workers=0)

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

    prefix = '2023-05-08-11-31-16'
    model_dir = args.checkpoint_path

    config = CONFIGS_TM['TransMorphBayes']
    model = TransMorph_Bayes.TransMorphBayes(config)

    if args.checkpoint_name is not None:
        # test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), True)
        main(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            # test_dirlab(args, checkpoint)
            main(args, checkpoint)

