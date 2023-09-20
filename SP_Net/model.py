import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def conv(in_planes,out_planes,kernel_size=3,stride=1,padding=1):
    return nn.Sequential(
            nn.Conv3d(in_planes,out_planes,kernel_size=kernel_size,
                        stride=stride,padding=padding),
            nn.LeakyReLU(0.1)
    )

def predict_flow(in_planes):
    flow = nn.Conv3d(in_planes,3,kernel_size=3,stride=1,padding=1)
    nd = Normal(0, 1e-5)
    flow.weight = nn.Parameter(nd.sample(flow.weight.shape))
    flow.bias = nn.Parameter(torch.zeros(flow.bias.shape))
    return flow


def deconv(in_planes,out_planes,kernel_size=4,stride=2,padding=1):
    return nn.ConvTranspose3d(in_planes,out_planes,kernel_size,stride,padding)


# class SpatialTransformer(nn.Module):
#     """
#     [SpatialTransformer] represesents a spatial transformation block
#     that uses the output from the UNet to preform an grid_sample
#     https://pytorch.org/docs/stable/nn.functional.html#grid-sample
#     """
#     def __init__(self, size, mode='bilinear'):
#         """
#         Instiatiate the block
#             :param size: size of input to the spatial transformer block
#             :param mode: method of interpolation for grid_sampler
#         """
#         super(SpatialTransformer, self).__init__()
#
#         # Create sampling grid
#         vectors = [ torch.arange(0, s) for s in size ]
#         grids = torch.meshgrid(vectors)
#         grid  = torch.stack(grids) # y, x, z
#         grid  = torch.unsqueeze(grid, 0)  #add batch
#         grid = grid.type(torch.FloatTensor)
#         #self.register_buffer('grid', grid)
#
#         self.mode = mode
#         self.grid = grid
#
#     def forward(self, src, flow):
#         """
#         Push the src and flow through the spatial transform block
#             :param src: the original moving image
#             :param flow: the output from the U-Net
#         """
#         device = src.device
#         self.grid = self.grid.to(device)
#         new_locs = self.grid + flow
#
#         shape = flow.shape[2:]
#
#         # Need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
#
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1,0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2,1,0]]
#
#         return F.grid_sample(src, new_locs, mode=self.mode)

class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image

    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim
        self.grid_dict = {}
        self.norm_coeff_dict = {}

    def forward(self, input_image, flow):
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, 2, h, w) or (n, 3, d, h, w)

        return:
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        img_shape = input_image.shape[2:]
        if img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            grids = torch.meshgrid([torch.arange(0, s) for s in img_shape])
            # grid = torch.stack(grids)
            grid = torch.stack(grids[::-1], dim=0)  # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            grid = torch.unsqueeze(grid, 0)
            grid = grid.to(dtype=flow.dtype, device=flow.device)
            norm_coeff = 2. / (torch.tensor(img_shape[::-1], dtype=flow.dtype, device=flow.device) - 1.)  # the coefficients to map image coordinates to [-1, 1]
            self.grid_dict[img_shape] = grid
            self.norm_coeff_dict[img_shape] = norm_coeff

        new_grid = grid + flow

        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1)  # n x h x w x 2
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1)  # n x d x h x w x 3
            # new_grid = new_grid[..., [2, 1, 0]]

        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        warped_input_img = F.grid_sample(input_image, new_grid * norm_coeff - 1., mode='bilinear', align_corners=True, padding_mode='border')
        return warped_input_img


def vec_inter(vec,transform,nb_steps=7):

    vec = vec/2**nb_steps
    for _ in range(nb_steps):
        vec1 = transform(vec,vec)
        vec = vec + vec1
    flow = vec
    return flow

class SPnet(nn.Module):
    """
    PCRnet
    """
    def __init__(self,vol_size=[160,192,160]):
       
        super(SPnet,self).__init__()
        self.conv0a = conv(1,8,kernel_size=3,stride=1)
        self.conv0b = conv(8,8,kernel_size=3,stride=1)

        self.conv1a = conv(8,16,kernel_size=3,stride=2)
        self.conv1b = conv(16,16,kernel_size=3,stride=1)
        self.conv_img1 = conv(1,8,kernel_size=3,stride=1)

        self.conv2a = conv(16+8,32,kernel_size=3,stride=2)
        self.conv2b = conv(32,32,kernel_size=3,stride=1)
        self.conv_img2 = conv(1,8,kernel_size=3,stride=1)

        self.conv3a = conv(32+8,32,kernel_size=3,stride=2)
        self.conv3b = conv(32,32,kernel_size=3,stride=1)
        self.conv_img3 = conv(1,8,kernel_size=3,stride=1)

        od = 32+32+16
        self.conv3_0 = conv(od,32,kernel_size=3,stride=1)
        self.conv3_1 = conv(32,32,kernel_size=3,stride=1)
        self.predict_flow3 = predict_flow(32)
        self.upfeat3 = deconv(32,8,kernel_size=4,stride=2,padding=1)
        self.wrap3 = SpatialTransformer()

        od = 32*2+8+3+16
        self.conv2_0 = conv(od,32,kernel_size=3,stride=1)
        self.conv2_1 = conv(32,32,kernel_size=3,stride=1)
        self.predict_flow2 = predict_flow(32)
        self.upfeat2 = deconv(32,8,kernel_size=4,stride=2,padding=1)
        self.wrap2 = SpatialTransformer()

        od = 16*2+8+3+16
        self.conv1_0 = conv(od,16,kernel_size=3,stride=1)
        self.conv1_1 = conv(16,16,kernel_size=3,stride=1)
        self.predict_flow1 = predict_flow(16)
        self.upfeat1 = deconv(16,8,kernel_size=4,stride=2,padding=1)
        self.wrap1 = SpatialTransformer()

        od = 8*2+8+3
        self.conv0_0 = conv(od,16,kernel_size=3,stride=1)
        self.conv0_1 = conv(16,16,kernel_size=3,stride=1)
        self.predict_flow0 = predict_flow(16)
        self.wrap0 = SpatialTransformer()

        

    def forward(self,fixed,moving):
        # Downsampling with three different multiples
        moving1,fixed1 = F.interpolate(moving,scale_factor=0.5),F.interpolate(fixed,scale_factor=0.5)
        moving2,fixed2 = F.interpolate(moving,scale_factor=0.25),F.interpolate(fixed,scale_factor=0.25)
        moving3,fixed3 = F.interpolate(moving,scale_factor=0.125),F.interpolate(fixed,scale_factor=0.125)
        #the first Conv ==> F1
        c10 = self.conv0b(self.conv0a(fixed))
        c20 = self.conv0b(self.conv0a(moving))

        #the second Conv ==> F2
        c11 = self.conv1b(self.conv1a(c10))
        c11_img = self.conv_img1(fixed1)
        c11 = torch.cat([c11,c11_img],dim=1)
        c21 = self.conv1b(self.conv1a(c20))
        c21_img = self.conv_img1(moving1)
        c21 = torch.cat([c21,c21_img],dim=1)

        #third Conv ==> F3
        c12 = self.conv2b(self.conv2a(c11))
        c12_img = self.conv_img2(fixed2)
        c12 = torch.cat([c12,c12_img],dim=1)
        c22 = self.conv2b(self.conv2a(c21))
        c22_img = self.conv_img2(moving2)
        c22 = torch.cat([c22,c22_img],dim=1)

        #four Conv ==> F4
        c13 = self.conv3b(self.conv3a(c12))
        c13_img = self.conv_img3(fixed3)
        if c13.shape[2:] != c13_img.shape[2:]:
            c13 = F.interpolate(c13, c13_img.shape[2:],mode='trilinear')
        c13 = torch.cat([c13,c13_img],dim=1)
        c23 = self.conv3b(self.conv3a(c22))
        c23_img = self.conv_img3(moving3)
        if c23.shape[2:] != c23_img.shape[2:]:
            c23 = F.interpolate(c23, c23_img.shape[2:],mode='trilinear')
        c23 = torch.cat([c23,c23_img],dim=1)


        #Symmetric feature block; cat(moxing,fixed)
        x = torch.cat((c23,c13),1)
        x = self.conv3_0(x)
        x = x + self.conv3_1(x)
        fd_vec3 =  self.predict_flow3(x)
        fd_up_feat3 = self.upfeat3(x)
        x = torch.cat((c13,c23),1)
        x = self.conv3_0(x)
        x = x + self.conv3_1(x)
        bd_vec3 =  self.predict_flow3(x)
        bd_up_feat3 = self.upfeat3(x)
        vec3 = (fd_vec3-bd_vec3)/2
        flow3 = vec_inter(vec3,self.wrap3)
        sym_flow3 = vec_inter(-vec3,self.wrap3)
        up_flow3 = 2*F.interpolate(flow3,scale_factor=2,mode='trilinear')
        up_sym_flow3 = 2*F.interpolate(sym_flow3,scale_factor=2,mode='trilinear')

        """
        symmetric feature regisration block
        warped moving
        warped fixed
        cat(warped_moving, fixed, up_x, flow)
        """
        if c22.shape[2:] != up_flow3.shape[2:]:
            up_flow3 = F.interpolate(up_flow3, c22.shape[2:], mode='trilinear')
        if c12.shape[2:] != up_sym_flow3.shape[2:]:
            up_sym_flow3 = F.interpolate(up_sym_flow3, c12.shape[2:], mode='trilinear')
        if c12.shape[2:] != fd_up_feat3.shape[2:]:
            fd_up_feat3 = F.interpolate(fd_up_feat3, c12.shape[2:], mode='trilinear')
        if c22.shape[2:] != bd_up_feat3.shape[2:]:
            bd_up_feat3 = F.interpolate(bd_up_feat3, c22.shape[2:],mode='trilinear')
        wrap2 = self.wrap2(c22,up_flow3)
        sym_wrap2 = self.wrap2(c12,up_sym_flow3)
        x = torch.cat((wrap2,c12,fd_up_feat3,up_flow3),1)
        x = self.conv2_0(x)
        x = x + self.conv2_1(x)
        fd_vec2 =  self.predict_flow2(x)
        fd_up_feat2 = self.upfeat2(x)
        x = torch.cat((sym_wrap2,c22,bd_up_feat3,up_sym_flow3),1)
        x = self.conv2_0(x)
        x = x + self.conv2_1(x)
        bd_vec2 =  self.predict_flow2(x)
        bd_up_feat2 = self.upfeat2(x)
        vec2 = (fd_vec2-bd_vec2)/2
        flow2 = vec_inter(vec2,self.wrap2)
        sym_flow2 = vec_inter(-vec2,self.wrap2)
        up_flow2 = 2*F.interpolate(flow2,scale_factor=2,mode='trilinear')
        sym_up_flow2 = 2*F.interpolate(sym_flow2,scale_factor=2,mode='trilinear')


        wrap1 = self.wrap1(c21,up_flow2)
        sym_wrap1 = self.wrap1(c11,sym_up_flow2)
        x = torch.cat((wrap1,c11,fd_up_feat2,up_flow2),1)
        x = self.conv1_0(x)
        x = x + self.conv1_1(x)
        fd_vec1 =  self.predict_flow1(x)
        fd_up_feat1 = self.upfeat1(x)
        x = torch.cat((sym_wrap1,c21,bd_up_feat2,sym_up_flow2),1)
        x = self.conv1_0(x)
        x = x + self.conv1_1(x)
        bd_vec1 =  self.predict_flow1(x)
        bd_up_feat1 = self.upfeat1(x)
        vec1 = (fd_vec1-bd_vec1)/2
        flow1 = vec_inter(vec1,self.wrap1)
        sym_flow1 = vec_inter(-vec1,self.wrap1)
        up_flow1 = 2*F.interpolate(flow1,scale_factor=2,mode='trilinear')
        sym_up_flow1 = 2*F.interpolate(sym_flow1,scale_factor=2,mode='trilinear')


        wrap0 = self.wrap0(c20,up_flow1)
        sym_wrap0 = self.wrap0(c10,sym_up_flow1)
        x = torch.cat((wrap0,c10,fd_up_feat1,up_flow1),1)
        x = self.conv0_0(x)
        x = x + self.conv0_1(x)
        fd_vec0 =  self.predict_flow0(x)
        x = torch.cat((sym_wrap0,c20,bd_up_feat1,sym_up_flow1),1)
        x = self.conv0_0(x)
        x = x + self.conv0_1(x)
        bd_vec0 =  self.predict_flow0(x)
        vec0 = (fd_vec0-bd_vec0)/2
        flow0 = vec_inter(vec0,self.wrap0)
        sym_flow0 = vec_inter(-vec0,self.wrap0)


        wraped = self.wrap0(moving,flow0)
        sym_wraped = self.wrap0(fixed,sym_flow0)
        wraped1 = self.wrap1(moving1,flow1)
        sym_wraped1 = self.wrap1(fixed1,sym_flow1)
        wraped2 = self.wrap2(moving2,flow2)
        sym_wraped2 = self.wrap2(fixed2,sym_flow2)
        wraped3 = self.wrap3(moving3,flow3)
        sym_wraped3 = self.wrap3(fixed3,sym_flow3)

        return ([wraped,wraped1,wraped2,wraped3],[fixed,fixed1,fixed2,fixed3],
                [sym_wraped,sym_wraped1,sym_wraped2,sym_wraped3],[moving,moving1,moving2,moving3],
                [flow0,flow1,flow2,flow3],[sym_flow0,sym_flow1,sym_flow2,sym_flow3],
                [vec0,vec1,vec2,vec3])

