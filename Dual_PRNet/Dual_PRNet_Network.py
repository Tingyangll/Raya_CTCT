import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import layers
from modelio import store_config_args, LoadableModel
from utilses.config import get_args
args = get_args()


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.main(x)
        out = self.activation(out)
        return out

class CorrTorch(nn.Module):
    def __init__(self, pad_size=1, max_displacement=1, stride1=1, stride2=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad3d(pad_size, 0)
        self.activate = nn.LeakyReLU(0.2)
        self.conv = nn.Conv3d(in_channels=27, out_channels=3, kernel_size=(1, 1, 1), stride=1)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsetz, offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1)])

        depth, hei, wid = in1.shape[2], in1.shape[3], in1.shape[4]

        sum = []
        for dz, dx, dy in zip(offsetz.reshape(-1), offsetx.reshape(-1), offsety.reshape(-1)):
            sum.append(torch.mean(in1 * in2_pad[:, :, dz:dz + depth, dy:dy + hei, dx:dx + wid], 1, keepdim=True))

        output = torch.cat(sum, 1)
        # output = self.conv(output)

        return self.activate(output)

class PRmoduleBlock(nn.Module):
    def __init__(self, dim,inchannels,outchannels):
        super().__init__()
        self.dim = dim
        self.corralation = CorrTorch(pad_size=1, max_displacement=1, stride1=1, stride2=1)
        self.conv1 = ConvBlock(dim, in_channels=inchannels*2+27,out_channels=inchannels ,stride=1) #reduce the channels
        self.conv2 = ConvBlock(dim, in_channels=inchannels , out_channels=inchannels ,stride=1)  # preserve more context information
        self.conv3 = ConvBlock(dim, in_channels=inchannels , out_channels=3, stride=1)  # estimate the deformation
        self.transformer = layers.SpatialTransformer(self.dim)

    def forward(self,source,target,dvf):
        dvf = F.interpolate(dvf, source.shape[2:], mode='trilinear', align_corners=True)
        warped = self.transformer(source,dvf)
        feature = self.corralation(target,warped)
        feature = torch.cat((source,feature),dim=1)
        feature = torch.cat((feature,target),dim=1)

        feature = self.conv1(feature)
        feature1 = self.conv2(feature)
        featuer1 = feature + feature1
        feature1 = self.conv3(featuer1)
        return feature1

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 dim=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = dim
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf
        self.PRmodule1 = PRmoduleBlock(ndims,inchannels=32,outchannels=32)
        self.PRmodule2 = PRmoduleBlock(ndims, inchannels=32, outchannels=32)
        self.PRmodule3 = PRmoduleBlock(ndims, inchannels=16, outchannels=16)
        self.PRmodule4 = PRmoduleBlock(ndims, inchannels=16, outchannels=16)
        self.PRmodule5 = PRmoduleBlock(ndims, inchannels=8, outchannels=8)

    def forward(self, source, target):

        # encoder forward pass
        source_history = [source]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                source = conv(source)
            source_history.append(source)
            source = F.interpolate(source, scale_factor=0.5, mode='trilinear',align_corners=True, recompute_scale_factor=False)

        target_history = [target]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                target = conv(target)
            target_history.append(target)
            target = F.interpolate(target, scale_factor=0.5, mode='trilinear', align_corners=True, recompute_scale_factor=False)

        # decoder forward pass with upsampling and concatenation
        source_decoder = [source]
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                source = conv(source)
                source_decoder.append(source)
            if not self.half_res or level < (self.nb_levels - 2):
                source = F.interpolate(source, source_history[-1].shape[2:], mode='trilinear',align_corners=True)
                source = torch.cat([source, source_history.pop()], dim=1)

        target_decoder = [target]
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                target = conv(target)
                target_decoder.append(target)
            if not self.half_res or level < (self.nb_levels - 2):
                target = F.interpolate(target, target_history[-1].shape[2:], mode='trilinear', align_corners=True)
                target = torch.cat([target, target_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
             source = conv(source)

        for conv in self.remaining:
             target = conv(target)

        _,_,D,H,C = source_decoder[2].shape
        DVF0 = torch.zeros((1,3,D,H,C),dtype=float).to(args.device).float()
        DVF1 = self.PRmodule1(source_decoder[2],target_decoder[2],DVF0)
        DVF2 = self.PRmodule2(source_decoder[3], target_decoder[3], DVF1)
        DVF3 = self.PRmodule3(source_decoder[4], target_decoder[4], DVF2)
        DVF4 = self.PRmodule4(source_decoder[5], target_decoder[5], DVF3)
        DVF5 = self.PRmodule5(source,target, DVF4)

        return DVF5


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 dim,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.dim = dim
        # ensure correct dimensionality
        assert dim == 3, 'ndims should be 3. found: %d' % dim

        # configure core unet model
        self.unet_model = Unet(
            self.dim,
            # infeats=(src_feats + trg_feats),
            infeats=(src_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % self.dim)
        # self.flow = Conv(self.unet_model.final_nf, self.dim, kernel_size=3, padding=1)
        self.flow = Conv(self.dim, self.dim, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, self.dim)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, self.dim)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        # down_shape = [int(dim / int_downsize) for dim in inshape]
        # self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.integrate = layers.VecInt(None, int_steps) if int_steps > 0 else None
        # configure transformer
        # self.transformer = layers.SpatialTransformer(inshape)
        self.transformer = layers.SpatialTransformer(self.dim)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet


        x = self.unet_model(source,target)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        if source.shape != pos_flow.shape:
            pos_flow = F.interpolate(pos_flow, source.shape[2:], mode='trilinear',
                                     align_corners=True)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow, pos_flow) if self.bidir else (y_source, preint_flow, pos_flow)
        else:
            return y_source, pos_flow



