import torch.nn as nn
import torch
from torch.nn import Conv3d, LayerNorm, Parameter, Dropout, Linear, Softmax, GroupNorm, Upsample, MaxPool3d,BatchNorm3d
import math
""""""
from torch.distributions.normal import Normal
from layers import SpatialTransformer as SpatialTransformerself
import utilize
class Embedding(nn.Module):
    def __init__(self, configs):
        super(Embedding, self).__init__()
        d, h, w = configs['input_size']
        patch_size = configs['patch_size']
        hidden_dim = configs['hidden_dim']
        n_patches = (d // patch_size[0]) * (h // patch_size[1]) * (w // patch_size[2])
        self.patch_embbeddings = Conv3d(in_channels=configs['emb_in_channels'],
                                        out_channels=hidden_dim,
                                        kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = Parameter(torch.zeros(1, hidden_dim, n_patches,))
        self.dropout = Dropout(configs['embedding_dprate'])

    def forward(self, x):
        embed = self.position_embeddings
        _,_,d,h,w = x.shape

        x = self.patch_embbeddings(x)
        if x.shape != embed.shape:
            embed = embed.view(1, 256, 10, 10, 10)
            embed = nn.functional.interpolate(embed, x.shape[2:], mode='trilinear', align_corners=True)
            embed = embed.flatten(2)
        # utilize.show_slice(x.cpu().detach().numpy())
        _, _, d, h, w = x.shape
        x = x.flatten(2)  # start from 3

        embeddings = x + embed
        # utilize.show_slice(embeddings.cpu().detach().numpy())
        embeddings = self.dropout(embeddings)
        return d,h,w,embeddings


class AttentionBlock(nn.Module):
    def __init__(self, hidden_ch, head_num):
        super(AttentionBlock, self).__init__()
        self.head_num = head_num
        self.head_size = hidden_ch // head_num
        self.all_head_size = self.head_num * self.head_size
        self.query = Linear(hidden_ch, self.all_head_size)
        self.key = Linear(hidden_ch, self.all_head_size)
        self.value = Linear(hidden_ch, self.all_head_size)
        self.out = Linear(hidden_ch, hidden_ch)
        self.softmax = Softmax(dim=-1)

    def reshape_multi_heads(self, x):
        new_size = x.size()[:-1] + (self.head_num, self.head_size)
        x = x.view(*new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query_layer = self.reshape_multi_heads(query)
        key_layer = self.reshape_multi_heads(key)
        value_layer = self.reshape_multi_heads(value)

        att_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        att_scores = att_scores / math.sqrt(self.head_size)
        att_probs = self.softmax(att_scores)
        context_layer = torch.matmul(att_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        att_out = self.out(context_layer)
        return att_out


class MLP(nn.Module):
    def __init__(self, hidden_ch, mlp_dim, dp_rate):
        super(MLP, self).__init__()
        self.fc1 = Linear(hidden_ch, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_ch)
        self.act_fn = nn.functional.gelu
        self.dropout = Dropout(dp_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, configs):
        super(TransformerBlock, self).__init__()
        self.att_norm = LayerNorm(configs['hidden_dim'], eps=1e-6)
        self.attention = AttentionBlock(configs['hidden_dim'], configs['head_num'])
        # self.mlp = MLP(configs['hidden_ch'], configs['mlp_din'], configs['mlp_dprate'])
        self.mlp = MLP(configs['hidden_dim'], configs['mlp_dim'], configs['mlp_dprate'])
        self.fn_norm = LayerNorm(configs['hidden_dim'], eps=1e-6)

    def forward(self, x):
        h = x
        B,C,DHW = x.shape
        x = x.view(B,DHW,C)
        x = self.att_norm(x)
        attention = self.attention(x)
        attention = attention.view(B,C,DHW)
        x = h + attention

        h = x
        x = x.view(B, DHW, C)
        x = self.fn_norm(x)
        x = self.mlp(x)
        x = x.view(B, C, DHW)
        x = x + h
        return x


class Transformer(nn.Module):
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.layer = nn.ModuleList()
        self.embeddings = Embedding(configs)
        self.encoder_norm = LayerNorm(configs['hidden_dim'], eps=1e-6)
        for _ in range(configs['trans_layers']):
            transblock = TransformerBlock(configs)
            self.layer.append(transblock)

    def forward(self, x):
        d,h,w,x = self.embeddings(x)
        for layer_block in self.layer:
            x = layer_block(x)
        B, C, DHW = x.shape
        x = x.view(B, DHW, C)
        encoded = self.encoder_norm(x)

        return d,h,w,encoded#(B, DHW, C)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(0.2),
            # GroupNorm(num_groups=4, num_channels=out_channels)
            nn.InstanceNorm3d(out_channels)
            # BatchNorm3d(out_channels)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, conv_num, in_channels, out_channels, kernel_size):#2,1,8,3
        super(ConvLayer, self).__init__()
        self.layer = nn.ModuleList()
        self.block0 = ConvBlock(in_channels, out_channels, 1)
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size)
        for _ in range(conv_num-1):
            self.layer.append(ConvBlock(in_channels, out_channels, kernel_size))

    def forward(self, x):
        h = self.block0(x)
        for layer_block in self.layer:
            x = layer_block(x)
        return h + x


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.conv_l1 = ConvLayer(2, 2, 8, 3)
        self.d1 = MaxPool3d(kernel_size=2, stride=2)
        self.conv_l2 = ConvLayer(2, 8, 16, 3)
        self.d2 = MaxPool3d(kernel_size=2, stride=2)
        self.conv_l3 = ConvLayer(2, 16, 32, 3)
        self.transformer = Transformer(configs)

    def forward(self, x):
        #b,c,d,h,w
        cl1 = self.conv_l1(x)
        # utilize.show_slice(cl1.cpu().detach().numpy())
        dl1 = self.d1(cl1)
        cl2 = self.conv_l2(dl1)
        # utilize.show_slice(cl2.cpu().detach().numpy())
        dl2 = self.d2(cl2)
        cl3 = self.conv_l3(dl2)
        # utilize.show_slice(cl3.cpu().detach().numpy())
        d,h,w,out = self.transformer(cl3)#(B, DHW, C)
        features = [cl3, cl2, cl1]
        return d,h,w,out, features


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch):
        super(DecoderBlock, self).__init__()
        self.up = Upsample(scale_factor=2)
        self.downsample = MaxPool3d(2, 2)
        self.conv1 = ConvBlock(in_ch+skip_ch, out_ch, 3)
        self.conv2 = ConvBlock(out_ch, out_ch, 3)

    def forward(self, x, feat):
        # x = self.up(x)

        if feat is not None:
            feat = self.downsample(feat)
            # utilize.show_slice(feat.cpu().detach().numpy())
            # if x.shape[2:]!=feat.shape[2:]:
            x = nn.functional.interpolate(x, feat.shape[2:], mode='trilinear', align_corners=True)
            # utilize.show_slice(x.cpu().detach().numpy())
            x = torch.cat([x, feat], dim=1)
            # utilize.show_slice(x.cpu().detach().numpy())
        x = self.conv1(x)
        # utilize.show_slice(x.cpu().detach().numpy())
        x = self.conv2(x)
        # utilize.show_slice(x.cpu().detach().numpy())
        return x


class CUP(nn.Module):
    def __init__(self, configs):
        super(CUP, self).__init__()
        self.n_skip = configs['n_skip']
        self.patch_size = configs['patch_size']
        d, h, w = configs['input_size']
        patch_size = configs['patch_size']
        self.ori_size = d//patch_size[0], h//patch_size[1], w//patch_size[2]
        decoder_blocks = [DecoderBlock(in_ch, out_ch, skip_ch) for in_ch, out_ch, skip_ch in configs['decoder_channels']]
        self.blocks = nn.ModuleList(decoder_blocks)
        self.conv = ConvBlock(configs['hidden_dim'], configs['head_channels'], 3)

    def forward(self, d,h,w,x, features):
        B, n_patch, hidden_dim = x.size()#(B, DHW, C)
        x = x.permute(0, 2, 1)
        new_shape = (B, hidden_dim,) + self.ori_size
        if n_patch == (self.ori_size[0])*(self.ori_size[1])*(self.ori_size[2]):
            x = x.contiguous().view(*new_shape)
        else:
            x = x.contiguous().view(B,hidden_dim,d,h,w)
            # x = nn.functional.interpolate(x, new_shape[2:], mode='trilinear', align_corners=True)

        # utilize.show_slice(x.cpu().detach().numpy())
        x = self.conv(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                feat = features[i] if i < self.n_skip else None
            else:
                feat = None
            x = decoder_block(x, feat)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SegmentationHead(nn.Module):
    def __init__(self, in_ch, n_class):
        super(SegmentationHead, self).__init__()
        self.up = Upsample(scale_factor=2)
        # self.head = Conv3d(in_channels=in_ch, out_channels=3, kernel_size=1, stride=1)
        self.head = RegistrationHead(in_channels=in_ch,out_channels=3,kernel_size=3)
        # self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.up(x)
        x = self.head(x)
        # x = self.softmax(x)
        return x


class TransUnet(nn.Module):
    def __init__(self, configs):
        super(TransUnet, self).__init__()
        self.encoder = Encoder(configs)
        self.decoder = CUP(configs)
        self.head = SegmentationHead(configs['last_in_ch'], configs['n_class'])
        self.STN = SpatialTransformerself()
    def forward(self, x,train=True):
        #b,c,d,h,w
        source = x[:, 0:1, :, :]
        d,h,w,x, features = self.encoder(x)
        x = self.decoder(d,h,w,x, features)
        # utilize.show_slice(x.cpu().detach().numpy())
        x = self.head(x)
        # utilize.show_slice(x.cpu().detach().numpy())

        if train:
            out = self.STN(source, x)
            return out, x
        else:
            return x


# if __name__ == '__main__':
#     img = torch.randn(1, 1, 128, 256, 256)
#     configs = {
#         'hidden_dim': 256,
#         'input_size': (32, 64, 64),
#         'patch_size': (4, 4, 4),
#         'trans_layers': 2,
#         'mlp_dim': 512,
#         'embedding_dprate': 0.1,
#         'head_num': 4,
#         'emb_in_channels': 32,
#         'mlp_dprate': 0.2,
#         'n_class': 2,
#         'last_in_ch': 16,
#         'n_skip': 3,
#         'head_channels': 128,
#         'decoder_channels': [(128, 64, 32), (64, 32, 16), (32, 16, 8)]
#     }
#     model = TransUnet(configs)
#     pred = model(img)
#     print(pred.shape)

configs = {
        'hidden_dim': 256,
        'input_size': (32, 64, 64),
        'patch_size': (4, 4, 4),
        'trans_layers': 2,
        'mlp_dim': 512,
        'embedding_dprate': 0.1,
        'head_num': 4,
        'emb_in_channels': 32,
        'mlp_dprate': 0.2,
        'n_class': 3,
        'last_in_ch': 16,
        'n_skip': 3,
        'head_channels': 128,
        'decoder_channels': [(128, 64, 32), (64, 32, 16), (32, 16, 8)]
    }