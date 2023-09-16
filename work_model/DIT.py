import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from timm.models.layers import DropPath, trunc_normal_
from thop import profile
from torch.autograd import Variable
import numpy as np
from utilis.LOSS import FocalLoss

import copy

import unfoldNd
from utilis.pnp import SEModule_2D,SEModule_3D
# helpers


def exists(val):
    return val is not None


def conv_output_size(image_size, kernel_size, stride, padding=0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num


# classes


class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


class RearrangeImage1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return rearrange(x, 'b (d h w) c -> b c d h w', h=round(pow(int(x.shape[1])//4, 1 / 3))*2,
                         w=round(pow(int(x.shape[1])//4, 1 / 3))*2)

class unfd(nn.Module):
    def __init__(self,kernel_size,stride,padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #self.unfold = unfoldNd.unfoldNd(x,kernel_size,stride,padding)

    def forward(self,x):
        return unfoldNd.unfoldNd(x,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

def pair(t):
    # 把t变成一对输出
    return t if isinstance(t, tuple) else (t, t)

class Inception(nn.Module):
    def __init__(self, unfold1, pool, unfold2):
        super().__init__()
        self.unfold1_kernel = unfold1[0]
        self.unfold1_stride = unfold1[1]
        self.unfold1_padding = unfold1[2]

        self.pool = nn.AvgPool3d(kernel_size=pool[0],stride=pool[1],padding=pool[2])

        self.unfold2_kernel = unfold2[0]
        self.unfold2_stride = unfold2[1]
        self.unfold2_padding = unfold2[2]

    def forward(self,x):
        x1 = unfoldNd.unfoldNd(x,kernel_size=self.unfold1_kernel,stride=self.unfold1_stride,padding=self.unfold1_padding)
        x2 = self.pool(x)
        x2 = unfoldNd.unfoldNd(x2,kernel_size=self.unfold2_kernel,stride=self.unfold2_stride,padding=self.unfold2_padding)
        x = torch.cat((x1,x2),dim=1)
        #print(x.shape)
        return x

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, dropout=0.):
        super().__init__()
        # 这个前传过程其实就是几层全连接
        """ print(in_dim)
        print(hidden_dim)
        print(out_dim) """
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #print(x.shape)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, out_dim=None, dropout=0., se=0):
        super().__init__()
        # dim_head是每个头的特征维度
        # 多个头的特征是放在一起计算的
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.out_dim = out_dim
        self.dim = dim

        self.attend = nn.Softmax(dim=-1)
        # 这个就是产生QKV三组向量因此要乘以3
        #print(dim,inner_dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if out_dim is None:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, out_dim),
            ) if project_out else nn.Identity()

        self.to_out_dropout = nn.Dropout(dropout)

        self.se = se
        if self.se > 0:
            self.se_layer = SE(dim)

    def forward(self, x):
        # b是batch size h 是注意力头的数目 n 是图像块的数目
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        if self.se:
            out = self.se_layer(out)

        out = self.to_out_dropout(out)

        if self.out_dim is not None and self.out_dim != self.dim:
            # 这个时候需要特殊处理，提前做一个残差
            """ print(1)
            print(v.squeeze(1).shape)
            print(out.shape) """
            out = out + v.squeeze(1)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_out_dim=None, ff_out_dim=None, dropout=0., se=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn_out_dim = attn_out_dim
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, out_dim=attn_out_dim, dropout=dropout, se=se)),
                PreNorm(dim if not attn_out_dim else attn_out_dim,
                        FeedForward(dim if not attn_out_dim else attn_out_dim, mlp_dim, out_dim=ff_out_dim,
                                    dropout=dropout))
            ]))

    def forward(self, x):
        #print("syc--x.shape(input) = {}\t",x.shape) #syc自己加的
        for attn, ff in self.layers:
            # 都是残差学习
            if self.attn_out_dim is not None and self.dim != self.attn_out_dim:
                x = attn(x)
            else:
                x = attn(x) + x
            #print(x.shape)
            x = ff(x) + x
        # print("syc--x.shape(output) = {}",x.shape)
        return x


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        #print(x.shape)
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class T2TViT(nn.Module):
    def __init__(self, *,
                 image_size,
                 num_classes,
                 dim,
                 depth=None,
                 heads=None,
                 mlp_dim=None,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 transformer=None,
                 t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = [channels * 7 * 7, 64 * 3 * 3]
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            # layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(
                output_image_size, kernel_size, stride, stride // 2)
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size,
                          stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64/2, mlp_dim=64/2, attn_out_dim=64/2, ff_out_dim=64/2,
                            dropout=dropout) if not is_last else nn.Identity(),

                #Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64,mlp_dim=64, attn_out_dim=64, ff_out_dim=64,dropout=dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim[1], dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]
                       ), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # return self.mlp_head(x)
        return x


class DiT(nn.Module):
    def __init__(self, *,
                 image_size,
                 num_classes,
                 dim,
                 depth=None,
                 heads=None,
                 mlp_dim=None,
                 pool='cls',
                 channels=1,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 patch_emb='share',
                 time_emb=False,
                 pos_emb='share',  # share or isolated # 共享或独立或不加
                 use_scale=False,
                 transformer=None,
                 t2t_layers=((7, 4,2), (3, 2,1), (3, 1,1))):
        super().__init__()
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = [channels * 7 * 7 * 7, 64 * 3 * 3 * 3]
        output_image_size = image_size

        #layers.append(SEModule_3D(in_channels=channels,reduction_ratio=4))#在之前加入一个SE看看效果
        for i, (kernel_size, stride,padding) in enumerate(t2t_layers):
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(
                output_image_size, kernel_size, stride, padding)
            layers.extend([
                #RearrangeImage() if not is_first else nn.Identity(),
                RearrangeImage1() if not is_first else nn.Identity(),
                #nn.Unfold(kernel_size=kernel_size,stride=stride, padding=stride // 2),
                unfd(kernel_size, stride, padding),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64,
                            mlp_dim=64, attn_out_dim=64, ff_out_dim=64,
                            dropout=dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim[1], dim))
        layers.append(SE(dim,16))
        #layers.append(nn.AvgPool1d(kernel_size=2))

##################### inception尝试
        """layer_dim = [304 , 128]
        for i, (unfold1, pool1, unfold2) in enumerate(t2t_layers):
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            layers.extend([
                RearrangeImage1() if not is_first else nn.Identity(),
                Inception(unfold1,pool1,unfold2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64,
                            mlp_dim=64, attn_out_dim=64, ff_out_dim=64,
                            dropout=dropout) if not is_last else nn.Identity(),
            ])
        output_image_size = 4
        layers.append(SE(dim,16)) """
#########################

        self.patch_emb = patch_emb
        if self.patch_emb == 'share':
            self.to_patch_embedding = nn.Sequential(*layers)  # 共用
        elif self.patch_emb == 'isolated':
            layers_before = copy.deepcopy(layers)
            layers_after = copy.deepcopy(layers)
            self.to_patch_embedding_before = nn.Sequential(*layers_before)  # 不共用
            self.to_patch_embedding_after = nn.Sequential(*layers_after)

        """
        尝试学习一个矩阵
        """
        self.mat = nn.Parameter(torch.randn(layer_dim[1],dim))
        self.se = SE(dim,16)

        self.pos_emb = pos_emb
        if self.pos_emb == 'share':
            #print(output_image_size)
            self.pos_embedding_before_and_after = nn.Parameter(
                torch.randn(1, output_image_size ** 3 //2 , dim))
        elif self.pos_emb == 'sin':
            self.pos_embedding_before_and_after = self.get_sinusoid_encoding(output_image_size ** 3 , dim)
        elif self.pos_emb == 'isolated':
            self.pos_embedding_before = nn.Parameter(
                torch.randn(1, output_image_size ** 3 //2 , dim))
            self.pos_embedding_after = nn.Parameter(
                torch.randn(1, output_image_size ** 3 //2 , dim))

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Parameter(torch.randn(2, dim))

        self.use_scale = use_scale
        if self.use_scale:
            # self.scale = nn.Parameter(torch.randn(1, 2)) # 当前最佳
            self.scale = nn.Sequential(
                nn.Linear(2 * output_image_size ** 3 //2 , 2)
            )
            self.softmax = nn.Softmax(dim=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]
                       ), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def get_sinusoid_encoding(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).cuda()

    def forward(self, before_x, after_x):
        #print(before_x.shape)
        if self.patch_emb == 'share':
            before_x = self.to_patch_embedding(before_x)
            after_x = self.to_patch_embedding(after_x)
        elif self.patch_emb == 'isolated':
            before_x = self.to_patch_embedding_before(before_x)
            # print("layers_before := \t", before_x.shape) #torch.Size([3, 1024, 512])
            after_x = self.to_patch_embedding_after(after_x)
        #print(before_x.shape)
        """  
        尝试学习一个矩阵
        """
        """  before_x = torch.matmul(before_x,self.mat)
        after_x = torch.matmul(after_x,self.mat)
        before_x = self.se(before_x)
        after_x = self.se(after_x)
        """
        b, n, _ = before_x.shape

        # 把cls token弄进去
        if self.pos_emb == 'share' or self.pos_emb == 'sin':
            #print(before_x.shape)
            #print(self.pos_embedding_before_and_after.shape)
            before_x += self.pos_embedding_before_and_after
            after_x += self.pos_embedding_before_and_after
        elif self.pos_emb == 'isolated':
            before_x += self.pos_embedding_before
            after_x += self.pos_embedding_after

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if not self.use_scale:
            x = torch.cat((cls_tokens, before_x, after_x), dim=1)
        else:
            x = torch.cat((before_x, after_x), dim=1)

        if self.time_emb:
            if not self.use_scale:
                x[:, 1:(n + 1)] += self.time_embedding[0]
                x[:, (n + 1):] += self.time_embedding[1]
            else:
                x[:, :n] += self.time_embedding[0]
                x[:, n:] += self.time_embedding[1]

        x = self.dropout(x)
        x = self.transformer(x)
        if not self.use_scale:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        else:
            # scale = self.softmax(self.scale)  # 之前最佳
            # x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1) # 之前最佳
            #print(x.shape) #(1,128,128)
            #print(x.mean(dim=-1).shape)
            scale = self.scale(x.mean(dim=-1))
            scale = self.softmax(scale)
            #print("前后特征权重",scale)
            # scale = scale.view(scale.shape[0], scale.shape[1], 1)
            x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1)

        x = self.to_latent(x)
        # return self.mlp_head(x)
        return x


class ViT_v2(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim=None,
                 depth=None,
                 heads=None,
                 mlp_dim=None,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 time_emb=False,
                 pos_emb='share',
                 use_scale=False,
                 transformer=None):
        super().__init__()
        # 图像的长宽和每个Patch的长宽
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 会有多少个patch
        num_patches = (image_height // patch_height) * \
                      (image_width // patch_width)
        # 图像的维数
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 编码每一个Patch的信息
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_emb = pos_emb

        if self.pos_emb == 'share':
            self.pos_embedding_before_and_after = nn.Parameter(
                torch.randn(1, num_patches, dim))
        elif self.pos_emb == 'isolated':
            self.pos_embedding_before = nn.Parameter(
                torch.randn(1, num_patches, dim))
            self.pos_embedding_after = nn.Parameter(
                torch.randn(1, num_patches, dim))

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Parameter(torch.randn(2, dim))

        self.use_scale = use_scale
        if self.use_scale:
            # self.scale = nn.Parameter(torch.randn(1, 2)) # 当前最佳
            self.scale = nn.Sequential(
                nn.Linear(2 * num_patches, 2)
            )
            self.softmax = nn.Softmax(dim=1)

        # 类别token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]
                       ), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        # self.transformer = Transformer(
        #     dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # 最后的层我们自己融合
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img_before, img_after):
        before_x = self.to_patch_embedding(img_before)
        after_x = self.to_patch_embedding(img_after)

        b, n, _ = before_x.shape

        # 把cls token弄进去
        if self.pos_emb == 'share':
            before_x += self.pos_embedding_before_and_after
            after_x += self.pos_embedding_before_and_after
        elif self.pos_emb == 'isolated':
            before_x += self.pos_embedding_before
            after_x += self.pos_embedding_after

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if not self.use_scale:
            x = torch.cat((cls_tokens, before_x, after_x), dim=1)
        else:
            x = torch.cat((before_x, after_x), dim=1)

        if self.time_emb:
            if not self.use_scale:
                x[:, 1:(n + 1)] += self.time_embedding[0]
                x[:, (n + 1):] += self.time_embedding[1]
            else:
                x[:, :n] += self.time_embedding[0]
                x[:, n:] += self.time_embedding[1]

        # dropout操作
        x = self.dropout(x)
        # 开始transformer
        x = self.transformer(x)
        print(self.use_scale)
        # 如果是mean模式，则对图像块所有的输出作为平均从而进行下一步分类
        # 如果是cls，则用token的输出作为特征来进行下一步的分类
        if not self.use_scale:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        else:
            # scale = self.softmax(self.scale)  # 之前最佳
            # x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1) # 之前最佳
            scale = self.scale(x.mean(dim=-1))
            scale = self.softmax(scale)
            # scale = scale.view(scale.shape[0], scale.shape[1], 1)
            x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1)

        x = self.to_latent(x)

        # return self.mlp_head(x)
        return x


class DiT_basic(nn.Module):
    def __init__(self,
                 basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                 patch_emb='isolated',
                 time_emb=True,
                 pos_emb='share',
                 use_scale=False,
                 pool='cls',
                 loss_f='focal',  # focal of ce
                 input_type='both',
                 label_type='MP',
                 output_type='probability'):
        super(DiT_basic, self).__init__()
        print('basic_model: %s\n'
              'input_type: %s\n'
              'label_type: %s\n'
              'loss_f: %s\n'
              'patch_emb: %s\n'
              'time_emb: %d\n'
              'use_scale: %d\n'
              'pos_emb: %s\n'
              'pool: %s\n'
              % (
                  basic_model, input_type, label_type, loss_f, patch_emb, time_emb, use_scale, pos_emb, pool))

        self.input_type = input_type
        self.label_type = label_type
        self.output_type = output_type

        self.feature_net = nn.Sequential()

        # baseline模型是普通resnet18不做任何修改

        if basic_model == 't2t':
            if input_type == 'both':
                self.net = DiT(image_size=64,
                               num_classes=2,
                               dim=128,
                               depth=None,
                               heads=None,
                               mlp_dim=None,
                               pool=pool,
                               channels=2,
                               dim_head=64,
                               dropout=0.,
                               emb_dropout=0.,
                               patch_emb=patch_emb,
                               pos_emb=pos_emb,
                               time_emb=time_emb,
                               use_scale=use_scale,
                               transformer=Transformer(dim=128,
                                                       depth=12,
                                                       heads=16,
                                                       dim_head=64,
                                                       mlp_dim=256),
                               t2t_layers=((7, 4,2), (3, 2,1), (3, 2,1)),
                               #t2t_layers=((7, 4,2), (3, 2,1), (2, 1,0)),
                               #t2t_layers = (([5,4,2],[3,2,1],[3,2,1]), 
                                #             ([1,2,0],[3,2,1],[1,1,0]), 
                                 #            ([1,2,0],[3,2,1],[1,1,0]))
                               )
            else:
                self.net = T2TViT(image_size=224,
                                  num_classes=1,
                                  dim=256,
                                  pool=pool,
                                  channels=3,
                                  dim_head=64,
                                  dropout=0.,
                                  emb_dropout=0.,
                                  transformer=Transformer(dim=256,
                                                          depth=16,
                                                          heads=16,
                                                          dim_head=64,
                                                          mlp_dim=512),
                                  t2t_layers=((7, 4), (3, 2), (3, 2)))
        elif basic_model == 'vit':
            self.net = ViT_v2(image_size=224,
                              patch_size=16,
                              num_classes=2,
                              dim=256,
                              pool=pool,
                              channels=3,
                              dim_head=64,
                              dropout=0.,
                              emb_dropout=0.,
                              time_emb=time_emb,
                              pos_emb=pos_emb,
                              use_scale=use_scale,
                              transformer=Transformer(dim=256,
                                                      depth=16,
                                                      heads=16,
                                                      dim_head=64,
                                                      mlp_dim=512))

        if self.input_type == 'both':
            self.fc = nn.Sequential(nn.LayerNorm(128),
                                    nn.Linear(128, 2))
        else:
            self.fc = nn.Sequential(nn.LayerNorm(4096),
                                    nn.Linear(4096, 2))

        # loss函数和softmax
        if label_type == 'MP':
            weight = torch.tensor([1.0, 1.0])
        elif label_type == 'LNM':
            weight = torch.tensor([1.0, 1.0])

        if loss_f == 'ce':
            self.loss = nn.CrossEntropyLoss(weight=weight)
            # self.loss = nn.functional.binary_cross_entropy(weight=weight)
        elif loss_f == 'focal':
            self.loss = FocalLoss(class_num=2, alpha=weight, gamma=2)
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        print("initialize weights for network!")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='gelu')
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight, gain=1) #尝试其他初始化方式
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                #trunc_normal_(m.weight, std=0.02)
                #torch.nn.init.xavier_normal_(m.weight.data,gain=1) #尝试其他初始化方式
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, before_x, after_x, labels_MP=None, labels_LNM=None):
        #print("brfore_x: = {}",before_x.shape)  #[1, 3, 224, 224]
        before_x = self.feature_net(before_x)
        #print("brfore_x: = {}",before_x.shape)  #[1, 3, 224, 224]
        after_x = self.feature_net(after_x)

        if self.input_type == 'before':
            before_out = self.net(before_x)
            out = before_out
        elif self.input_type == 'after':
            after_out = self.net(after_x)
            out = after_out
        elif self.input_type == 'both':
            # 两个都用before_net精度还不错
            out = self.net(before_x, after_x)

        out = out.view(out.shape[0], -1)
        #print(out.shape)
        out = self.fc(out)

        if labels_MP is not None or labels_LNM is not None:  # training or validation process
            # output loss and acc
            labels_MP = labels_MP.view(-1)
            labels_LNM = labels_LNM.view(-1)
            #print(out.shape) #torch.Size([3, 2])
            prob = self.softmax(out)
            if self.label_type == 'MP':
                # cls_loss = self.loss(out, labels_MP.float())
                # print(labels_MP)
                cls_loss = self.loss(out, labels_MP)
            elif self.label_type == 'LNM':
                cls_loss = self.loss(out, labels_LNM)

            _, predicted = torch.max(prob.data, 1)
            # print(predicted)
            # predicted = prob.data > 0.5
            if self.label_type == 'MP':
                acc = (predicted == labels_MP).sum().item() / out.size(0)
            elif self.label_type == 'LNM':
                acc = (predicted == labels_LNM).sum().item() / out.size(0)
            # print(cls_loss,acc)
            return cls_loss, acc
        else:  # test process
            # output probability of each class
            if self.output_type == 'probability':
                out = self.softmax(out)
                return out[:, 1]
            elif self.output_type == 'score':
                out = out
                return out

            return None
#############
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
####################

def main():
    from dataset import train_dataset
    from config import args
    from torch.utils.data import DataLoader
    train_ds = train_dataset.Train_Dataset(args)
    train_dl = DataLoader(train_ds, 3, False, num_workers=1)
    print(len(train_dl))
    for i, (pre, post, label) in enumerate(train_dl):
        print(pre.shape,post.shape,label)
        output = net(pre,post,label,label)
            # print(output)



if __name__ == '__main__':
    net = DiT_basic(basic_model='t2t',  # 使用vit还是t2t vit只支持both 因为其他的不需要消融探究
                    patch_emb='isolated',
                    time_emb=True,
                    pos_emb='share',
                    use_scale=True,
                    loss_f='ce',#ce  # focal of ce
                    input_type='both',
                    pool='cls',
                    output_type='probability')
    # net = ViT_basic(basic_model='t2t')
    input = torch.randn(1, 2, 32,64, 64)
    # output = net(input,input)
    # print(output)

    #print_network(net)
    net(input,input)
    flops, params = profile(net, (input, input,))
    print('flops: ', flops, 'params: ', params)
    # main()

#flops:  8928161402.0       params:  21632480.0           （原版）
#flops:  90331027978.0 (10倍)params:  350980576.0（16倍）  844
#flops:  92775832586.0      params:  43935328.0           16 8 8
#flops:  91756614666.0      params:  42927072.0           16 8 8 - v2（就是这个文件）
#flops:  90494653450.0      params:  42289824.0   （1 1 1）        16 8 8 - v2（这个文件的上一个版本）
#flops:  91367068682.0      params:  43141792.0  (3 1 1的版本)
#flops:  60406071966.0 params:  43140440.0
    # total_trainable_params = sum(
    # p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
