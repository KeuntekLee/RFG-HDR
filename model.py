import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class CALayer(nn.Module):
    def __init__(self, n_dim, reduction=4):
        super(CALayer, self).__init__()
        #nChannels_ = nChannels
        self.n_dim = n_dim
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(n_dim, n_dim//reduction,kernel_size=1)
        self.conv2 = nn.Conv2d(n_dim//reduction, n_dim,kernel_size=1)
    def forward(self, x):
        y = self.pool(x)
        
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, n_dim, reduction=4):
        super(RCAB, self).__init__()
        #nChannels_ = nChannels
        self.n_dim = n_dim
        #self.ln = nn.GroupNorm(1, n_dim)
        #self.ln = nn.LayerNorm(n_dim)
        self.ln = LayerNorm(n_dim, 'WithBias')
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(n_dim,n_dim,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(n_dim,n_dim,kernel_size=3,padding=1)
        self.ca = CALayer(n_dim,reduction)
    def forward(self, x):
        #print(x.shape)
        #x = x.permute(0,2,3,1)
        y = self.ln(x)
        #print(y.shape)
        #y = y.permute(0,3,1,2)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.ca(y)
        #x = x.permute(0,3,1,2)
        return x + y

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FeedForward_DE(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_DE, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in_x = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_in_y = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        #x = self.project_in(x)
        x = self.project_in_x(x)
        y = self.project_in_y(y)
        xy = torch.cat([y,x],dim=1)
        x1, x2 = self.dwconv(xy).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_DE(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_DE, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        #self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x, y):
        b,c,h,w = x.shape
        q = self.q_conv(x)
        k = self.k_conv(y)
        v = self.v_conv(y)
        qkv = self.qkv_dwconv(torch.cat([q,k,v],dim=1))
        #qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################

class TransformerBlock_EN(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_EN, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        #print("EN")
        x_identity = x
        ta = self.attn(self.norm1(x))
        sg = self.ffn(self.norm2(x))
        #x = x + self.attn(self.norm1(x))
        #x = x + self.ffn(self.norm2(x))
        out = self.project_out(torch.cat([ta,sg],dim=1))
        out = out+x_identity
        return out

class TransformerBlock_DE(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_DE, self).__init__()
        #print("DE!")
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.cross_attn = Attention_DE(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.norm5 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_DE(dim, ffn_expansion_factor, bias)
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
    def forward(self, x, global_M):
        #print("DE")
        x_identity = x
        x = self.attn(self.norm1(x))
        x = x + x_identity
        x_identity = x
        ta = self.cross_attn(self.norm2(x), self.norm4(global_M))
        sg = self.ffn(self.norm3(x), self.norm5(global_M))
        out = self.project_out(torch.cat([ta,sg],dim=1))
        out = out + x_identity

        return out

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class LocalMergingUnit(nn.Module):
    ##Linear along H
    def __init__(self, n_dim, scale_factor=1):
        super(LocalMergingUnit, self).__init__()
        self.n_dim = n_dim
        self.scale_factor=scale_factor
        #if scale_factor!=1:
        #    self.up1 = nn.ConvTranspose2d(128,n_dim,kernel_size=4, stride=1, padding=1)
        #    self.up2 = nn.ConvTranspose2d(128,n_dim,kernel_size=4, stride=1, padding=1)
        #    self.up3 = nn.ConvTranspose2d(128,n_dim,kernel_size=4, stride=1, padding=1)
        #nChannels_ = nChannels
        self.conv1 = nn.Conv2d(128,n_dim,kernel_size=1)
        self.conv2 = nn.Conv2d(128,n_dim,kernel_size=1)
        self.conv3 = nn.Conv2d(128,n_dim,kernel_size=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_feats, local_feats):
        if self.scale_factor==1:
            local_1 = self.conv1(local_feats[0])
            local_2 = self.conv2(local_feats[1])
            local_3 = self.conv3(local_feats[2])
        else:
            local_1 = self.conv1(F.interpolate(local_feats[0],scale_factor=self.scale_factor))
            local_2 = self.conv2(F.interpolate(local_feats[1],scale_factor=self.scale_factor))
            local_3 = self.conv3(F.interpolate(local_feats[2],scale_factor=self.scale_factor))
        local_1 = self.sigmoid(local_1)
        local_2 = self.sigmoid(local_2)
        local_3 = self.sigmoid(local_3)
        input_feat1 = input_feats[0]*local_1
        input_feat2 = input_feats[1]*local_2
        input_feat3 = input_feats[2]*local_3

        local_merged = input_feat1 + input_feat2 + input_feat3
        return local_merged

class GlobalMergingUnit(nn.Module):
    ##Linear along H
    def __init__(self, n_dim):
        super(GlobalMergingUnit, self).__init__()
        self.n_dim=n_dim
        #nChannels_ = nChannels
        self.global_vectors = nn.ModuleList([])
        for i in range(3):
            self.global_vectors.append(nn.Conv2d(128, n_dim, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_feats, global_feats):
        global_attention_vectors = [self.global_vectors[0](global_feats[0].unsqueeze(-1).unsqueeze(-1)), 
        self.global_vectors[1](global_feats[1].unsqueeze(-1).unsqueeze(-1)), 
        self.global_vectors[2](global_feats[2].unsqueeze(-1).unsqueeze(-1))]
        
        #freq_attention_vectors = []
        global_attention_vectors = torch.cat(global_attention_vectors, dim=1)
        global_attention_vectors = global_attention_vectors.view(-1, 3, self.n_dim, 1, 1)
        global_attention_vectors = self.softmax(global_attention_vectors)
        #print(input_feats[0].shape)
        global_merged = torch.sum(torch.stack(input_feats, dim=1) * global_attention_vectors, dim=1)
        return global_merged
        
class MultiscaleAlign(nn.Module):
    def __init__(self, dim):
        super(MultiscaleAlign, self).__init__()
        self.act = nn.LeakyReLU()
        self.reduce = nn.Conv2d(128, dim,kernel_size=1)
        self.down1_ref = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.down1_nonref = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

        self.down2_ref = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.down2_nonref = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        #self.upsample_x2 = F.interpolate(scale_factor=2)
        #self.upsample_x4 = F.interpolate(scale_factor=4)
        #self.downsample_x2 = F.interpolate(scale_factor=0.5)
        #self.downsample_x4 = F.interpolate(scale_factor=0.25)

        #self.conv_inp_local = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_nonref = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_ref = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_mask_x4 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_mask_x2 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_mask_x1 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_x2 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv_x1 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        #self.reduce_conv = nn.Conv2d(64,dim,kernel_size=1)
    def forward(self, inp, ref, local_feat_inp, local_feat_ref):
        local_feat_inp = self.reduce(local_feat_inp)
        local_feat_ref = self.reduce(local_feat_ref)

        inp_down_x2 = self.act(self.down1_nonref(inp))
        ref_down_x2 = self.act((self.down1_ref(ref)))
        inp_down_x4 = self.act(self.down2_nonref(inp_down_x2))
        ref_down_x4 = self.act(self.down2_ref(ref_down_x2))

        x4_nonref = self.act(self.conv_nonref(torch.cat([inp_down_x4,local_feat_inp],dim=1)))
        x4_ref = self.act(self.conv_ref(torch.cat([ref_down_x4,local_feat_ref],dim=1)))

        x4_mask = self.act(self.conv_mask_x4(torch.cat([x4_ref,x4_nonref],dim=1)))
        x4_mask = F.interpolate(x4_mask, scale_factor=2)

        x2_nonref = self.act(self.conv_x2(torch.cat([inp_down_x2, x4_mask],dim=1)))
        x2_mask = self.act(self.conv_mask_x2(torch.cat([ref_down_x2,x2_nonref],dim=1)))
        x2_mask = F.interpolate(x2_mask, scale_factor=2)

        x_nonref = self.act(self.conv_x1(torch.cat([inp, x2_mask],dim=1)))
        x_mask = self.conv_mask_x1(torch.cat([ref,x_nonref],dim=1))


        feat = F.sigmoid(x_mask)
        return inp * feat
        
class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.act = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    def forward(self, inp, ref):
        cat = torch.cat([inp,ref], dim=1)
        feat = self.conv1(cat)
        feat = self.act(feat)
        feat = self.conv2(feat)
        feat = F.sigmoid(feat)
        return inp * feat
    
class RFGViT(nn.Module):
    def __init__(self, 
        inp_channels=6, 
        dim = 32,
        num_blocks = [1,1,1,4], 
        num_refinement_blocks = 4,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(RFGViT, self).__init__()

        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed3 = OverlapPatchEmbed(inp_channels, dim)

        self.sp_att1 = MultiscaleAlign(dim)
        self.sp_att3 = MultiscaleAlign(dim)

        self.encoder_level1_1 = nn.Sequential(*[TransformerBlock_EN(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_2 = nn.Sequential(*[TransformerBlock_EN(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_3 = nn.Sequential(*[TransformerBlock_EN(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        

        self.encoder_level2_1 = nn.Sequential(*[TransformerBlock_EN(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_2 = nn.Sequential(*[TransformerBlock_EN(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_3 = nn.Sequential(*[TransformerBlock_EN(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.encoder_level3_1 = nn.Sequential(*[TransformerBlock_EN(dim=dim*4, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_2 = nn.Sequential(*[TransformerBlock_EN(dim=dim*4, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_3 = nn.Sequential(*[TransformerBlock_EN(dim=dim*4, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down1_1 = Downsample(dim) ## From Level 1 to Level 2
        self.down1_2 = Downsample(dim)
        self.down1_3 = Downsample(dim)
            
        self.down2_1 = Downsample(dim*2) ## From Level 1 to Level 2
        self.down2_2 = Downsample(dim*2) ## From Level 1 to Level 2
        self.down2_3 = Downsample(dim*2) ## From Level 1 to Level 2

        self.down3_1 = Downsample(dim*4) ## From Level 1 to Level 2
        self.down3_2 = Downsample(dim*4) ## From Level 1 to Level 2
        self.down3_3 = Downsample(dim*4) ## From Level 1 to Level 2

        self.up3 = Upsample(dim*8)
        self.up2 = Upsample(dim*4)
        self.up1 = Upsample(dim*2)
        self.bottleneck = nn.ModuleList([TransformerBlock_EN(dim=dim*4, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.reduce2 = nn.Conv2d(dim*4, dim*2, kernel_size=1, bias=bias)
        self.reduce1 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.decoder_level3 = TransformerBlock_DE(dim=dim*4, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.decoder_level2 = TransformerBlock_DE(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.decoder_level1 = TransformerBlock_DE(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.refinement = nn.ModuleList([TransformerBlock_EN(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.global_merge1 = GlobalMergingUnit(dim)
        self.global_merge2 = GlobalMergingUnit(dim*2)
        self.global_merge3 = GlobalMergingUnit(dim*4)
        self.local_merge1 = LocalMergingUnit(dim,scale_factor=4)
        self.local_merge2 = LocalMergingUnit(dim*2,scale_factor=2)
        self.local_merge3 = LocalMergingUnit(dim*4)
    def forward(self, x1, x2, x3, local_feats, global_feats ):

        F1_1 = self.patch_embed1(x1)
        F2_1 = self.patch_embed2(x2)
        F3_1 = self.patch_embed3(x3)

        F1_1 = self.sp_att1(F1_1, F2_1, local_feats[0], local_feats[1])
        F3_1 = self.sp_att3(F3_1, F2_1, local_feats[2], local_feats[1])


        F1_1 = self.encoder_level1_1(F1_1)
        G1_1 = F1_1
        F2_1 = self.encoder_level1_2(F2_1)
        G2_1 = F2_1
        F3_1 = self.encoder_level1_3(F3_1)
        G3_1 = F3_1

        #LM_1 = self.local_merge1([L1_1,L2_1,L3_1], local_feats)
        GM_1 = self.global_merge1([G1_1,G2_1,G3_1], global_feats)

        F1_2 = self.down1_1(F1_1)
        F2_2 = self.down1_2(F2_1)
        F3_2 = self.down1_3(F3_1)

        F1_2 = self.encoder_level2_1(F1_2)
        G1_2 = F1_2
        F2_2 = self.encoder_level2_2(F2_2)
        G2_2 = F2_2
        F3_2 = self.encoder_level2_3(F3_2)
        G3_2 = F3_2

        #LM_2 = self.local_merge2([L1_2,L2_2,L3_2], local_feats)
        GM_2 = self.global_merge2([G1_2,G2_2,G3_2], global_feats)

        F1_3 = self.down2_1(F1_2)
        F2_3 = self.down2_2(F2_2)
        F3_3 = self.down2_3(F3_2)

        # F1_3, L1_3, G1_3 = self.encoder_level3_1(F1_3)
        # F2_3, L2_3, G2_3 = self.encoder_level3_2(F2_3)
        # F3_3, L3_3, G3_3 = self.encoder_level3_3(F3_3)

        # LM_3 = self.local_merge3([L1_3,L2_3,L3_3], local_feats)
        # GM_3 = self.global_merge3([G1_3,G2_3,G3_3], global_feats)

        # F1_3 = self.down3_1(F1_3)
        # F2_3 = self.down3_2(F2_3)
        # F3_3 = self.down3_3(F3_3)

        F1_3 = self.encoder_level3_1(F1_3)
        F2_3 = self.encoder_level3_2(F2_3)
        F3_3 = self.encoder_level3_3(F3_3)

        FM_3 = F1_3+F2_3+F3_3

        for bottleneck in self.bottleneck:
            FM_3 = bottleneck(FM_3)
        FM_3 = FM_3 + F2_3
        # FM_2 = self.up3(FM_3)

        # FM_3 = self.decoder_level3(FM_3, LM_3, GM_3)

        FM_2 = self.up2(FM_3)
        FM_2 = self.decoder_level2(FM_2, GM_2) + F2_2
        FM_1 = self.up1(FM_2)
        FM_1 = self.decoder_level1(FM_1, GM_1) + F2_1
        for refinement in self.refinement:
            FM_1 = refinement(FM_1)
        out = self.output(FM_1)
        out = F.sigmoid(out)
        return out
