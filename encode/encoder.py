import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
import numpy as np

from utils.model_utils.net_utils import get_norm_layer
from utils.depth_utils.resize import SimpleResizeCNN

from transformers import AutoModel, AutoConfig
from PIL import Image
from timm.data.transforms_factory import create_transform

def make_encoder(conf, **kwargs):
    enc_type = conf['encoder_type']  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net


# 定义卷积和上采样模块
class FeatureAdjust(nn.Module):
    def __init__(self, in_channels, out_channels, target_size):
        super(FeatureAdjust, self).__init__()
        # 将输入通道数调整为 256
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 上采样到目标尺寸 [242, 324]
        self.upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 先进行卷积调整通道数，再进行上采样
        x = self.conv(x)
        x = self.upsample(x)
        return x

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        use_diffu_prior = True
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()
        

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        self.use_diffu_prior = use_diffu_prior 
        norm_layer = get_norm_layer(norm_type)
        # self.Mamba_feature_weights_2 = nn.Parameter(
        #     torch.clamp(
        #         torch.empty(1).uniform_(0.001, 0.999), min=0.001, max=0.999
        #     )
        # )


        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            exit()
        else:
            # print("Using torchvision", backbone, "encoder")
            # self.model = getattr(torchvision.models, backbone)(
            #     pretrained=pretrained, norm_layer=norm_layer
            # )
            # # Following 2 lines need to be uncommented for older configs
            # self.model.fc = nn.Sequential()
            # self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
            # 加载模型配置文件，不加载预训练权重
            # 只加载配置文件，不加载预训练权重
            # config = AutoConfig.from_pretrained(
            #     "nvidia/MambaVision-B-1K", trust_remote_code=True
            # )
            # self.model = AutoModel.from_config(config, trust_remote_code=True)
            self.model = AutoModel.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)
            # local_model_path = "/work/SSR/luoxi/SSR-code/MambaVision-B-1K"
            # model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True).cuda()
            # model = AutoModelForImageClassification.from_pretrained(
            #     local_model_path, 
            #     trust_remote_code=True
            # ).cuda().eval()
        
        if self.use_diffu_prior:
            self.model_D = SimpleResizeCNN()
            self.depth_weight = nn.Parameter(torch.tensor(0.5))  # 定义可训练的权重参数
            # self.diffu_weight = 0

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )        

        # self.latent (B, L, H, W)
    
    def index(self, uv, cam_z=None, image_size=(), diffu_prior=None, roi_feat=None, z_bounds=None, offset_xy=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N_uv, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :param offset_xy, if use deformable attention, x y offset, [-1, 1]
        :param sample_size, if use deformable attention, get sample_size pixel img feature mean
        :return (B, L, N) L is latent size
        """
        if self.use_diffu_prior:
            diffu_prior = diffu_prior.cuda().to(torch.float32)
            self.diffu_latent = self.model_D(diffu_prior)
            # self.latent_mix = self.diffu_weight * self.diffu_latent + (1-self.diffu_weight) * self.latent
            self.latent_mix = self.diffu_latent


        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N_uv, 1, 2)

            if roi_feat is not None:
                samples = F.grid_sample(                # grid_sample, uv[0] --> x --> img_W, uv[1] --> y --> img_H
                    roi_feat,
                    uv,
                    align_corners=True,
                    mode=self.index_interp,
                    padding_mode=self.index_padding,
                )

            else:
                samples = F.grid_sample(                # grid_sample, uv[0] --> x --> img_W, uv[1] --> y --> img_H
                    self.latent_mix,
                    uv,
                    align_corners=True,
                    mode=self.index_interp,
                    padding_mode=self.index_padding,
                )           # not deform attention: (B, C, N, 1); use deform attention: (B, C, N, sample_size)

            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )

        x = x.to(device=self.latent.device)

        out_avg_pool, features = self.model(x)
        target_size = (242, 324)
        # target_size = (192, 192) --> the other datasets size


        # 定义每个特征层的调整
        # adjust1 = FeatureAdjust(128, 256, target_size).cuda()
        # # adjust2 = FeatureAdjust(256, 256, target_size).cuda()
        # adjust3 = FeatureAdjust(512, 256, target_size).cuda()
        # adjust4 = FeatureAdjust(1024, 256, target_size).cuda()

        # 将每层特征调整到目标尺寸 [12, 256, 242, 324]
        # f1 = adjust1(features[0])
        # f2 = adjust1(features[1])
        f2 = F.interpolate(features[1], size=target_size, mode='bilinear', align_corners=False)
        # f3 = adjust3(features[2])
        # f4 = adjust4(features[3])

        # 使用可训练的权重进行自适应融合
        latent = f2 

        self.latent = latent

        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf["backbone"],
            pretrained=conf["pretrained"],
            num_layers=conf["num_layers"],
            index_interp="bilinear",                # default value
            index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            use_diffu_prior = conf['use_diffu_prior']  # 判断是否使用Diffusion Prior
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf["encoder"]["backbone"],
            pretrained=conf["encoder"]["pretrained"],
            latent_size=conf["latent_feature_dim"],
        )
