import warnings
import torch.nn as nn
from nnunet.network_architecture.neural_network import SegmentationNetwork

from .blocks import *

__all__ = (
    "VNet",
    "VNet_CSE",
    "VNet_SSE",
    "VNet_SCSE",
    "VNet_ASPP",
    "VNet_MABN",
    "VBNet",
    "VBNet_CSE",
    "VBNet_SSE",
    "VBNet_SCSE",
    "VBNet_ASPP",
    "SKVNet",
    "SKVNet_ASPP",
)


class VNetBase(SegmentationNetwork):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNetBase, self).__init__()
        self.conv_op = nn.Conv3d
        self.num_classes = out_channels
        norm_type = nn.BatchNorm3d
        act_type = nn.ReLU
        se_type = None
        drop_type = None
        feats = [16, 32, 64, 128, 256]
        num_blocks = [1, 2, 3, 3]
        block_name = "residual"
        self._use_aspp = False
        self.do_ds = False
        if "do_ds" in kwargs.keys():
            self.do_ds = kwargs["do_ds"]
        # if 'norm_type' in kwargs.keys():
        #     norm_type = kwargs['norm_type']
        # if 'act_type' in kwargs.keys():
        #     act_type = kwargs['act_type']
        if "feats" in kwargs.keys():
            feats = kwargs["feats"]
        if "se_type" in kwargs.keys():
            se_type = kwargs["se_type"]
        if "num_blocks" in kwargs.keys():
            num_blocks = kwargs["num_blocks"]
        if "drop_type" in kwargs.keys():
            drop_type = kwargs["drop_type"]
        if "use_aspp" in kwargs.keys():
            self._use_aspp = kwargs["use_aspp"]
        if "block_name" in kwargs.keys():
            block_name = kwargs["block_name"]

        self.in_conv = InputBlock(in_channels, feats[0], norm_type=norm_type, act_type=act_type)

        self.down1 = DownBlock(
            feats[0],
            feats[1],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[0],
            block_name=block_name,
        )
        self.down2 = DownBlock(
            feats[1],
            feats[2],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[1],
            block_name=block_name,
        )
        self.down3 = DownBlock(
            feats[2],
            feats[3],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[2],
            block_name=block_name,
        )
        self.down4 = DownBlock(
            feats[3],
            feats[4],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[3],
            block_name=block_name,
        )
        if self._use_aspp:
            self.aspp = ASPP(
                feats[4],
                dilations=[1, 2, 3, 4],
                norm_type=norm_type,
                act_type=act_type,
                drop_type=drop_type,
            )
        self.up4 = UpBlock(
            feats[4],
            feats[3],
            feats[4],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[3],
            block_name=block_name,
        )
        self.up3 = UpBlock(
            feats[4],
            feats[2],
            feats[3],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[2],
            block_name=block_name,
        )
        self.up2 = UpBlock(
            feats[3],
            feats[1],
            feats[2],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[1],
            block_name=block_name,
        )
        self.up1 = UpBlock(
            feats[2],
            feats[0],
            feats[1],
            norm_type=norm_type,
            act_type=act_type,
            se_type=se_type,
            drop_type=drop_type,
            num_blocks=num_blocks[0],
            block_name=block_name,
        )

        self.out_block = OutBlock(feats[1], out_channels, norm_type, act_type)

        self.ds1 = OutBlock(feats[2], out_channels, norm_type, act_type)
        self.ds2 = OutBlock(feats[3], out_channels, norm_type, act_type)
        self.ds3 = OutBlock(feats[4], out_channels, norm_type, act_type)
        self.ds4 = OutBlock(feats[4], out_channels, norm_type, act_type)

        init_weights(self)

    def forward(self, input):
        # print(input.shape)
        # if input.size(2) // 16 == 0:
        #     raise RuntimeError("input tensor shape is too small")
        # elif input.size(3) // 16 == 0:
        #     raise RuntimeError("input tensor shape is too small")
        # elif input.size(4) // 16 == 0:
        #     raise RuntimeError("input tensor shape is too small")
        input = self.in_conv(input)
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        if self._use_aspp:
            down4 = self.aspp(down4)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, input)

        out = self.out_block(up1)

        if self.do_ds:
            ds1 = self.ds1(up2)
            ds2 = self.ds2(up3)
            ds3 = self.ds3(up4)
            ds4 = self.ds4(down4)
            return [out, ds1, ds2, ds3, ds4]
        else:
            return out


class VNet(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet, self).__init__(in_channels, out_channels, **kwargs)


class VNet_MABN(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_MABN, self).__init__(in_channels, out_channels, norm_type=MABN3d, **kwargs)


class VNet_CSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_CSE, self).__init__(in_channels, out_channels, se_type="cse", **kwargs)


class VNet_SSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_SSE, self).__init__(in_channels, out_channels, se_type="sse", **kwargs)


class VNet_SCSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_SCSE, self).__init__(in_channels, out_channels, se_type="scse", **kwargs)


class VNet_ASPP(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_ASPP, self).__init__(in_channels, out_channels, use_aspp=True, **kwargs)


class VBNet(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet, self).__init__(in_channels, out_channels, block_name="bottleneck", **kwargs)


class VBNet_MABN(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_MABN, self).__init__(
            in_channels, out_channels, block_name="bottleneck", norm_type=MABN3d, **kwargs
        )


class VBNet_CSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_CSE, self).__init__(
            in_channels, out_channels, block_name="bottleneck", se_type="cse", **kwargs
        )


class VBNet_SSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_SSE, self).__init__(
            in_channels, out_channels, block_name="bottleneck", se_type="sse", **kwargs
        )


class VBNet_SCSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_SCSE, self).__init__(
            in_channels, out_channels, block_name="bottleneck", se_type="scse", **kwargs
        )


class VBNet_ASPP(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_ASPP, self).__init__(
            in_channels, out_channels, block_name="bottleneck", use_aspp=True, **kwargs
        )


class SKVNet(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        if "se_type" in kwargs.keys():
            warnings.warn("`se_type` keyword not working in `SKVNet`", UserWarning)
        super(SKVNet, self).__init__(in_channels, out_channels, block_name="sk", **kwargs)


class SKVNet_ASPP(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        if "se_type" in kwargs.keys():
            warnings.warn("`se_type` keyword not working in `SKVNet_ASPP`", UserWarning)
        super(SKVNet_ASPP, self).__init__(in_channels, out_channels, block_name="sk", use_aspp=True, **kwargs)
