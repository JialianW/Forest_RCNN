import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead


@HEADS.register_module
class RetinaHead(AnchorHead):
    """
    An anchor-based head used in [1]_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        if self.use_forest:
            self.parent_cls_convs = nn.ModuleList()
            self.parent_cls = nn.ModuleList()
            for cls_num in self.all_classes_num[:-1]:
                self.parent_cls_convs.append(
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.parent_cls.append(nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * (cls_num - 1),
                    3,
                    padding=1))


    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

        if self.use_forest:
            for m in self.parent_cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.parent_cls:
                bias_cls = bias_init_with_prob(0.01)
                normal_init(m, std=0.01, bias=bias_cls)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        i = 1
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
            if self.use_forest:
                if i == self.stacked_convs - 1:
                    cls_feat_mid = cls_feat.detach()
            i += 1
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.retina_reg(reg_feat)

        if self.use_forest:
            cls_score = []
            # parent
            for layer1, layer2 in zip(self.parent_cls_convs, self.parent_cls):
                cls_score.append(layer2(layer1(cls_feat_mid)))
            # fine-grained
            cls_score.append(self.retina_cls(cls_feat))
        else:
            cls_score = self.retina_cls(cls_feat)

        return cls_score, bbox_pred
