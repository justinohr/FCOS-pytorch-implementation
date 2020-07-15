import math
import torch
import torch.nn as nn

from detectron2.layers import Conv2d, DeformConv, ShapeSpec
from fcos.layers import Scale, normal_init
from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FCOSHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    In our Implementation, schemetic structure is as following:

                                    /-> logits
                    /-> cls convs ->
                   /                \-> centerness
    shared convs ->
                    \-> reg convs -> regressions
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_channels       = input_shape[0].channels
        self.num_classes       = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides       = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_shared_convs  = cfg.MODEL.FCOS.NUM_SHARED_CONVS
        self.num_stacked_convs = cfg.MODEL.FCOS.NUM_STACKED_CONVS
        self.prior_prob        = cfg.MODEL.FCOS.PRIOR_PROB
        self.use_deformable    = cfg.MODEL.FCOS.USE_DEFORMABLE
        self.norm_layer        = cfg.MODEL.FCOS.NORM
        self.ctr_on_reg        = cfg.MODEL.FCOS.CTR_ON_REG
        # fmt: on

        self._init_layers()
        self._init_weights()
        

    def _init_layers(self):
        """
        Initializes six convolutional layers for FCOS head and a scaling layer for bbox predictions.
        """
        activation = nn.ReLU()

        # constructing the convolution layers in head
        self.shared_convs = self.make_shared_convs()
        self.cls_convs = nn.Sequential(
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU())
        self.reg_convs = nn.Sequential(
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU())
        # activation function is not needed for the last layers,
        # because we will handle them at inference stage if one is needed
        self.cls_logits = nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        # making the scales learnable parameters
        # each variable corresponds to trainable scalars of P3, P4, P5, P6 and P7
        self.scales = torch.Tensor([1.0 for i in range(5)]).requires_grad_(True).to(device)

    def make_shared_convs(self):
        # if num_shared_convs is zero, None is returned
        if self.num_shared_convs == 0:
            return None

        # one conv layer and one ReLU activation function are added "num_shared_convs" times to layers
        layers = []
        for i in range(self.num_shared_convs):
            if not layers:
                layers += [nn.Conv2d(self.in_channels,256, kernel_size=3, stride=1, padding=1)]
            else:
                layers += [nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)]
            layers += nn.ReLU()
        return nn.Sequential(*layers)

    def _init_weights(self):
        for modules in [
            self.shared_convs, self.cls_convs, self.reg_convs,
            self.cls_logits, self.bbox_pred, self.centerness
        ]:
            # weight initialization with mean=0, std=0.01
            # if module is None, skip it
            if modules == None:
                continue
            # if module is nn.Sequential, initialize weight of Conv2d only
            # i.e. skip it if the component is activation layer
            elif isinstance(modules, nn.Sequential):
                for component in modules:
                    if isinstance(component, nn.Conv2d):
                        normal_init(component, mean=0, std=0.01)
            else:
                normal_init(modules, mean=0, std=0.01)

        # initialize the bias for classification logits
        # detailed calculation of bias_cls can be found on report
        bias_cls = math.log(self.prior_prob/ (1-self.prior_prob))  # calculate proper value that makes cls_probability with `self.prior_prob`
        # In other words, make the initial 'sigmoid' activation of cls_logits as `self.prior_prob`
        # by controlling bias initialization
        nn.init.constant_(self.cls_logits.bias, bias_cls)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            cls_scores (list[Tensor]): list of #feature levels, each has shape (N, C, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of C object classes.
            bbox_preds (list[Tensor]): list of #feature levels, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (l, t, r, b) box regression values for
                every position of featur map. These values are the distances from
                a specific point to each (left, top, right, bottom) edge
                of the corresponding ground truth box that the point belongs to.
            centernesses (list[Tensor]): list of #feature levels, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness logits, where these values used to
                downweight the bounding box scores far from the center of an object.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        for feat_level, feature in enumerate(features):
            x = feature
            # if there are shared convolution layers, forward the x through them
            if self.shared_convs != None:
                x = self.shared_convs(x)
            # the left below is same as the diagram in line 24-28
            cls_output = self.cls_convs(x)

            classification = self.cls_logits(cls_output)
            cent = self.centerness(cls_output)
            
            regr = self.bbox_pred(self.reg_convs(x))
            cls_scores.append(classification)
            # for bounding box prediction we preict them in log scales
            # since each feature has different scale between each other,
            # different scalar is given for each feature level
            bbox_preds.append(torch.exp(self.scales[feat_level] * regr))
            centernesses.append(cent)

        return cls_scores, bbox_preds, centernesses