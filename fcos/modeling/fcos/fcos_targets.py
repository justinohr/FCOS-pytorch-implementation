import torch
import math

from detectron2.layers import cat
from fcos.utils import multi_apply

INF = 100000000


def FCOSTargets(all_level_points, gt_instances, cfg):
    # fmt: off
    num_classes           = cfg.MODEL.FCOS.NUM_CLASSES
    fpn_strides           = cfg.MODEL.FCOS.FPN_STRIDES
    sizes_of_interest     = cfg.MODEL.FCOS.SIZES_OF_INTEREST
    center_sample         = cfg.MODEL.FCOS.CENTER_SAMPLE
    center_radius         = cfg.MODEL.FCOS.POS_RADIUS
    normalize_reg_targets = cfg.MODEL.FCOS.NORMALIZE_REG_TARGETS
    # fmt: on

    regress_ranges = generate_regress_ranges(sizes_of_interest)

    center_sample_cfg = dict(center_sample=center_sample, center_radius=center_radius)

    return fcos_target(
        all_level_points,
        regress_ranges,
        gt_instances,
        fpn_strides,
        center_sample_cfg,
        normalize_reg_targets,
        num_classes=num_classes
    )


def generate_regress_ranges(sizes_of_interest):
    # generate sizes of interest
    regress_ranges = []
    prev_size = -1
    for s in sizes_of_interest:
        regress_ranges.append([prev_size, s])
        prev_size = s
    regress_ranges.append([prev_size, INF])
    return regress_ranges


def get_points(features, fpn_strides):
    """Get points according to feature map sizes.

    Args:
        features (list[Tensor]): Multi-level feature map. Axis 0 represents the number of
            images `N` in the input data; axes 1-3 are channels, height, and width, which
            may vary between feature maps (e.g., if a feature pyramid is used).
        fpn_strides (list[int]): Feature map strides corresponding to each level of multi-level
            feature map.

    Returns:
        points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
    """
    assert len(features) == len(fpn_strides)

    points = []
    for feat, stride in zip(features, fpn_strides):
        featmap_size = feat.size()[-2:]
        points.append(
            # run on single feature-level
            get_points_single(featmap_size, stride, feat.device))
    return points


def get_points_single(featmap_size, stride, device):
    """point prediction per feature-level.

    Args:
        featmap_size (Tuple): feature map size (Hi, Wi) where 'i' denotes specific feature level.
        stride (list[int]): feature map stride corresponding to each feature level 'i'.
        device: the same device type with feature map tensor.

    Returns:
        points (Tensor): Tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi)
            of all feature map locations on feature-level 'i' in image coordinate.
    """

    # this piece of codes map the points in feature maps to
    # the corresponding point in the original size of image
    points = torch.zeros(featmap_size[0] * featmap_size[1], 2)
    for y in range(featmap_size[0]):
        for x in range(featmap_size[1]):
            points[featmap_size[1]*y + x][0] = x*stride + int(stride/2)
            points[featmap_size[1]*y + x][1] = y*stride + int(stride/2)
    return points.to(device)


def fcos_target(
    points,
    regress_ranges,
    gt_instance_list,
    fpn_strides,
    center_sample_cfg,
    normalize_reg_targets,
    num_classes=80
):
    """Compute class labels and regression targets for every feature points on all feature levels.

    Args:
        points (list[Tensor]): list of #feature levels.
            Each entry contains tensor of size (N*Hi*Wi, )
        regress_ranges (list[tuple]): list of #feature levels. Each entry denotes the
            lower bound and upper bound of regression range of bbox target
            for the corresponding feature level.
        gt_instance_list (list[Instances]): a length `N` list of `Instances`s.
            Each `Instances` stores ground-truth instances for the corresponding image.
        fpn_strides (list[int]): list of #feature levels.
        center_sample_cfg (dict): hyperparameters for center sampling.
        normalize_reg_targets (bool): whether to normalize regression targets by each stride of
            corresponding feature stride.
        num_classes (int)

    Returns:
        concat_labels (list[Tensor]): list of #feature levels. Each entry contains
            tensor of size (N*Hi*Wi, )
        concat_bbox_targets (list[Tensor]): list of #feature levels. Each entry contains
            tensor of size (N*Hi*Wi, 4)
    """
    assert len(points) == len(regress_ranges)
    num_levels = len(points)

    # expand regress ranges to align with points
    expanded_regress_ranges = [
        points[i].new_tensor(regress_ranges[i])[None].expand_as(points[i])
        for i in range(num_levels)
    ]

    # concat all levels points and regress ranges
    concat_regress_ranges = cat(expanded_regress_ranges, dim=0)
    concat_points = cat(points, dim=0)

    # the number of points per img, per level
    num_points = [center.size(0) for center in points]

    # get labels and bbox_targets of each image; per-image computation.
    labels_list, bbox_targets_list = multi_apply(
        fcos_target_single_image,
        gt_instance_list,
        points=concat_points,
        regress_ranges=concat_regress_ranges,
        num_points_per_level=num_points,
        fpn_strides=fpn_strides,
        center_sample_cfg=center_sample_cfg,
        normalize_reg_targets=normalize_reg_targets,
        num_classes=num_classes
    )

    # split to per img, per feature level
    labels_list = [labels.split(num_points, 0) for labels in labels_list]
    bbox_targets_list = [
        bbox_targets.split(num_points, 0)
        for bbox_targets in bbox_targets_list
    ]

    # concat per level image
    concat_labels = []
    concat_bbox_targets = []
    for i in range(num_levels):
        concat_labels.append(
            cat([labels[i] for labels in labels_list])
        )

        if normalize_reg_targets:
            # we normalize reg_targets by FPN's strides here
            normalizer = float(fpn_strides[i])
        else:
            normalizer = 1.0

        concat_bbox_targets.append(
            cat([bbox_targets[i] / normalizer for bbox_targets in bbox_targets_list])
        )
    return concat_labels, concat_bbox_targets


def fcos_target_single_image(
    gt_instances,
    points,
    regress_ranges,
    num_points_per_level,
    fpn_strides,
    center_sample_cfg,
    normalize_reg_targets,
    num_classes=80
):
    """Compute class labels and regression targets for single image.

    Args:
        gt_instances (Instances): stores ground-truth instances for the corresponding image.
        all other args are the same as in `self.fcos_target` where all elements in the list
            are concatenated to form a single tensor.

    Returns:
        labels (Tensor): class label of every feature point in all feature levels for single image.
        bbox_targets (Tensor): regression targets of every feature point in all feature levels
            for a single image. each column corresponds to a tensor shape of (l, t, r, b).
    """

    center_sample = center_sample_cfg['center_sample']
    center_radius = center_sample_cfg['center_radius']

    # here, num_points accumulates all locations across all feature levels.
    num_points = points.size(0)
    num_gts = len(gt_instances)

    # get class labels and bboxes from `gt_instances`.
    gt_labels = gt_instances.get(name='gt_classes')
    gt_bboxes = gt_instances.get(name='gt_boxes')

    if num_gts == 0:
        return (
            gt_labels.new_zeros(num_points) + num_classes,
            gt_bboxes.new_zeros((num_points, 4))
        )

    # `areas`: should be `torch.Tensor` shape of (num_points, num_gts, 1)
    areas = gt_bboxes.area()  # 1. `torch.Tensor` shape of (num_gts, 1)
    areas = areas.unsqueeze(1).repeat(num_points, 1, 1)  # 2. hint: use :func:`torch.repeat`.

    # `regress_ranges`: should be `torch.Tensor` shape of (num_points, num_gts, 2)
    regress_ranges = regress_ranges.unsqueeze(1).expand(num_points, num_gts, 2)  # hint: use :func:`torch.expand`.

    # `gt_bboxes`: should be `torch.Tensor` shape of (num_points, num_gts, 4)
    gt_bboxes = gt_bboxes.tensor.unsqueeze(0).expand(num_points, num_gts, 4)  # hint: use :func:`torch.expand`.

    # align each coordinate  component xs, ys in shape as (num_points, num_gts)
    xs, ys = points[:, 0], points[:, 1]
    xs = xs.unsqueeze(1).expand(num_points, num_gts)  # hint: use :func:`torch.expand`.
    ys = ys.unsqueeze(1).expand(num_points, num_gts)  # hint: use :func:`torch.expand`.

    # distances to each four side of gt bboxes.
    # The equations correspond to equation(1) from FCOS paper.
    left =  xs - gt_bboxes[...,0]
    right = gt_bboxes[...,2] - xs
    top = ys - gt_bboxes[...,1]
    bottom = gt_bboxes[...,3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)

    
    # condition1: a point should be inside a gt bbox
    inside_gt_bbox_mask = torch.min(bbox_targets, axis=2).values > 0
    # to match the shape with areas, unsqueeze the mask
    inside_gt_bbox_mask.unsqueeze_(2)
    
    # condition2: limit the regression range for each location
    max_regress_distance = torch.max(bbox_targets, axis=2).values   # hint: use :func:`torch.max`.

    # The mask whether `max_regress_distance` on every points is bounded
    #   between the side values regress_ranges.
    # See section 3.2 3rd paragraph on FCOS paper.
    # to match the shape with areas, unsqueeze the mask
    inside_regress_range = (max_regress_distance > regress_ranges[...,0]) & (max_regress_distance < regress_ranges[...,1])
    inside_regress_range.unsqueeze_(2)

    # filter areas that violate condition1 and condition2 above.
    areas[~inside_gt_bbox_mask] = INF
    areas[~inside_regress_range] = INF

    # If there are still more than one objects for a location,
    # we choose the one with minimal area across `num_gts` axis.
    min_area, min_area_inds = torch.min(areas, axis=1)

    # ground-truth assignments w.r.t. bbox area indices
    labels = gt_labels[min_area_inds]

    labels[min_area == INF] = num_classes
    # to make single row tensor, squeeze the tensor below 
    labels.squeeze_(1)

    # to make tensor shape of (num_point, 4), squeeze the min_area_inds
    bbox_targets = bbox_targets[range(num_points), min_area_inds.squeeze(1)]
    
    return labels, bbox_targets


def compute_centerness_targets(pos_bbox_targets):
    """Compute centerness targets for every feature points, given bbox targets.

    Args:
        pos_bbox_targets (Tensor): regression targets of every positive feature point in all
            feature levels and for all images. Each column corresponds to a tensor shape of
            (l, t, r, b). shape of (num_pos_samples, 4)

    Returns:
        centerness_targets (Tensor): A tensor with same rows from 'pos_bbox_targets' Tensor.
    """

    centerness_targets = pos_bbox_targets.new_zeros(pos_bbox_targets.size(0), 1)
    for i in range(len(centerness_targets)):
        l = pos_bbox_targets[i][0]
        t = pos_bbox_targets[i][1]
        r = pos_bbox_targets[i][2]
        b = pos_bbox_targets[i][3]

        centerness_targets[i][0] = math.sqrt(min(l,r) * min(t,b) / (max(l,r) * max(t,b)))
    # to make single row tensor, squeeze the tensor below
    centerness_targets.squeeze_(1)

    return centerness_targets
