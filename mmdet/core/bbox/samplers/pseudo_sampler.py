import torch

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult
import numpy as np


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)

        cls_scores = kwargs['cls_scores']
        cls_scores, _ = torch.max(cls_scores, dim=1, keepdim=True)
        proposals = torch.cat((bboxes, cls_scores), dim = 1)
        gt_labels = kwargs['gt_labels']
        pos_labels = gt_labels[sampling_result.pos_assigned_gt_inds]
        pos_proposals = proposals[pos_inds]
        keep = self.nms_resampling_linear(pos_proposals, pos_labels)
        sampled_pos_inds = pos_inds[keep]
        sampling_result = SamplingResult(sampled_pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

    def nms_resampling_linear(self, proposals, proposals_labels):
        assert any(proposals_labels > 0)
        assert proposals.shape[0] == proposals_labels.shape[0]

        thresh = np.load('thresh_discrete.npy')

        proposals_labels = proposals_labels.detach().cpu().numpy()
        t = thresh[proposals_labels-1]
        keep = self.nms_py(proposals.detach().cpu().numpy(), t)
        keep = np.array(keep)

        # return proposals[keep, :]
        return keep

    def nms_py(self, dets, thresh):
        """
        greedily select boxes with high confidence and overlap with current maximum <= thresh
        rule out overlap >= thresh
        :param dets: [[x1, y1, x2, y2 score]]
        :param thresh: retain overlap < thresh
        :return: indexes to keep
        """
        if dets.shape[0] == 0:
            return []
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh[i])[0]
            order = order[inds + 1]
        return keep

