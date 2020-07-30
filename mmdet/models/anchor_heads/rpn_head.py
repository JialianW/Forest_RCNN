import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..registry import HEADS
from .anchor_head import AnchorHead
from mmdet.core.bbox.geometry import bbox_overlaps
import numpy as np

@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          gt_bboxes,
                          gt_labels,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)

            if cfg.nms_resampling is not None:  # only used in training
                if cfg.nms_resampling[0] == 'discrete':
                    a_r = cfg.nms_resampling[1]
                    a_c = cfg.nms_resampling[2]
                    a_f = cfg.nms_resampling[3]
                    proposals = self.nms_resampling_discrete(proposals, gt_bboxes, gt_labels, a_r, a_c, a_f)
                elif cfg.nms_resampling[0] == 'linear':
                    thresh = cfg.nms_resampling[1]
                    proposals = self.nms_resampling_linear(proposals, gt_bboxes, gt_labels, thresh)
            else:
                proposals, _ = nms(proposals, cfg.nms_thr)

            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

    def nms_resampling_linear(self, proposals, gt_bboxes, gt_labels, thresh):
        assert any(gt_labels>0)
        iou = bbox_overlaps(proposals[:, :4], gt_bboxes)
        max_iou, gt_assignment = iou.max(dim=1)
        proposals_labels = gt_labels[gt_assignment]
        # proposal is considered as background when its iou with gt < 0.3
        proposals_labels[max_iou < 0.3] = 0

        proposals_labels = proposals_labels.cpu().numpy()
        t = thresh[proposals_labels]
        keep = self.nms_py(proposals.cpu().numpy(), t)
        keep = np.array(keep)

        return proposals[keep, :]

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


    def nms_resampling_discrete(self, proposals, gt_bboxes, gt_labels, a_r, a_c, a_f):
        assert any(gt_labels>0)
        # proposal is considered as background when its iou with gt < 0.3
        select_thresh = 0.3
        out= []

        rare, common, frequent = self.get_category_frequency(gt_labels.device)
        rare_gtbox = torch.zeros((2000, 4), device=gt_labels.device)
        rare_gtbox_idx = 0
        common_gtbox = torch.zeros((2000, 4), device=gt_labels.device)
        common_gtbox_idx = 0
        frequent_gtbox = torch.zeros((2000, 4), device=gt_labels.device)
        frequent_gtbox_idx = 0
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            if gt_label in rare:
                rare_gtbox[rare_gtbox_idx, ...] = gt_bbox
                rare_gtbox_idx += 1
            elif gt_label in common:
                common_gtbox[common_gtbox_idx, ...] = gt_bbox
                common_gtbox_idx += 1
            else:
                frequent_gtbox[frequent_gtbox_idx, ...] = gt_bbox
                frequent_gtbox_idx += 1
        rare_gtbox = rare_gtbox[:rare_gtbox_idx, ...]
        common_gtbox = common_gtbox[:common_gtbox_idx, ...]

        frequent_proposals, _ = nms(proposals, a_f)
        if len(rare_gtbox) > 0:
            rare_proposals, _ = nms(proposals, a_r)
            rare_overlaps = bbox_overlaps(rare_gtbox, rare_proposals[:, :4])
            rare_max_overlaps, rare_argmax_overlaps = rare_overlaps.max(dim=0)
            rare_pos_inds = rare_max_overlaps >= select_thresh
            rare_proposals = rare_proposals[rare_pos_inds, :]
            out.append(rare_proposals)

            frequent_rare_overlaps = bbox_overlaps(rare_gtbox, frequent_proposals[:, :4])
            frequent_rare_max_overlaps, frequent_rare_argmax_overlaps = frequent_rare_overlaps.max(dim=0)
            valid_inds = frequent_rare_max_overlaps < select_thresh
            frequent_proposals = frequent_proposals[valid_inds, :]
        if len(common_gtbox) > 0:
            common_proposals, _ = nms(proposals, a_c)
            common_overlaps = bbox_overlaps(common_gtbox, common_proposals[:, :4])
            common_max_overlaps, common_argmax_overlaps = common_overlaps.max(dim=0)
            common_pos_inds = common_max_overlaps >= select_thresh
            common_proposals = common_proposals[common_pos_inds, :]
            out.append(common_proposals)

            frequent_common_overlaps = bbox_overlaps(common_gtbox, frequent_proposals[:, :4])
            frequent_common_max_overlaps, frequent_common_argmax_overlaps = frequent_common_overlaps.max(dim=0)
            valid_inds = frequent_common_max_overlaps < select_thresh
            frequent_proposals = frequent_proposals[valid_inds, :]
        out.append(frequent_proposals)
        if len(out) > 1:
            out_proposals = torch.cat(out, 0)
        else:
            out_proposals = frequent_proposals

        return out_proposals

    def get_category_frequency(self, device):
        # rare, common, frequent are defined by the LVIS v0.5 dataset
        rare = torch.tensor([1, 7, 10, 14, 15, 16, 21, 22, 31, 38, 39, 40, 42, 46, 49, 51, 52,
                64, 65, 70, 72, 74, 83, 86, 94, 100, 101, 105, 106, 107, 113, 116, 117,
                120, 122, 125, 127, 130, 131, 136, 140, 142, 143, 144, 147, 150, 155, 159,
                161, 163, 164, 167, 169, 173, 181, 182, 184, 196, 199, 203, 205, 206, 209,
                213, 214, 217, 218, 219, 226, 227, 231, 236, 238, 239, 241, 242, 243, 245,
                246, 249, 250, 251, 252, 253, 255, 258, 259, 265, 266, 270, 271, 273, 280,
                284, 287, 291, 293, 295, 296, 298, 300, 303, 304, 306, 307, 310, 311, 313,
                316, 317, 318, 320, 321, 322, 324, 326, 328, 329, 330, 335, 336, 342, 344,
                350, 351, 354, 356, 357, 358, 359, 360, 361, 366, 368, 369, 370, 372, 378,
                379, 385, 386, 388, 389, 393, 394, 402, 403, 404, 406, 408, 411, 413, 414,
                417, 420, 421, 423, 427, 430, 433, 434, 435, 438, 439, 441, 442, 446, 454,
                455, 456, 462, 464, 469, 473, 476, 477, 478, 483, 485, 486, 488, 489, 493,
                495, 496, 498, 509, 510, 512, 514, 515, 516, 518, 521, 524, 525, 526, 527,
                530, 534, 541, 542, 543, 545, 548, 551, 552, 553, 555, 556, 562, 564, 569,
                572, 573, 581, 582, 584, 585, 586, 587, 590, 592, 593, 594, 596, 597, 600,
                602, 605, 609, 610, 612, 613, 616, 617, 626, 627, 629, 630, 631, 634, 636,
                643, 645, 646, 650, 656, 658, 659, 663, 664, 665, 671, 674, 676, 677, 683,
                684, 686, 690, 696, 698, 700, 703, 712, 713, 716, 722, 723, 724, 725, 727,
                730, 732, 734, 735, 739, 741, 742, 745, 749, 755, 759, 765, 767, 768, 769,
                772, 773, 775, 777, 778, 782, 783, 785, 790, 791, 795, 796, 797, 799, 800,
                804, 806, 807, 808, 809, 816, 818, 821, 822, 823, 825, 826, 828, 833, 834,
                836, 837, 841, 843, 845, 847, 857, 863, 864, 865, 866, 867, 869, 870, 871,
                872, 873, 876, 878, 883, 887, 893, 894, 898, 899, 901, 902, 905, 906, 908,
                916, 919, 920, 921, 922, 923, 927, 928, 931, 932, 934, 940, 941, 945, 946,
                947, 949, 951, 952, 954, 955, 956, 957, 959, 960, 962, 963, 964, 970, 975,
                976, 989, 991, 992, 999, 1000, 1002, 1004, 1006, 1009, 1010, 1011, 1013, 1016,
                1021, 1023, 1026, 1027, 1029, 1030, 1033, 1034, 1047, 1048, 1049, 1050, 1051,
                1056, 1067, 1068, 1069, 1073, 1074, 1077, 1078, 1087, 1095, 1100, 1104, 1112,
                1133, 1136, 1138, 1139, 1140, 1141, 1145, 1147, 1149, 1151, 1153, 1154, 1157,
                1159, 1166, 1167, 1168, 1169, 1170, 1172, 1179, 1180, 1181, 1187, 1188, 1189,
                1190, 1204, 1205, 1206, 1214, 1216, 1219, 1225, 1226, 1228], device=device)

        common = torch.tensor([2, 5, 6, 8, 9, 11, 18, 19, 23, 25, 26, 27, 28, 29, 33, 37, 44,
                  47, 48, 54, 55, 62, 63, 68, 71, 73, 75, 76, 80, 82, 85, 92, 93,
                  97, 98, 102, 103, 108, 109, 111, 114, 115, 119, 121, 123, 128,
                  129, 134, 135, 141, 145, 148, 149, 151, 152, 153, 156, 157, 158,
                  162, 165, 166, 168, 171, 175, 176, 177, 178, 186, 188, 189, 190,
                  192, 193, 195, 200, 201, 202, 204, 207, 210, 215, 216, 220, 222,
                  223, 224, 225, 228, 230, 232, 233, 234, 244, 247, 248, 254, 256,
                  257, 262, 264, 267, 268, 272, 274, 275, 278, 279, 283, 285, 286,
                  290, 292, 294, 297, 305, 309, 312, 314, 315, 319, 325, 331, 332,
                  333, 337, 338, 339, 340, 341, 343, 346, 348, 349, 355, 363, 364,
                  367, 373, 374, 375, 376, 380, 381, 384, 387, 391, 396, 398, 399,
                  400, 401, 405, 409, 412, 415, 419, 424, 425, 426, 431, 432, 440,
                  443, 445, 448, 449, 450, 453, 457, 460, 461, 463, 466, 468, 470,
                  471, 472, 474, 479, 481, 482, 484, 487, 490, 491, 492, 494, 497,
                  499, 500, 501, 503, 505, 507, 511, 513, 519, 520, 522, 523, 528,
                  529, 532, 533, 535, 536, 537, 539, 540, 544, 547, 549, 557, 561,
                  563, 565, 566, 567, 570, 574, 576, 583, 588, 589, 591, 595, 598,
                  599, 604, 607, 608, 611, 614, 618, 620, 622, 623, 633, 635, 640,
                  644, 647, 648, 657, 660, 661, 662, 667, 668, 670, 675, 678, 679,
                  681, 682, 685, 689, 692, 693, 694, 695, 701, 705, 706, 707, 709,
                  719, 731, 733, 737, 738, 740, 743, 744, 748, 750, 751, 752, 753,
                  754, 756, 757, 758, 760, 762, 763, 766, 774, 776, 780, 781, 786,
                  787, 788, 789, 792, 794, 798, 801, 802, 803, 810, 811, 814, 815,
                  820, 832, 835, 838, 839, 844, 848, 849, 854, 855, 856, 858, 860,
                  861, 868, 875, 877, 879, 880, 881, 882, 884, 885, 886, 888, 889,
                  891, 892, 897, 903, 904, 907, 909, 911, 915, 917, 918, 936, 938,
                  939, 942, 943, 944, 948, 950, 953, 958, 967, 968, 971, 972, 978,
                  981, 987, 988, 990, 994, 995, 1003, 1005, 1007, 1008, 1015, 1017,
                  1019, 1020, 1022, 1025, 1031, 1032, 1036, 1041, 1052, 1053, 1055,
                  1058, 1059, 1060, 1061, 1064, 1066, 1071, 1079, 1081, 1082, 1083,
                  1085, 1086, 1088, 1089, 1090, 1093, 1096, 1101, 1102, 1103, 1105,
                  1106, 1107, 1108, 1109, 1110, 1114, 1116, 1120, 1121, 1124, 1125,
                  1126, 1127, 1131, 1134, 1142, 1146, 1148, 1150, 1152, 1156, 1158,
                  1160, 1161, 1163, 1165, 1171, 1173, 1174, 1175, 1176, 1178, 1182,
                  1185, 1186, 1191, 1192, 1193, 1194, 1195, 1197, 1198, 1199, 1202,
                  1203, 1207, 1208, 1209, 1210, 1212, 1217, 1218, 1220, 1221, 1222,
                  1223, 1227, 1230], device=device)
        frequent = torch.tensor([3, 4, 12, 13, 17, 20, 24, 30, 32, 34, 35, 36, 41, 43, 45, 50, 53,
                    56, 57, 58, 59, 60, 61, 66, 67, 69, 77, 78, 79, 81, 84, 87, 88,
                    89, 90, 91, 95, 96, 99, 104, 110, 112, 118, 124, 126, 132, 133,
                    137, 138, 139, 146, 154, 160, 170, 172, 174, 179, 180, 183, 185,
                    187, 191, 194, 197, 198, 208, 211, 212, 221, 229, 235, 237, 240,
                    260, 261, 263, 269, 276, 277, 281, 282, 288, 289, 299, 301, 302,
                    308, 323, 327, 334, 345, 347, 352, 353, 362, 365, 371, 377, 382,
                    383, 390, 392, 395, 397, 407, 410, 416, 418, 422, 428, 429, 436,
                    437, 444, 447, 451, 452, 458, 459, 465, 467, 475, 480, 502, 504,
                    506, 508, 517, 531, 538, 546, 550, 554, 558, 559, 560, 568, 571,
                    575, 577, 578, 579, 580, 601, 603, 606, 615, 619, 621, 624, 625,
                    628, 632, 637, 638, 639, 641, 642, 649, 651, 652, 653, 654, 655,
                    666, 669, 672, 673, 680, 687, 688, 691, 697, 699, 702, 704, 708,
                    710, 711, 714, 715, 717, 718, 720, 721, 726, 728, 729, 736, 746,
                    747, 761, 764, 770, 771, 779, 784, 793, 805, 812, 813, 817, 819,
                    824, 827, 829, 830, 831, 840, 842, 846, 850, 851, 852, 853, 859,
                    862, 874, 890, 895, 896, 900, 910, 912, 913, 914, 924, 925, 926,
                    929, 930, 933, 935, 937, 961, 965, 966, 969, 973, 974, 977, 979,
                    980, 982, 983, 984, 985, 986, 993, 996, 997, 998, 1001, 1012, 1014,
                    1018, 1024, 1028, 1035, 1037, 1038, 1039, 1040, 1042, 1043, 1044,
                    1045, 1046, 1054, 1057, 1062, 1063, 1065, 1070, 1072, 1075, 1076,
                    1080, 1084, 1091, 1092, 1094, 1097, 1098, 1099, 1111, 1113, 1115,
                    1117, 1118, 1119, 1122, 1123, 1128, 1129, 1130, 1132, 1135, 1137,
                    1143, 1144, 1155, 1162, 1164, 1177, 1183, 1184, 1196, 1200, 1201,
                    1211, 1213, 1215, 1224, 1229], device=device)
        return rare, common, frequent
