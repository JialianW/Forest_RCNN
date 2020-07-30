from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn

from mmdet.core import auto_fp16, get_classes, tensor2imgs
import json
import cv2
import os




# def show_result(data, result, dataset=None, score_thr=0.3):
def show_result(dataset=None, score_thr=0.3):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)
    if dataset is None:
        class_names = self.CLASSES
    elif isinstance(dataset, str):
        class_names = get_classes(dataset)
    elif isinstance(dataset, (list, tuple)):
        class_names = dataset
    else:
        raise TypeError(
            'dataset must be a valid dataset name or a sequence'
            ' of class names, not {}'.format(type(dataset)))
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        bboxes = np.vstack(bbox_result)
        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        mmcv.imshow_det_bboxes(
            img_show,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr)
def get_cat():
    rare = [1, 7, 10, 14, 15, 16, 21, 22, 31, 38, 39, 40, 42, 46, 49, 51, 52,
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
                         1190, 1204, 1205, 1206, 1214, 1216, 1219, 1225, 1226, 1228]

    common = [2, 5, 6, 8, 9, 11, 18, 19, 23, 25, 26, 27, 28, 29, 33, 37, 44,
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
                           1223, 1227, 1230]
    return common,rare

if __name__=='__main__':
    score_thr = 0.4
    # entry = 'nms050709_thres03'
    entry = 'nms070809_thres03'
    # entry = 'mask_rcnn_r50_fpn_1x'
    dets = json.load(open('work_dirs/{}/temp_0.segm.json'.format(entry)))
    imgs = os.listdir('data/lvis/val2017/')
    anns = json.load(open('data/lvis/annotations/lvis_v0.5_val.json'))['annotations']
    for idx, img in enumerate(imgs):
        if idx > 1000:
            break
        image_id = int(img.split('.')[0])

        show = False
        for ann in anns:
            if image_id == ann['image_id']:
                cat = ann['category_id']
                common, rare = get_cat()
                if (cat in rare):
                    show = True
                    break
        if show == False:
            continue



        im_name = '%012d'% image_id+'.jpg'
        img_show = cv2.imread('data/lvis/val2017/'+im_name)
        i=0
        boxes = np.zeros((1000,5))
        boxes[:,-1] = -1
        labels=[]
        for det in dets:
            if det['image_id'] == image_id:
                boxes[i, :4] = [det['bbox'][0],det['bbox'][1],det['bbox'][0]+det['bbox'][2],det['bbox'][1]+det['bbox'][3]]
                boxes[i, -1] = det['score']
                i+=1
                labels.append(det['category_id']-1)
                if det['score'] > score_thr:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(det['segmentation']).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5


        bboxes=boxes[:i, :]
        labels = np.array(labels)
        class_names = get_classes('lvis')

        if not os.path.exists('vis/{}/'.format(entry)):
            os.mkdir('vis/{}/'.format(entry))
        mmcv.imshow_det_bboxes(
            img_show,
            bboxes,
            labels,
            show=False,
            class_names=class_names,
            score_thr=score_thr,
            out_file='vis/{}/'.format(entry)+im_name,
            show_class=True,
            show_box=True)