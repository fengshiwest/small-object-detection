import json
import math
import os
import numpy as np

data_root = 'data/VisDrone/'
ann_file = data_root + "annotations/instances_val2017.json"
jd = json.load(open(ann_file))

our_anno_file = data_root + "annotations/instances_val2017_gfocal_r50_fpn_1x_ciou_pretrained.json"
our_jd = json.load(open(our_anno_file))

def distribution_main(jd, num_part, mode="instance", func=np.mean, times=4):
    def get_instance_scales(jd):
        bbox_list = []
        for ann in jd['annotations']:
            bbox_list.append(ann['bbox'])

        bboxes = np.array(bbox_list)
        scales = np.sqrt(bboxes[:, 2] * bboxes[:, 3])
        scales.sort()
        return scales

    def get_image_scales(jd, num_part, func):
        image_scales_list = [[] for i in jd['images']]
        iid_ind, cnt = {}, 0
        for ann in jd['annotations']:
            iid = ann['image_id']
            if iid not in iid_ind:
                iid_ind[iid] = cnt
                cnt += 1
            area = np.sqrt(ann['bbox'][2] * ann['bbox'][3])
            image_scales_list[iid_ind[iid]].append(area)
        scales = np.array([func(np.array(scas)) for scas in image_scales_list if scas])
        scales.sort()
        return scales

    def separate_vals(scales: np.array, num_part: int) -> (np.array, np.array):
        num_bbox = scales.shape[0]
        # intervals = round(num_bbox / num_part)
        # separate_inds = [i for i in range(0, num_bbox, intervals)]
        # if scales[separate_inds[-1]] != scales[-1]:
        #     separate_inds.append(-1)
        # return separate_inds, scales[separate_inds]

        # max_length = math.ceil(scales[num_bbox - 1])
        max_length = 150
        dict = {}
        for i in range(max_length):
            if i % 10 == 0:
                dict[i] = 0

        for i in range(num_bbox):
            index = math.floor(scales[i] / 10)
            if index < len(dict.keys()):
                dict[index * 10] += 1
            # else:
            #     dict[(len(dict.keys()) - 1) * 10] += 1

        key_list = [i + 10 for i in dict.keys()]
        value_list = [round(i/num_bbox * 100) for i in dict.values()]

        return key_list, value_list

    assert mode in ["instance", "image"]
    scales = get_image_scales(jd, num_part, func) if mode == "image" else get_instance_scales(jd)
    # separate_inds, separate_scale = separate_vals(scales, num_part)
    # return scales, separate_scale
    return separate_vals(scales, num_part)


# scales, sca = distribution_main(jd, num_part=4, mode="image", func=np.max)
# scales, sca = distribution_main(jd, num_part=4)
separate_key, separate_value = distribution_main(jd, num_part=4)
our_separate_key, our_separate_value = distribution_main(our_jd, num_part=4)

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


# def show_distribution(bbox_scales, separate_scale, times=4):
#     plt.figure(dpi=200)
#     plt.hist(bbox_scales*times, bins=50,alpha = 0.6, label='scales')
#     parts = len(separate_scale)-1
#     for i in range(0, parts+1):
#         plt.axvline(x=separate_scale[i]*times, color='orange',)
#
#     plt.savefig(f"visdrone_scale_max_{parts}.png", dpi=200, bbox_inches='tight')

# show_distribution(scales, sca)


def show_distribution(separate_key, separate_value, our_separate_key, our_separate_value):
    arr_separate_key = np.array(separate_key)
    arr_separate_value = np.array(separate_value)
    arr_our_separate_key = np.array(our_separate_key)
    arr_our_separate_value = np.array(our_separate_value)

    plt.figure(figsize=(7, 5))

    plt.plot(arr_separate_key, arr_separate_value, marker='.', markersize=12, alpha=0.5, linewidth=4)
    plt.plot(arr_our_separate_key, arr_our_separate_value, marker='.', markersize=12, alpha=0.5, linewidth=4)

    plt.bar(arr_separate_key, arr_separate_value, 8, alpha=0.5, label='scales')
    plt.bar(arr_our_separate_key, arr_our_separate_value, 8, alpha=0.5, label='scales')

    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)

    plt.xlabel("Instances Scale(pixel$^2$)", fontsize=16, fontweight='bold', fontproperties='Times New Roman')
    plt.ylabel("Percentage(%)", fontsize=16, fontweight='bold', fontproperties='Times New Roman')

    plt.legend(['w/o Scale Adaptation', 'w/ Scale Adaptation'], fontsize=18)

    plt.savefig(f"visdrone_scale.png", dpi=300, bbox_inches='tight', transparent=True)

    plt.show()


show_distribution(separate_key, separate_value, our_separate_key, our_separate_value)
