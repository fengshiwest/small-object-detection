import argparse
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import warnings
import cv2
import mmcv
import torch
import math
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import numpy as np
from mmdet.core import UnifiedForegroundPacking
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
import json

from mmdet.core import bbox2result
from mmdet.core.visualization import imshow_det_bboxes
import torchvision

CLASSES = ['car', 'truck', 'bus']

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 128, 255),
    (128, 255, 128),
    (255, 128, 128),
    (255, 255, 255),

]


def my_show_result_pyplot(model,
                          img,
                          result,
                          img_name,
                          score_thr=0.3,
                          title='result',
                          wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module

    img_dir = './data/VisDrone/visualize/UFP_dataset_mp_gfocal'
    save_file = os.path.join(img_dir, img_name)

    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        out_file=save_file,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))


def show_result(img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=True,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    if not (show or out_file):
        return img


def compute_iof(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / min(area1, area2)


def compute_iou(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / (area1 + area2 - inter)


# modify test
class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results, bbox=None, img_data=None):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None

        if img_data is None:
            img = mmcv.imread(results['img'])
        else:
            img = img_data
        if bbox:
            x1, x2, y1, y2, _ = bbox
            img = img[x1:x2, y1:y2, :]
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def my_inference_detector(model, data):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    # else:
    #     # Use torchvision ops for CPU mode instead
    #     for m in model.modules():
    #         if isinstance(m, (RoIPool, RoIAlign)):
    #             if not m.aligned:
    #                 # aligned=False is not implemented on CPU
    #                 # set use_torchvision on-the-fly
    #                 m.use_torchvision = True
    #     warnings.warn('We set use_torchvision=True in CPU mode.')
    #     # just get the actual data from DataContainer
    #     data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result[0]


def my_inference_detector_mp(model, data):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    # else:
    #     # Use torchvision ops for CPU mode instead
    #     for m in model.modules():
    #         if isinstance(m, (RoIPool, RoIAlign)):
    #             if not m.aligned:
    #                 # aligned=False is not implemented on CPU
    #                 # set use_torchvision on-the-fly
    #                 m.use_torchvision = True
    #     warnings.warn('We set use_torchvision=True in CPU mode.')
    #     # just get the actual data from DataContainer
    #     data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, 10)
            for det_bboxes, det_labels in result[0]
        ]

    return bbox_results[0]


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
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

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def display_merge_result(results, img, img_name, w, h):
    w = math.ceil(w)
    h = math.ceil(h)
    img_data = cv2.imread(img)
    new_img = np.zeros((h, w, 3))
    for result in results:
        x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in result]
        if w == 0 or h == 0:
            continue
        new_img[n_y:n_y + h * scale_factor, n_x:n_x + w * scale_factor, :] = cv2.resize(
            img_data[y1:y1 + h, x1:x1 + w, :], (w * scale_factor, h * scale_factor))
    return new_img


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('coarse_detector_config', help='')
    parser.add_argument('coarse_detector_config_ckpt', help='')
    parser.add_argument('mp_det_config', help='')
    parser.add_argument('mp_det_config_ckpt', help='')
    parser.add_argument('dataset_anno', help='')
    parser.add_argument('dataset_root', help='')
    # parser.add_argument('dataset_ufp_root', help='')
    args = parser.parse_args()
    return args


def main():
    device = 'cuda'
    args = parse_args()
    coarse_detecor_config = args.coarse_detector_config
    coarse_detecor_config_ckpt = args.coarse_detector_config_ckpt
    mp_det_config = args.mp_det_config
    mp_det_config_ckpt = args.mp_det_config_ckpt

    coarse_detecor = init_detector(coarse_detecor_config, coarse_detecor_config_ckpt, device=device)
    mp_det = init_detector(mp_det_config, mp_det_config_ckpt, device=device)
    dataset_anno = args.dataset_anno
    dataset_root = args.dataset_root
    # dataset_ufp_root = args.dataset_ufp_root

    # with open(dataset_anno) as f:
    #     json_info = json.load(f)
    # annotation_set = {}
    # for annotation in json_info['annotations']:
    #     image_id = annotation['image_id']
    #     if not image_id in annotation_set.keys():
    #         annotation_set[image_id] = []
    #     annotation_set[image_id].append(annotation)

    coco = COCO(dataset_anno)  # 导入验证集
    size = len(list(coco.imgs.keys()))
    results = []
    times = []
    # shape_set = set()
    cnt = 1
    rm_cnt = 1e-6
    tp_cnt = 1e-6
    sum_rm = 0
    sum_tp = 0
    for key in range(size):
        print(cnt, '/', size)
        cnt += 1
        # if cnt > 10:
        #     continue
        image_id = key
        width = coco.imgs[key]['width']
        height = coco.imgs[key]['height']
        img_name = coco.imgs[key]['file_name']
        img = os.path.join(dataset_root, img_name)
        # img_ufp = os.path.join(dataset_ufp_root, img_name)
        data = dict(img=img)
        # data_ufp = dict(img=img_ufp)

        # cur_annotation = annotation_set[key]
        first_results = my_inference_detector(coarse_detecor, LoadImage()(data))
        # show_result_pyplot(coarse_detecor, img, first_results, score_thr=0.1)

        result = np.concatenate(first_results)
        rec, w, h = UnifiedForegroundPacking(result[:, :4], 1.5, input_shape=[width, height])

        # dens = np.load(img.replace("images", "dens_MCNN").replace("jpg", "npy"))

        # dens_norm = (dens - dens.min()) / (dens.max() - dens.min())
        # scale_param = 1.5 * np.exp(dens_norm)
        # scale_param = 1.5 * (dens_norm +1)
        # scale_param = 1.5 * np.power(dens_norm + 1, 2)

        # scale_param = 1.5 * (np.power(dens_norm, 2) + 1)

        # rec, w, h = UnifiedForegroundPacking(result[:, :4], scale_param, input_shape=[width, height])


        next_image = display_merge_result(rec, img, img_name, w, h)

        # 展示裁剪框
        # import matplotlib.pyplot as plt
        # plt.imshow(next_image)
        # plt.show()

        # ignore_list = gen_ignore_list(img_name)
        # time2 = time.time()

        # 推理切图
        second_results = my_inference_detector(mp_det, LoadImage()(data, img_data=next_image))
        # second_results = my_inference_detector_mp(mp_det, LoadImage()(data, img_data=next_image))

        # 切好的图
        # second_results = my_inference_detector_mp(mp_det, LoadImage()(data_ufp))

        # show_result_pyplot(mp_det, next_image, second_results, score_thr=0.1)

        # for save image
        # my_show_result_pyplot(mp_det, next_image, second_results, img_name, score_thr=0.3)
        # my_show_result_pyplot(mp_det, img, second_results, img_name, score_thr=0.3)

        # time3 = time.time()
        # times.append(time3-time2)

        final_results = []
        for first_result in first_results:
            final_results.append(first_result)

        new_second_result = []
        for i in range(3):
            new_second_result.append([])

        for chips in rec:
            o_x1, o_y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in chips]
            chip_bbox = [n_x, n_y, n_x + w * scale_factor, n_y + h * scale_factor]
            for idx, _results in enumerate(second_results):
                for _result in _results:
                    x1, y1, x2, y2, score = _result
                    t_bbox = [x1, y1, x2, y2]
                    if compute_iof(t_bbox, chip_bbox) > 0.9:
                        new_w = (x2 - x1) / scale_factor
                        new_h = (y2 - y1) / scale_factor
                        new_x = (x1 - n_x) / scale_factor + o_x1
                        new_y = (y1 - n_y) / scale_factor + o_y1
                        new_bbox = [new_x, new_y, new_x + new_w, new_y + new_h, score]
                        new_second_result[idx].append(new_bbox)

        for idx in range(len(new_second_result)):
            new_second_result[idx] = np.array(new_second_result[idx])

        assert len(final_results) == len(new_second_result), "len(final_results) !== len(new_second_result)"

        for idx in range(len(final_results)):
            if len(final_results[idx]) == 0:
                final_results[idx] = new_second_result[idx]
            elif len(new_second_result[idx]) == 0:
                final_results[idx] = final_results[idx]
            else:
                final_results[idx] = np.append(final_results[idx], new_second_result[idx], axis=0)

        # show_result(img, final_results, score_thr=0.1)

        for idx, result in enumerate(final_results):
            # result = np.array(result)
            if result.shape[0] == 0:
                continue
            # keep = py_cpu_nms(result, 0.6)
            keep = torchvision.ops.nms(torch.tensor(result[:, 0:4]), torch.tensor(result[:, 4]), 0.6)
            keep = keep.numpy().tolist()

            for bbox in result[keep]:
                # for bbox in result:
                rm_cnt += 1
                x1, y1, x2, y2, score = bbox
                # sum_rm += score
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                pred_bbox = [x1, y1, x2, y2]
                image_result = {
                    'image_id': image_id,
                    'category_id': idx,
                    'score': float(score),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                }
                results.append(image_result)
        # append detection to results
    # write output
    # print(sum(times)/len(times))

    dataset_name = 'UAVDT'
    file_name = 'uavdt_fine_gfocal_r50_fpn_1x_ciou_pretrained_ignore_lr0002_tta'

    json.dump(results, open('./result/{}_bbox_result_tmp_{}.json'.format(dataset_name, file_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = coco
    coco_pred = coco_true.loadRes('./result/{}_bbox_result_tmp_{}.json'.format(dataset_name, file_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    # coco_eval.params.imgIds = image_ids
    coco_eval.params.maxDets = [10, 100, 500]
    coco_eval.evaluate()

    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
