import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

output_name = 'my_label/0000335_04117_d_0000064_auto_result.json'
coco = COCO('my_label/0000335_04117_d_000064_gt.json')

# with open(output_name, 'w') as f:
#     json.dump(new_json_data, f)

if len(coco.dataset['annotations']) > 0:
    cocoDt = COCO(output_name)

    metrics = ['bbox']

    for metric in metrics:
        coco_eval = COCOeval(coco, cocoDt, iouType=metric)
        # coco_eval.params.catIds = [2]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        CLASSES = ('person', 'bicycle', 'car', 'truck', 'bus', 'motorcycle')

        precisions = coco_eval.eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        cat_ids = cocoDt.getCatIds(catNms=CLASSES)
        print(len(cat_ids))
        print(precisions.shape)
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = cocoDt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        print(results_per_category)
