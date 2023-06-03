import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_path = 'data/test/u.jpg'
detector = init_detector(
    'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
    'checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    device=device
)
pose_estimator = init_pose_estimator(
    'configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py',
    'checkpoint/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
)

init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
detect_result = inference_detector(detector, img_path)
print(detect_result)

CONF_THRES = 0.5

pred_instance = detect_result.pred_instances.cpu().numpy()
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

pose_results = inference_topdown(pose_estimator, img_path, bboxes)
data_samples = merge_data_samples(pose_results)

# 半径
pose_estimator.cfg.visualizer.radius = 10
# 线宽
pose_estimator.cfg.visualizer.line_width = 8
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# 元数据
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

img_output = visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show_kpt_idx=True,
            show=False,
            wait_time=0,
            out_file='outputs/B3.jpg'
)

plt.figure(figsize=(10,10))
plt.imshow(img_output)
