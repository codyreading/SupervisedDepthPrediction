#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:33
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
"""
import os
import numpy as np
import cv2

from dp.visualizers.utils import depth_to_color, error_to_color
from dp.visualizers.base_visualizer import BaseVisualizer
from dp.utils.pyt_ops import tensor2numpy, interpolate


class dorn_visualizer(BaseVisualizer):
    def __init__(self, config, writer=None):
        super(dorn_visualizer, self).__init__(config, writer)

    def visualize(self, batch, out, epoch=0):
        """
            :param batch_in: minibatch
            :param pred_out: model output for visualization, dic, {"target": [NxHxW]}
            :param tensorboard: if tensorboard = True, the visualized image should be in [0, 1].
            :return: vis_ims: image for visualization.
            """
        fn = batch["fn"]
        if batch["target"].shape != out["target"][-1].shape:
            h, w = batch["target"].shape[-2:]
            # batch = interpolate(batch, size=(h, w), mode='nearest')
            out = interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        image = batch["image_n"].numpy()

        has_gt = False
        if batch.get("target") is not None:
            depth_gts = tensor2numpy(batch["target"])
            has_gt = True

        for i in range(len(fn)):
            image = image[i].astype(np.float)
            depth = np.zeros((out['target'][0].shape[0], batch["H"], batch["W"]), dtype=np.float32)
            depth[:] = np.nan
            out_depth = tensor2numpy(out['target'][0])

            for j in range(depth.shape[0]):
                x0 = batch["x0"][j]
                x1 = batch["x1"][j]
                y0 = batch["y0"][j]
                y1 = batch["y1"][j]
                target_patch = depth[j, y0:y1, x0:x1]
                patch = cv2.resize(out_depth[j], (target_patch.shape[1],
                                                  target_patch.shape[0]), interpolation=cv2.INTER_LINEAR)

                depth[j, y0:y1, x0:x1] = patch
            depth = np.nanmean(depth, axis=0)
            depth = cv2.resize(depth, (batch["W_img"], batch["H_img"]), interpolation=cv2.INTER_LINEAR)
            output_dir, sample_idx = os.path.split(batch["target_path"][i])
            output_dir = os.path.dirname(output_dir)
            output_file = os.path.join(output_dir, "depth_pred", sample_idx)
            depth_image = (depth * 255.0).astype(np.uint16)
            cv2.imwrite(output_file, depth_image)
            # print("!! depth shape:", depth.shape)

            # if has_gt:
            #     depth_gt = depth_gts[i]

            #     err = error_to_color(depth, depth_gt)
            #     depth_gt = depth_to_color(depth_gt)

            # depth = depth_to_color(depth)
            # # print("pred:", depth.shape, " target:", depth_gt.shape)
            # group = np.concatenate((image, depth), axis=0)

            # if has_gt:
            #     gt_group = np.concatenate((depth_gt, err), axis=0)
            #     group = np.concatenate((group, gt_group), axis=1)

            # if self.writer is not None:
            #     group = group.transpose((2, 0, 1)) / 255.0
            #     group = group.astype(np.float32)
            #     # print("group shape:", group.shape)
            #     self.writer.add_image(fn[i] + "/image", group, epoch)
