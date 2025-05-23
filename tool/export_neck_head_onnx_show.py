# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys; sys.path.insert(0, "./CenterPoint")

import os
import pickle
import torch
import onnx
import argparse
from onnxsim import simplify
import numpy as np
from torch import nn

from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.core import box_torch_ops
from det3d.torchie import Config
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer import load_checkpoint

def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default='tool/checkpoint/epoch_20.pth', action='store',
                        type=str, help='checkpoint')
    parser.add_argument('--config', dest='config',
                        default='CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py', action='store',
                        type=str, help='config')
    parser.add_argument("--save-onnx", type=str, default="rpn_centerhead_sim.onnx", help="output onnx")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    return args

class CenterPointVoxelNet_Post(nn.Module):
    def __init__(self, model):
        super(CenterPointVoxelNet_Post, self).__init__()
        self.model = model
        assert( len(self.model.bbox_head.tasks) == 2 )
        print("len(self.model.bbox_head.tasks) = ", len(self.model.bbox_head.tasks) )

    def forward(self, x):
        x = self.model.neck(x)
        x = self.model.bbox_head.shared_conv(x)
        
        pred = [ task(x) for task in self.model.bbox_head.tasks ]

        return pred[0]['reg'], pred[0]['height'], pred[0]['dim'], pred[0]['rot'], pred[0]['vel'], pred[0]['hm'], \
        pred[1]['reg'], pred[1]['height'], pred[1]['dim'], pred[1]['rot'], pred[1]['vel'], pred[1]['hm']
        # pred[2]['reg'], pred[2]['height'], pred[2]['dim'], pred[2]['rot'], pred[2]['vel'], pred[2]['hm'], \
        # pred[3]['reg'], pred[3]['height'], pred[3]['dim'], pred[3]['rot'], pred[3]['vel'], pred[3]['hm'], \
        # pred[4]['reg'], pred[4]['height'], pred[4]['dim'], pred[4]['rot'], pred[4]['vel'], pred[4]['hm'], \
        # pred[5]['reg'], pred[5]['height'], pred[5]['dim'], pred[5]['rot'], pred[5]['vel'], pred[5]['hm']

def predict(reg, hei, dim, rot, vel, hm, test_cfg):
    """decode, nms, then return the detection result.
    """
    # convert N C H W to N H W C
    reg = reg.permute(0, 2, 3, 1).contiguous()
    hei = hei.permute(0, 2, 3, 1).contiguous()
    dim = dim.permute(0, 2, 3, 1).contiguous()
    rot = rot.permute(0, 2, 3, 1).contiguous()
    vel = vel.permute(0, 2, 3, 1).contiguous()
    hm = hm.permute(0, 2, 3, 1).contiguous()

    hm = torch.sigmoid(hm)
    dim = torch.exp(dim)

    rot = torch.atan2(rot[..., 0:1], rot[..., 1:2])

    batch, H, W, num_cls = hm.size()

    reg = reg.reshape(batch, H*W, 2)
    hei = hei.reshape(batch, H*W, 1)

    rot = rot.reshape(batch, H*W, 1)
    dim = dim.reshape(batch, H*W, 3)
    hm = hm.reshape(batch, H*W, num_cls)
    vel = vel.reshape(batch, H*W, 2)

    ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    ys = ys.view(1, H, W).repeat(batch, 1, 1).to(hm)
    xs = xs.view(1, H, W).repeat(batch, 1, 1).to(hm)

    xs = xs.view(batch, -1, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, -1, 1) + reg[:, :, 1:2]

    xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
    ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

    box_preds = torch.cat([xs, ys, hei, dim, vel, rot], dim=2)

    box_preds = box_preds[0]
    hm_preds = hm[0]

    scores, labels = torch.max(hm_preds, dim=-1)

    score_mask = scores > test_cfg.score_threshold

    post_center_range = test_cfg.post_center_limit_range

    if len(post_center_range) > 0:
        post_center_range = torch.tensor(
            post_center_range,
            dtype=hm.dtype,
            device=hm.device,
        )
    distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
        & (box_preds[..., :3] <= post_center_range[3:]).all(1)

    mask = distance_mask & score_mask

    box_preds = box_preds[mask]
    scores = scores[mask]
    labels = labels[mask]

    boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

    selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(),
                        thresh=test_cfg.nms.nms_iou_threshold,
                        pre_maxsize=test_cfg.nms.nms_pre_max_size,
                        post_max_size=test_cfg.nms.nms_post_max_size)

    ret = {}
    ret["box3d_lidar"] = box_preds[selected]
    ret["scores"] = scores[selected]
    ret["label_preds"] = labels[selected]

    return ret

def main(args):
    cfg = Config.fromfile(args.config)

    # Get 1 frame data as model.reader input
    example = dict()
    test_pkl = 'data/pkl/nusc_test_in.pkl'
    # test_pkl = 'data/custom_0411_zzg/infos_val_01sweeps_withvelo_filter_True.pkl'
    gt_pkl = 'data/pkl/nusc_gt_out_fp32.pkl'
    if args.half:
        gt_pkl = 'data/pkl/nusc_gt_out_fp16.pkl'

    if os.path.isfile(test_pkl):
        with open(test_pkl, 'rb') as handle:
            example = pickle.load(handle)
    else:
        dataset = build_dataset(cfg.data.val)

        data_loader = build_dataloader(
            dataset,
            batch_size=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False,
        )
        data_iter = iter(data_loader)
        data_batch = next(data_iter)
        example = example_to_device(data_batch, torch.device("cuda"), non_blocking=False)
        with open(test_pkl, 'wb') as handle:
            pickle.dump(example, handle)
    # print("++++++++: ", len(example))
    assert(len(example.keys()) > 0)
    # print("Token: ", example[0].keys())
    assert(len(example["points"]) == 1)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    post_model = CenterPointVoxelNet_Post(model)

    if args.half:
        model.eval().half()
        post_model.eval().half()
    else:
        model.eval()
        post_model.eval()

    model = model.cuda()
    post_model = post_model.cuda()

    with torch.no_grad():
        example = dict(
            features=example['voxels'],
            num_voxels=example["num_points"],
            coors=example["coordinates"],
            batch_size=len(example['points']),
            input_shape=example["shape"][0],
        )

        input_features = model.reader(example["features"], example['num_voxels'])
        input_indices = example["coors"]

        if args.half:
            input_features = input_features.half()

        x, _ = model.backbone(
            input_features, input_indices, example["batch_size"], example["input_shape"]
            )

        rpn_input  = torch.zeros(x.shape,dtype=torch.float32,device=torch.device("cuda"))
        if args.half:
            rpn_input  = rpn_input.half()

        torch.onnx.export(post_model, rpn_input, "tmp.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            keep_initializers_as_inputs=False, input_names = ['input'],
            output_names = ['reg_0', 'height_0', 'dim_0', 'rot_0', 'vel_0', 'hm_0',
                            'reg_1', 'height_1', 'dim_1', 'rot_1', 'vel_1', 'hm_1'],
            # output_names = ['reg_0', 'height_0', 'dim_0', 'rot_0', 'vel_0', 'hm_0',
            #                 'reg_1', 'height_1', 'dim_1', 'rot_1', 'vel_1', 'hm_1',
            #                 'reg_2', 'height_2', 'dim_2', 'rot_2', 'vel_2', 'hm_2',
            #                 'reg_3', 'height_3', 'dim_3', 'rot_3', 'vel_3', 'hm_3',
            #                 'reg_4', 'height_4', 'dim_4', 'rot_4', 'vel_4', 'hm_4',
            #                 'reg_5', 'height_5', 'dim_5', 'rot_5', 'vel_5', 'hm_5'],                            
            )

        sim_model, check = simplify_model("tmp.onnx")
        if not check:
            print("[ERROR]:Simplify %s error!"% "tmp.onnx")
        onnx.save(sim_model, args.save_onnx)
        print("[PASS] Export ONNX done.")

        if args.export_only:
            return

        # 添加ONNX推理部分
        print("\nStarting ONNX Inference...")
        
        # 加载ONNX模型
        ort_session = ort.InferenceSession(args.save_onnx)
        
        # 准备输入数据（使用导出时的rpn_input）
        input_name = ort_session.get_inputs()[0].name
        input_data = rpn_input.cpu().numpy()
        if args.half:
            input_data = input_data.astype(np.float16)
        else:
            input_data = input_data.astype(np.float32)
        
        # 运行推理
        ort_outputs = ort_session.run(None, {input_name: input_data})
        
        # 将输出转换为PyTorch张量并移至GPU
        ort_outputs = [torch.from_numpy(out).cuda() for out in ort_outputs]
        
        # 解析各任务头输出（根据实际任务数量调整）
        num_tasks = 2
        task_outputs = []
        for task_idx in range(num_tasks):
            start_idx = task_idx * 6
            task_data = {
                'reg': ort_outputs[start_idx],
                'height': ort_outputs[start_idx+1],
                'dim': ort_outputs[start_idx+2],
                'rot': ort_outputs[start_idx+3],
                'vel': ort_outputs[start_idx+4],
                'hm': ort_outputs[start_idx+5],
            }
            task_outputs.append(task_data)
        
        # 执行后处理
        box3d_lidar_onnx = torch.tensor([]).cuda()
        scores_onnx = torch.tensor([]).cuda()
        label_preds_onnx = torch.tensor([]).cuda()
        cls_nums = [0, 1]  # 根据实际类别调整
        
        for task_idx in range(len(cls_nums)):
            pred = task_outputs[task_idx]
            output = predict(
                pred['reg'], pred['height'], pred['dim'],
                pred['rot'], pred['vel'], pred['hm'],
                cfg.test_cfg
            )
            
            box3d_lidar_onnx = torch.cat([box3d_lidar_onnx, output['box3d_lidar']])
            scores_onnx = torch.cat([scores_onnx, output['scores']])
            label_preds_onnx = torch.cat([label_preds_onnx, output['label_preds'] + cls_nums[task_idx]])
        


if __name__ == "__main__":
    args = arg_parser()
    main(args)