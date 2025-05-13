import torch
import matplotlib
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
import glob
from typing import List, Optional, Tuple



#  130行   输出图片的路径
#  134行   trt部署出来的框的txt路径
#  135行    bin文件data路径
# 调整颜色映射表，确保颜色值在[0, 1]区间，并且添加更多颜色种类，使可视化效果更丰富
box_colormap = [
    [1, 0, 0],  # 红色，用于car
    [0, 0, 1],  # 蓝色，用于truck
    [0.5, 0.5, 0.5],  # 灰色
    [0, 1, 0],  
    [1, 1, 0],  
    [0, 1, 1],  
    [1, 0, 1],
    [0.5, 1, 0],  
    [0, 0.5, 1], 
    [1, 0, 0.5],      
]


def get_coor_colors(obj_labels):
    """
    根据对象标签获取对应的颜色值，确保颜色值在合法范围且转换为RGB格式
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = np.max(obj_labels)

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    # 对颜色值进行裁剪，确保每个通道都在[0, 1]范围
    label_rgba = np.clip(label_rgba, 0, 1)

    return label_rgba

def sort_by_name(filename):
    return filename


def draw_2d_bbox(x, y, w, h, theta_rad, thickness,color='r'):
    # 计算旋转矩阵
    # theta_rad = np.radians(angel)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                   [np.sin(theta_rad), np.cos(theta_rad)]])
    
    # 未旋转时的角点坐标
    corners = np.array([[-w/2, -h/2],
                        [w/2, -h/2],
                        [w/2, h/2],
                        [-w/2, h/2]])
    
    # 应用旋转
    rotated_corners = np.dot(corners, R.T)  # 注意转置，因为这里是点乘矩阵
    rotated_corners += np.array([x, y])  # 平移到中心点位置

    # 绘制边框
    plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color,linewidth=thickness)
    # 连接首尾形成闭合
    plt.plot([rotated_corners[0, 0], rotated_corners[-1, 0]],
             [rotated_corners[0, 1], rotated_corners[-1, 1]], color=color,linewidth=thickness)


def visualize_lidar(
        fpath=str,
        lidar=None,
        bboxes=None,
        labels=None,
        classes=None,
        xlim: Tuple[float, float] = (-50, 50),
        ylim: Tuple[float, float] = (-50, 50),
        color: Optional[Tuple[int, int, int]] = None,
        radius: float = 15,
        thickness: float = 25,
) -> None:
    """
    可视化激光雷达数据并保存为图片，这里对绘制2D框时的索引取值进行了调整，更符合常见边界框表示
    """
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )
    if bboxes is not None and len(bboxes) > 0:
        # 假设bboxes的格式是合适的，这里可能需要根据实际情况调整坐标提取等操作
        coords = bboxes
        # print('coords',coords)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            # 根据标签获取对应颜色，确保颜色合法且取自调整后的颜色映射表
            color = box_colormap[labels[index]]
            draw_2d_bbox(coords[index, 0], coords[index, 1], coords[index, 3],
                         coords[index, 4], coords[index, 6], thickness,
                         color)

    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def main():
    point_cloud_range = [-54, -54, -1.0, 54, 54, 7.0]
    object_classes = [
        'car', 'truck', 'lockbox', 'ped', 'lock','bridge',"bus", "trailer", "barrier", "motorcycle", "bicycle"
    ]
    out_dir_path = "CenterPoint/data/0513_idar_images"
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # txt_folder = "data/prediction"
    txt_folder = "data/test"
    bin_folder = "../../CenterPoint_deploy/Lidar_AI_Solution/CenterPoint/data/custom_0411_zzg/points/"
    # bin_folder = "CenterPoint/data/test"
    scores_threshold = 0.3
    if not os.path.exists(txt_folder):
        print("result txt_folder is not exists!")
        return
    bin_lists = os.listdir(bin_folder)
    sorted_lists = sorted(bin_lists, key=lambda x: float(os.path.splitext(x)[0]))

    for file in tqdm(sorted_lists):
        (filename, extension) = os.path.splitext(file)
        result_path = os.path.join(txt_folder, filename) + ".txt"
        bin_path = os.path.join(bin_folder, file)
        if not os.path.exists(result_path):
            print(result_path, "file is not exists!")
            continue
        if not os.path.exists(bin_path):
            print(bin_path, "file is not exists!")
            continue

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        result_boxes = np.loadtxt(result_path)
        # print('result_bbox',result_boxes)
        if len(result_boxes) < 1:
            continue
        if len(result_boxes.shape) < 2:
            result_boxes = result_boxes.reshape(1, -1)
        filtered_data = result_boxes[result_boxes[:, -1] > scores_threshold]    
        ref_boxes = np.hstack([filtered_data[:, :6], filtered_data[:, 8:9]]).astype(float)
        ref_labels = filtered_data[:, 9].astype(int)
        ref_scores = filtered_data[:, 10].astype(float)


        # 可视化并保存俯视图以及添加2D框后的图像
        visualize_lidar(
            os.path.join(out_dir_path, f"{filename}.png"),
            lidar=points[:, :2],  # 取前两维作为平面坐标
            bboxes=ref_boxes[:, [0, 1,2, 3, 4, 5,6]],  # 提取合适的坐标用于绘制2D框（这里假设格式合适，可能需调整）
            labels=ref_labels,
            classes=object_classes,
            xlim=[point_cloud_range[d] for d in [0, 3]],
            ylim=[point_cloud_range[d] for d in [1, 4]],
        )


if __name__ == '__main__':
    main()