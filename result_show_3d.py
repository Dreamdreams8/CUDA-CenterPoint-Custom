import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import argparse

class Visualizer3D:
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="3D Detection Viewer", width=1600, height=900)
        
        # 在__init__方法中添加点云渲染参数
        self.pcd = o3d.geometry.PointCloud()
        # 初始化颜色模式
        self.color_mode = 'fixed'
        self.pcd.paint_uniform_color([0.0, 0.0, 0.0])  # 默认固定颜色
        self.vis.add_geometry(self.pcd)
        render_option = self.vis.get_render_option()
        render_option.point_size = 1.5  # 调整点云大小（原默认值3.0）
        # 在__init__方法中添加线宽参数
        render_option.line_width = 10.0  # 设置边界框线宽
        
        # 初始化边界框集合
        self.line_set = None
        
        # 注册按键回调
        self.vis.register_key_callback(32, self.space_callback)  # 空格键
        self.vis.register_key_callback(ord('Q'), self.quit_callback)
        self.vis.register_key_callback(ord('R'), self.reset_view_callback)
        
        self.is_run = True
        self.next_frame = False

    def reset_view(self):
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([-0.5, -0.5, -0.5])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])

    def reset_view_callback(self, vis):
        self.reset_view()
        return False

    def space_callback(self, vis):
        self.next_frame = True
        return False

    def quit_callback(self, vis):
        self.is_run = False
        return False

    def update_scene(self, points, boxes, labels, scores):
        # 更新点云颜色
        if self.color_mode == 'intensity' and points.shape[1] >=4:
            intensities = points[:, 3]
            colors = np.zeros((points.shape[0], 3))
            colors[:, 0] = intensities  # 使用强度值作为红色通道
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 更新边界框
        if self.line_set is not None:
            self.vis.remove_geometry(self.line_set)
        
        self.line_set = self.create_line_set(boxes, scores, labels)
        if self.line_set is not None:
            self.vis.add_geometry(self.line_set)
        
        self.vis.poll_events()
        self.vis.update_renderer()

    def create_line_set(self, boxes, scores, labels, score_thr=0.3):
        lines = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        
        # 颜色映射规则
        colors = [
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
        
        all_corners = []
        all_lines = []
        all_colors = []
        
        for i, box in enumerate(boxes):
            if scores[i] < score_thr:
                continue
            
            center = box[:3]
            dims = box[3:6]
            yaw = box[6]
            
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])
            
            corners = np.array([
                [-dims[0]/2, -dims[1]/2, -dims[2]/2],
                [dims[0]/2, -dims[1]/2, -dims[2]/2],
                [dims[0]/2, dims[1]/2, -dims[2]/2],
                [-dims[0]/2, dims[1]/2, -dims[2]/2],
                [-dims[0]/2, -dims[1]/2, dims[2]/2],
                [dims[0]/2, -dims[1]/2, dims[2]/2],
                [dims[0]/2, dims[1]/2, dims[2]/2],
                [-dims[0]/2, dims[1]/2, dims[2]/2]])
            
            corners = corners @ rot_mat.T + center
            all_corners.extend(corners)
            all_lines.extend(np.array(lines) + len(all_corners) - 8)
            all_colors.extend([colors[labels[i]%len(colors)]]*12)
        
        if len(all_corners) == 0:
            return None
            
        # 在create_line_set方法末尾添加线宽设置
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_corners)
        line_set.lines = o3d.utility.Vector2iVector(all_lines)
        line_set.colors = o3d.utility.Vector3dVector(all_colors)
        # 移除统一颜色设置
        return line_set

def main():
    parser = argparse.ArgumentParser(description='3D Detection Visualizer')
    parser.add_argument('--data', type=str, default='data/test', help='点云数据目录')
    parser.add_argument('--results', type=str, default='results', help='检测结果目录')
    parser.add_argument('--score_thr', type=float, default=0.3, help='分数阈值')
    parser.add_argument('--color_mode', type=str, default='fixed', choices=['fixed', 'intensity'],
                       help='点云着色模式：fixed-固定颜色, intensity-强度着色')
    args = parser.parse_args()

    vis = Visualizer3D()
    vis.color_mode = args.color_mode  # 应用颜色模式参数
    bin_files = sorted([f for f in os.listdir(args.data) if f.endswith('.bin')])
    
    current_frame = 0
    while current_frame < len(bin_files) and vis.is_run:
        bin_path = os.path.join(args.data, bin_files[current_frame])
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        
        # 加载检测结果
        txt_path = os.path.join(args.results, f"{os.path.splitext(bin_files[current_frame])[0]}.txt")
        if os.path.exists(txt_path):
            boxes_data = np.loadtxt(txt_path)
            if boxes_data.ndim == 1:
                boxes_data = boxes_data.reshape(1, -1)
            boxes = boxes_data[:, :7]
            scores = boxes_data[:, -1]
            labels = boxes_data[:, -2].astype(int)
        else:
            boxes = np.zeros((0, 7))
            scores = np.zeros(0)
            labels = np.zeros(0)
        
        vis.update_scene(points, boxes, labels, scores)
        
        # 控制帧率
        while not vis.next_frame and vis.is_run:
            vis.vis.poll_events()
            vis.vis.update_renderer()
        
        vis.next_frame = False
        current_frame += 1

    vis.vis.destroy_window()

if __name__ == '__main__':
    main()
# # 固定颜色模式（默认）
# python result_show_3d.py --data ... --results ...

# # 强度着色模式
# python result_show_3d.py --color_mode intensity --data ... --results ...