import os, sys
import numpy as np
from datetime import datetime
import trimesh
import argparse
import torch
import json
import time
import subprocess
from joblib import Parallel, delayed, parallel_backend
import yaml
import scipy
from scipy.spatial import cKDTree

sys.path.append(os.getcwd())
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
import multiprocessing as mp
dist_chamfer=ChamferDistance()

from ssr.ssr_utils.network_utils import fix_random_seed

OCCNET_FSCORE_EPS = 1e-09

def percent_below(dists, thresh):
  return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0

def f_score(a_to_b, b_to_a, thresh):
  precision = percent_below(a_to_b, thresh)
  recall = percent_below(b_to_a, thresh)

  return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)
def pointcloud_neighbor_distances_indices(source_points, target_points):
  target_kdtree = scipy.spatial.cKDTree(target_points)
  distances, indices = target_kdtree.query(source_points, workers=-1)
  return distances, indices
def fscore(points1,points2,tau=0.002):
  """Computes the F-Score at tau between two meshes."""
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  f_score_tau = f_score(dist12, dist21, tau)
  return f_score_tau



class Evaluate:
    def __init__(self, config, render=None):
        self.cfg = config
        self.logfile = os.path.join(self.cfg['result_dir'], self.cfg['log'])
        self.cd_loss_dict = {}
        self.f_score_dict = {}
        self.psnr_dict = {}
        self.IoU_dict = {}
        self.debug = self.cfg['debug']
        self.classnames = self.cfg['class_name']
        self.cmd_app = './external/ldif/gaps/bin/x86_64/mshalign'
        self.loginfo = []
        self.get_files()

        self.render = render

    def show(self, data):
        if self.render is not None:
            self.render.show(data)
            self.render.clear()

    def get_files(self):
        self.split = []
        if isinstance(self.classnames, list):
            raise ValueError('not support list, must be all_subset!!!')

        else:
            self.split_path = os.path.join(self.cfg['split_path'], self.classnames + ".json")
            with open(self.split_path, 'rb') as f:
                split = json.load(f)
            if 'FRONT3D' in self.split_path:
                for idx in range(len(split)):
                    if idx > 2000 - 1:                  # only test 2000 items like InstPIFu
                        break
                    self.split.append(split[idx])
            else:
                self.split = split
        print('load {} test items'.format(len(self.split)))

    # 计算 psnr
    def calculate_psnr(self, pred_points, gt_points):
        """Calculate the PSNR between predicted and ground truth point clouds."""
        # 使用 KD-Tree 查找每个预测点在真实点云中的最近邻
        gt_kdtree = cKDTree(gt_points)
        distances, _ = gt_kdtree.query(pred_points)
        
        # 计算最近邻距离的 MSE
        mse = np.mean(distances ** 2)
        if mse == 0:
            return float('inf')  # If MSE is zero, PSNR is infinity

        # 使用点云的最大值来计算 PSNR
        max_value = max(pred_points.max(), gt_points.max())
        psnr = 10 * np.log10(max_value ** 2 / mse)
        return psnr
    
    # 计算 iou
    def calculate_iou(self, pred_mesh, gt_mesh, voxel_resolution=64):
        """Calculate the IoU between predicted and ground truth meshes using voxel grids."""
        # 获取两个网格的联合边界框，并调整网格到相同边界框内
        combined_bounds = np.array([pred_mesh.bounds, gt_mesh.bounds])
        min_bound = combined_bounds[:, 0].min(axis=0)
        max_bound = combined_bounds[:, 1].max(axis=0)

        # 设置稍微扩大的边界框，避免边界上的体素被遗漏
        buffer = 0.05 * (max_bound - min_bound)
        min_bound -= buffer
        max_bound += buffer
        
        # 手动设置一个固定边界框的大小
        bbox_size = max_bound - min_bound

        # 将两个网格都缩放到相同的边界框内
        pred_mesh.apply_translation(-min_bound)
        pred_mesh.apply_scale(1.0 / bbox_size.max())
        gt_mesh.apply_translation(-min_bound)
        gt_mesh.apply_scale(1.0 / bbox_size.max())

        # Voxelize predicted and ground truth meshes with fixed bounds and resolution
        pitch = 1.0 / voxel_resolution
        pred_voxel = pred_mesh.voxelized(pitch=pitch)
        gt_voxel = gt_mesh.voxelized(pitch=pitch)

        # 确保体素化边界相同
        pred_voxel_matrix = pred_voxel.matrix.astype(bool)
        gt_voxel_matrix = gt_voxel.matrix.astype(bool)

        # 使用形态学操作填充空心部分
        from scipy.ndimage import binary_fill_holes
        pred_voxel_matrix = binary_fill_holes(pred_voxel_matrix)
        gt_voxel_matrix = binary_fill_holes(gt_voxel_matrix)

        # 调整体素矩阵的形状以确保一致
        max_shape = np.maximum(pred_voxel_matrix.shape, gt_voxel_matrix.shape)
        padded_pred = np.zeros(max_shape, dtype=bool)
        padded_gt = np.zeros(max_shape, dtype=bool)

        padded_pred[:pred_voxel_matrix.shape[0], :pred_voxel_matrix.shape[1], :pred_voxel_matrix.shape[2]] = pred_voxel_matrix
        padded_gt[:gt_voxel_matrix.shape[0], :gt_voxel_matrix.shape[1], :gt_voxel_matrix.shape[2]] = gt_voxel_matrix

        # Calculate intersection and union
        intersection = np.logical_and(padded_pred, padded_gt).sum()
        union = np.logical_or(padded_pred, padded_gt).sum()

        # Calculate IoU
        if union == 0:
            return 0  # Avoid division by zero, return IoU as 0 if union is zero
        iou = intersection / union
        return iou


    def calculate_cd(self, pred, label):
        pred_sample_points=pred.sample(10000)
        gt_sample_points=label.sample(10000)
        fst=fscore(pred_sample_points,gt_sample_points)

        # 计算psnr
        psnr_value = self.calculate_psnr(pred_sample_points, gt_sample_points)
        # 计算IoU
        iou_value = self.calculate_iou(pred, label)

        pred_sample_gpu=torch.from_numpy(pred_sample_points).float().cuda().unsqueeze(0)
        gt_sample_gpu=torch.from_numpy(gt_sample_points).float().cuda().unsqueeze(0)
        dist1,dist2=dist_chamfer(gt_sample_gpu,pred_sample_gpu)[:2]
        cd_loss=torch.mean(dist1)+torch.mean(dist2)
        # return cd_loss.item()*1000, fst
        return cd_loss.item()*1000, fst, psnr_value, iou_value

    def get_result(self):
        total_cd = 0
        total_number = 0
        total_score = 0
        total_psnr = 0
        total_IoU = 0
        for key in self.cd_loss_dict:
            total_cd += np.sum(np.array(self.cd_loss_dict[key]))
            total_score += np.sum(np.array(self.f_score_dict[key]))
            total_psnr += np.sum(np.array(self.psnr_dict[key]))
            total_IoU += np.sum(np.array(self.IoU_dict[key]))

            total_number += len(self.cd_loss_dict[key])

            self.cd_loss_dict[key]=np.mean(np.array(self.cd_loss_dict[key]))
            self.f_score_dict[key]=np.mean(np.array(self.f_score_dict[key]))
            self.psnr_dict[key]=np.mean(np.array(self.psnr_dict[key]))
            self.IoU_dict[key]=np.mean(np.array(self.IoU_dict[key]))

        mean_f_score = total_score/total_number
        mean_cd = total_cd/total_number
        mean_psnr = total_psnr/total_number
        mean_IoU = total_IoU/total_number
        for key in self.cd_loss_dict:
            msg="cd/fscore/psnr/IoU loss of category %s is %f/ %f/ %f/ %f "%(key, self.cd_loss_dict[key], self.f_score_dict[key],self.psnr_dict[key],self.IoU_dict[key])
            print(msg)
            self.loginfo.append(msg)
        msg = "cd/fscore/psnr/IoU loss of mean %f/ %f/ %f/ %f "%(mean_cd, mean_f_score, mean_psnr, mean_IoU)
        print(msg)
        self.loginfo.append(msg)

        with open(self.logfile, 'a') as f:
            currentDateAndTime = datetime.now()
            time_str = currentDateAndTime.strftime('%D--%H:%M:%S')
            f.write('*'*30)
            f.write(time_str + '\n')
            for info in self.loginfo:
                f.write(info + "\n")

    def run_in_one(self, index):
        data = self.split[index]
        img_id, obj_id, classname = data

        # load truth size
        img_path = os.path.join(self.cfg['data_path'], img_id)
        post_fix = img_path.split('.')[-1]      # avoid '.png' '.jpg' '.jpeg'
        if 'rgb' in img_path:
            anno_path = img_path.replace('rgb', 'annotation').replace(f'.{post_fix}', '.json')
        else:
            anno_path = img_path.replace('img', 'annotation').replace(f'.{post_fix}', '.json')
        if not os.path.exists(anno_path):
            print(f'anno_path {anno_path} not exists')
            return 
        with open(anno_path, 'r') as f:
            sequence = json.load(f)             # load annotation
        size = np.array(sequence['obj_dict'][obj_id]['half_length'])

        img_id = img_id.split('/')[-1].split('.')[0]
        output_folder = os.path.join(self.cfg['result_dir'], classname, f'{str(img_id)}_{str(obj_id)}')
        pred_cube_mesh_path = os.path.join(output_folder, 'pred_cube.ply')
        gt_cube_mesh_path = os.path.join(output_folder, 'label_cube.ply')

        output_folder = os.path.join(output_folder, f'object_resize')
        os.makedirs(output_folder, exist_ok=True)

        align_mesh_path = os.path.join(output_folder, 'align.ply')
        if not os.path.exists(pred_cube_mesh_path):
            print(pred_cube_mesh_path)
            self.loginfo.append(f'pred: {pred_cube_mesh_path} is not exist!')
            return
        if not os.path.exists(gt_cube_mesh_path):
            print(gt_cube_mesh_path)
            self.loginfo.append(f'pred: {gt_cube_mesh_path} is not exist!')
            return
        
        # 添加内容检查
        try:
            pred_mesh = trimesh.load(pred_cube_mesh_path)
            gt_mesh = trimesh.load(gt_cube_mesh_path)

            # 检查文件内容是否为空
            if pred_mesh.vertices.shape[0] == 0 or gt_mesh.vertices.shape[0] == 0:
                print(f"Skipping {pred_cube_mesh_path} or {gt_cube_mesh_path}: empty mesh content.")
                self.loginfo.append(f"Skipping {pred_cube_mesh_path} or {gt_cube_mesh_path}: empty mesh content.")
                return
        except Exception as e:
            print(f"Error loading meshes: {e}")
            self.loginfo.append(f"Error loading meshes: {e}")
            return        

        if classname not in self.cd_loss_dict.keys():
            self.cd_loss_dict[classname]=[]
        if classname not in self.f_score_dict.keys():
            self.f_score_dict[classname] = []
        if classname not in self.psnr_dict.keys():
            self.psnr_dict[classname] = []
        if classname not in self.IoU_dict.keys():
            self.IoU_dict[classname] = []

        if not(os.path.exists(os.path.join(output_folder, 'cd.txt')) or
            os.path.exists(os.path.join(output_folder, 'f_score.txt')) or
            os.path.exists(os.path.join(output_folder, 'psnr.txt')) or
            os.path.exists(os.path.join(output_folder, 'IoU.txt'))):
            # load mesh files
            pred_mesh = trimesh.load(pred_cube_mesh_path)
            gt_mesh = trimesh.load(gt_cube_mesh_path)

            pred_mesh.vertices=pred_mesh.vertices/2*size/np.max(size)*2
            gt_mesh.vertices=gt_mesh.vertices/2*size/np.max(size)*2
            pred_mesh_path = os.path.join(output_folder, 'pred.ply')
            gt_mesh_path = os.path.join(output_folder, 'gt.ply')
            pred_mesh.export(pred_mesh_path)
            gt_mesh.export(gt_mesh_path)

            cmd = f"{self.cmd_app} {pred_mesh_path} {gt_mesh_path} {align_mesh_path}"

            ## align mesh use icp
            if os.path.exists(align_mesh_path):
                subprocess.check_output(cmd, shell=True)
            try:
                align_mesh = trimesh.load(align_mesh_path)
            except:
                subprocess.check_output(cmd, shell=True)
                align_mesh = trimesh.load(align_mesh_path)
            ## calculate the cd
            cd_loss, fscore, psnr, IoU = self.calculate_cd(align_mesh, gt_mesh)
        else:

            try:
                with open(os.path.join(output_folder, 'cd.txt'), 'r') as f:
                    cd_loss = f.readline().strip()
                    if not cd_loss:  # 检查是否为空
                        print(f"File is empty: {f.name}")
                    cd_loss = float(cd_loss)
            except Exception as e:
                print(f"Error processing 'cd.txt': {e}")

            try:
                with open(os.path.join(output_folder, 'f_score.txt'), 'r') as f:
                    fscore = f.readline().strip()
                    if not fscore:  # 检查是否为空
                        print(f"File is empty: {f.name}")
                    fscore = float(fscore)
            except Exception as e:
                print(f"Error processing 'f_score.txt': {e}")

            try:
                with open(os.path.join(output_folder, 'psnr.txt'), 'r') as f:
                    psnr = f.readline().strip()
                    if not psnr:  # 检查是否为空
                        print(f"File is empty: {f.name}")
                    psnr = float(psnr)
            except Exception as e:
                print(f"Error processing 'psnr.txt': {e}")

            try:
                with open(os.path.join(output_folder, 'IoU.txt'), 'r') as f:
                    IoU = f.readline().strip()
                    if not IoU:  # 检查是否为空
                        print(f"File is empty: {f.name}")
                    IoU = float(IoU)
            except Exception as e:
                print(f"Error processing 'IoU.txt': {e}")


        self.cd_loss_dict[classname].append(cd_loss)
        self.f_score_dict[classname].append(fscore)
        self.psnr_dict[classname].append(psnr)
        self.IoU_dict[classname].append(IoU)

        msg="processing %d/%d %s_%s ,class %s, cd loss: %f, fscore: %f, psnr: %f, IoU: %f" % (index,len(self.split), img_id, str(obj_id),classname,cd_loss, fscore, psnr, IoU)
        with open(os.path.join(output_folder, 'cd.txt'), 'w') as f:
            f.write(str(cd_loss))
        with open(os.path.join(output_folder, 'f_score.txt'), 'w') as f:
            f.write(str(fscore))
        with open(os.path.join(output_folder, 'psnr.txt'), 'w') as f:
            f.write(str(psnr))
        with open(os.path.join(output_folder, 'IoU.txt'), 'w') as f:
            f.write(str(IoU))

        print(msg)
        self.loginfo.append(msg)

    def run(self):
        for index, data in enumerate(self.split):
            self.run_in_one(index)
        self.get_result()


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ssr evaluate')
    parser.add_argument('--config', type=str, required=True, help='configure file for training or testing.')
    return parser.parse_args()


if __name__ == '__main__':
    fix_random_seed(seed=1029)

    # render = Render(off_screen=True)
    args = parse_args()

    with open(args.config, 'r') as f:
        eval_cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset = eval_cfg['data']['dataset']
    exp_folder = os.path.join(eval_cfg['save_root_path'], eval_cfg['exp_name'])
    testset = eval_cfg['data']['test_class_name']

    mode ='test'
    config = {
        'result_dir':  os.path.join(exp_folder, 'out'),
        'split_path': f'./dataset/{dataset}/split/test',
        'data_path': eval_cfg['data']['data_path'],
        'log': 'EvaluateLog.txt',
        'debug': True,
        'class_name': testset
    }
    evaluate = Evaluate(config)

    t1 = time.time()
    mp.set_start_method('spawn')
    with parallel_backend('multiprocessing', n_jobs=4):
        Parallel()(delayed(evaluate.run_in_one)(index) for index in range(len(evaluate.split)))

    # check all object align
    evaluate.run()

    t2 = time.time()
    print(f'total time {t2 - t1}s')