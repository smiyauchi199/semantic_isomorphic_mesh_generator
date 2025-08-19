"""
Evaluate structured meshes.
"""

import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import point_cloud_utils as pcu
import pytorch3d.loss
import torch
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

import deep_sdf.workspace as ws
from deep_sdf.metrics.mesh_metrics import mesh_acc
from deep_sdf.metrics.mesh_metrics import mesh_comp


def get_instance_filenames(data_source, split):
    objfiles = []
    plyfiles = []
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    class_name, "models", instance_name + ".obj"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.obj_samples_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                objfiles += [instance_filename]

                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".ply"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.surface_samples_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                plyfiles += [instance_filename]

                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.normalization_param_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]

    return objfiles, plyfiles, npzfiles


class Shapes(torch.utils.data.Dataset):

    def __init__(self, data_source, data_list, metric):

        self.data_source = data_source
        self.objfiles, self.plyfiles, self.npyfiles = get_instance_filenames(data_source, data_list)

        self.loaded_data = []

        for obj_f, ply_f, npz_f in tqdm(zip(self.objfiles, self.plyfiles, self.npyfiles), ascii=True):

            if metric == 'mesh_acc':
                obj_filename = os.path.join(self.data_source, ws.obj_samples_subdir, obj_f)
                points, faces = pcu.load_mesh_vf(obj_filename, dtype=np.float32)

                # normalize
                npz_filename = os.path.join(self.data_source, ws.normalization_param_subdir, npz_f)
                npz = np.load(npz_filename)
                tmp_points = points.copy()
                points[:, 0] = tmp_points[:, 2]
                points[:, 2] = -tmp_points[:, 0]
                normalized_points = points * npz['scale']
            else:
                ply_filename = os.path.join(self.data_source, ws.surface_samples_subdir, ply_f)
                points = pcu.load_mesh_v(ply_filename)
                faces = np.asarray([], dtype=np.int8)

                # normalize
                npz_filename = os.path.join(self.data_source, ws.normalization_param_subdir, npz_f)
                npz = np.load(npz_filename)
                normalized_points = (points + npz['offset']) * npz['scale']

            self.loaded_data.append(
                [
                    normalized_points,
                    faces,
                    os.path.basename(ply_filename).split('.')[0],
                ]
            )

        print("dataset size %d" % len(self))

    def __getitem__(self, index):
        try:
            return self.loaded_data[index]
        except Exception as e:
            warnings.warn(f"Error loading sample {index}: " + ''.join(
                traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
            import ipdb
            ipdb.set_trace()
            index += 1

    def __len__(self):
        return len(self.loaded_data)


def calc_earth_movers_distance(source, target):
    # ユークリッド距離
    d = cdist(source, target)
    # 線形割当問題の解
    assignment = linear_sum_assignment(d)
    # コスト
    dist = d[assignment].sum() / source.shape[0]
    return dist


def calc_metric(metric, source_file, target_points, target_faces, threshold):
    #source_points, source_faces = pcu.load_mesh_vf(source_file)
    mesh = trimesh.load(source_file)
    source_points = mesh.vertices
    source_faces = mesh.faces
    if metric == 'chamfer':
        dist = pytorch3d.loss.chamfer_distance(torch.from_numpy(source_points.astype(np.float32)).unsqueeze(0).cpu(), target_points.cpu())[0].detach().cpu().item()
    elif metric == 'emd':
        dist = calc_earth_movers_distance(source_points.astype(np.float32), target_points[0].cpu().numpy())
    elif metric == 'mesh_acc':
        target_mesh = trimesh.Trimesh(vertices=target_points[0].cpu().numpy(), faces=target_faces[0].cpu().numpy())
        dist = mesh_acc(target_mesh, source_points)
    elif metric == 'mesh_comp':
        source_mesh = trimesh.Trimesh(vertices=source_points, faces=source_faces)
        dist = mesh_comp(source_mesh, target_points[0].cpu().numpy(), threshold)
    return dist


def main_function(source_file_dir, target_file_dir, split_file, metric, pattern=None, missing=False, noise=False, template=False, threshold=0.01):

    with open(split_file, "r") as f:
        target_list = json.load(f)

    dataset = Shapes(target_file_dir, target_list, metric)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    output_dir = source_file_dir
    if pattern:
        suffix = '_ptn' + pattern
    else:
        suffix = ''
    prefix = metric + 's'

    if template:
        json_data = {}
        source_file = os.path.join(source_file_dir, 'template_im.ply')
        target_file = os.path.join(target_file_dir, 'template.ply')
        target_points, target_faces = pcu.load_mesh_vf(target_file)
        json_data['template'] = calc_metric(metric, source_file, target_points, target_faces)
        json_object = json.dumps(json_data, indent=4)
        output_file = os.path.join(output_dir, "{}{}.json".format(prefix, suffix))
        with open(output_file, 'w') as f:
            f.write(json_object)
    elif missing:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        for ratio in tqdm(missing_ratios):
            json_data = {}
            for target_points, target_faces, instance_name in tqdm(dataloader):
                source_file = os.path.join(source_file_dir, instance_name[0] + "_missing_{}.ply".format(ratio))
                json_data[instance_name[0]] = calc_metric(metric, source_file, target_points, target_faces)
            json_object = json.dumps(json_data, indent=4)
            output_file = os.path.join(output_dir, "{}_missing_{}{}.json".format(prefix, ratio, suffix))
            with open(output_file, 'w') as f:
                f.write(json_object)
    elif noise:
        noise_ratios = [0.2, 0.4, 0.6]
        standard_deviations = [0.01, 0.02, 0.03, 0.04, 0.05]
        for ratio in noise_ratios:
            for sd in tqdm(standard_deviations):
                json_data = {}
                for target_points, target_faces, instance_name in tqdm(dataloader):
                    source_file = os.path.join(source_file_dir, instance_name[0] + "_noised_{}_{}.ply".format(ratio, sd))
                    json_data[instance_name[0]] = calc_metric(metric, source_file, target_points, target_faces)
                json_object = json.dumps(json_data, indent=4)
                output_file = os.path.join(output_dir, "{}_noise_{}_{}{}.json".format(prefix, ratio, sd, suffix))
                with open(output_file, 'w') as f:
                    f.write(json_object)
    else:
        json_data = {}
        for target_points, target_faces, instance_name in tqdm(dataloader):
            source_file = os.path.join(source_file_dir, instance_name[0] + '.ply')           
            json_data[instance_name[0]] = calc_metric(metric, source_file, target_points, target_faces, threshold)
        json_object = json.dumps(json_data, indent=4)
        output_file = os.path.join(output_dir, "{}{}.json".format(prefix, suffix))
        with open(output_file, 'w') as f:
            f.write(json_object)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--source_data",
        "-sd",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--target_data",
        "-td",
        dest="data_target",
        required=True,
        help="The data target directory.",
    )
    arg_parser.add_argument(
        "--split_file",
        "-s",
        dest="split_file",
        required=True,
        help="The split file of target data.",
    )
    arg_parser.add_argument(
        "--metric",
        dest="metric",
        choices=["chamfer", "emd", "mesh_acc", "mesh_comp"],
        help="The name of metric for evaluation.",
    )
    arg_parser.add_argument(
        "--pattern",
        dest="pattern",
        default=None,
        help="The number of sampling pattern for emd and mesh completion.",
    )
    arg_parser.add_argument(
        "--missing",
        dest="missing",
        action="store_true",
        help="Set label of using missing point cloud.",
    )
    arg_parser.add_argument(
        "--noise",
        dest="noise",
        action="store_true",
        help="Set label of using noised point cloud.",
    )
    arg_parser.add_argument(
        "--template",
        dest="template",
        action="store_true",
        help="Set label of using template point cloud.",
    )
    arg_parser.add_argument(
        "--threshold",
        dest="threshold",
        default=None,
        help="Set threshold of mesh_comp.",
    )

    args = arg_parser.parse_args()

    main_function(args.data_source, args.data_target, args.split_file, args.metric, args.pattern, args.missing, args.noise, args.template, float(args.threshold))
