#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import plyfile
import point_cloud_utils as pcu
import torch
import tqdm

import deep_sdf
import deep_sdf.workspace as ws
import shapenet
import torch.utils.data as data_utils

from deep_sdf.io import write_to_plyfile


def get_instance_filenames(data_source, split):
    objfiles = []
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
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.normalization_param_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]

    return objfiles, npzfiles


class Shapes(torch.utils.data.Dataset):

    def __init__(self, data_source, data_list):

        self.data_source = data_source
        self.objfiles, self.npyfiles = get_instance_filenames(data_source, data_list)

        self.loaded_data = []

        #for obj_f, npz_f in tqdm(zip(self.objfiles, self.npyfiles), ascii=True):
        for i in range(len(self.objfiles)):
            obj_f = self.objfiles[i]
            npz_f = self.npyfiles[i]
            obj_filename = os.path.join(self.data_source, ws.obj_samples_subdir, obj_f)
            points, faces = pcu.load_mesh_vf(obj_filename, dtype=np.float32)

            # normalize
            npz_filename = os.path.join(self.data_source, ws.normalization_param_subdir, npz_f)
            npz = np.load(npz_filename)
            tmp_points = points.copy()
            points[:, 0] = tmp_points[:, 2]
            points[:, 2] = -tmp_points[:, 0]
            normalized_points = points * npz['scale']

            self.loaded_data.append(
                [
                    normalized_points,
                    faces,
                    os.path.basename(npz_filename).split('.')[0],
                ]
            )

        print("dataset size %d" % len(self))

    def __getitem__(self, index):
        try:
            return self.loaded_data[index], index
        except Exception as e:
            warnings.warn(f"Error loading sample {index}: " + ''.join(
                traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
            import ipdb
            ipdb.set_trace()
            index += 1

    def __len__(self):
        return len(self.loaded_data)


def mesh_to_correspondence(experiment_directory, checkpoint, start_id, end_id):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename, encoding="utf-8"))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    train_dataset = Shapes(data_source, train_split)

    class_name = train_split_file.split('_', 3)[1][:-1]

    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
    )

    latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, checkpoint)
    latent_vectors = latent_vectors.cuda()
    latent_vectors.requires_grad = False

    # load template mesh
    saved_model_state = torch.load(
        os.path.join(experiment_directory,
                     ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    start = time.perf_counter()

    #for i, (points, faces, instance_name, indices) in enumerate(tqdm.tqdm(train_loader)):
    for i, (test, indices) in enumerate(tqdm.tqdm(train_loader)):
        points, faces, instance_name = test

        if i < start_id:
            continue

        # if instance_name in check_list:
        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            class_name,
        )

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name[0])
        queries = points.reshape(-1, 3).cuda()
        num_samples = queries.shape[0]
        latent_repeat = latent_vectors[indices[0]].expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
        #warped_list, _ = decoder(inputs, generator=True)
        warped_list, _  = decoder(inputs, generator=True, output_warped_points=True)

        for j, warped in enumerate(warped_list[-1:]):
            warped = warped.cpu().detach().numpy()

            # store canonical coordinates as rgb color (in float format)
            verts_color = 255 * (0.5 + 0.5 * warped)
            verts_color = verts_color.astype(np.uint8)
            write_to_plyfile(mesh_filename + '_canonical_position_mesh.ply', warped, faces=faces.detach().cpu().numpy()[0])

        if i >= end_id:
            break


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--start_id",
        dest="start_id",
        type=int,
        default=0,
        help="start_id.",
    )
    arg_parser.add_argument(
        "--end_id",
        dest="end_id",
        type=int,
        default=20,
        help="end_id.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    mesh_to_correspondence(args.experiment_directory, args.checkpoint, args.start_id, args.end_id)
