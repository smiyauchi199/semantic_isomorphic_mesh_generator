#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import h5py
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import sys
import deep_sdf
import deep_sdf.workspace as ws


def sample_outer_surface_in_voxel(volume):
    # inner surface
    # a = F.max_pool3d(-volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    # b = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    # c = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0]
    # border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0)
    # surface = border + volume.float()

    # outer surface
    a = F.max_pool3d(volume[None, None].float().cuda(), kernel_size=(
        3, 1, 1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None, None].float().cuda(), kernel_size=(
        1, 3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None, None].float().cuda(), kernel_size=(
        1, 1, 3), stride=1, padding=(0, 0, 1))[0]
    border, _ = torch.max(torch.cat([a, b, c], dim=0), dim=0)
    surface = border - volume.float().cuda()
    return surface.long()


def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2 and len(
        shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2 * (vertices / (torch.max(shape) - 1) - 0.5)


def gather_voxels(sdf_list, hdf5_dir, class_name, modelIDs):

    # voxel_dim = 32
    voxel_dim = 64
    # voxel_dim = 256
    sample_num = 3000
    hdf5_path = os.path.join(
        hdf5_dir, "template_{}.hdf5".format(voxel_dim))
    print('hdf5_path: ', hdf5_path)

    hdf5_file = h5py.File(hdf5_path, 'w')
    hdf5_file.create_dataset(
        "voxels", [len(sdf_list), voxel_dim, voxel_dim, voxel_dim], np.float64, compression=9)
    hdf5_file.create_dataset(
        "labels", [len(sdf_list), voxel_dim, voxel_dim, voxel_dim], np.float64, compression=9)
    # hdf5_file.create_dataset(
    #     "surface_points", [len(sdf_list), sample_num, 3], np.float64, compression=9)
    hdf5_file.create_dataset('modelId', [len(sdf_list)], h5py.string_dtype(length=32), compression='gzip')

    for idx, sdf_values in enumerate(sdf_list):
        print(idx)

        indices = np.where(sdf_values <= 0.0)

        voxels = torch.zeros(voxel_dim, voxel_dim, voxel_dim, 1).float()
        labels = torch.zeros(voxel_dim, voxel_dim, voxel_dim, 1).float()
        voxels[indices[0], indices[1], indices[2], 0] = 1.0
        labels[indices[0], indices[1], indices[2], 0] = 1.0
        voxels = voxels.squeeze(-1)
        labels = labels.squeeze(-1)

        hdf5_file['voxels'][idx] = voxels
        hdf5_file['labels'][idx] = labels

        # shape = torch.tensor(labels.shape)[None].float().cuda()
        # y_outer = sample_outer_surface_in_voxel((labels == 1.0).long().cuda())
        # surface_points = torch.nonzero(y_outer).cuda()
        # # convert z,y,x -> x, y, z
        # # surface_points = torch.flip(surface_points, dims=[1]).float()
        # surface_points_normalized = normalize_vertices(
        #     surface_points, shape).cpu()

        # perm = torch.randperm(len(surface_points_normalized))
        # # randomly pick 3000 points
        # hdf5_file["surface_points"][idx] = surface_points_normalized[perm[:np.min(
        #     [len(perm), sample_num])]]

        hdf5_file['modelId'][idx] = "template"

    hdf5_file.close()
    print("finished")


def code_to_mesh(experiment_directory, checkpoint):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

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

    mesh_dir = os.path.join(
        experiment_directory,
        ws.training_meshes_subdir,
        str(saved_model_epoch),
    )

    class_name = os.path.basename(experiment_directory)[:-4]
    print(class_name)
    modelIDs = [class_name]

    if not os.path.isdir(mesh_dir):
        os.makedirs(mesh_dir)

    mesh_filename = os.path.join(mesh_dir, 'template')

    print(mesh_filename)
    offset = None
    scale = None

    sdf_list = []
    with torch.no_grad():
        sdf_values = deep_sdf.mesh.create_mesh(
                         decoder.forward_template,
                         None,
                         mesh_filename,
                         # N=512,
                         N=64,
                         # N=32,
                         max_batch=int(2 ** 20),
                         offset=offset,
                         scale=scale,
                         volume_size=2.0,
                         save_voxel=True
                     )
        sdf_list.append(sdf_values)

    gather_voxels(sdf_list, mesh_dir, class_name, modelIDs)


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
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(args.experiment_directory, args.checkpoint)
