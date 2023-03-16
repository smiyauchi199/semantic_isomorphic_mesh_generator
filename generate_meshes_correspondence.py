#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.io import read_from_plyfile, write_to_ply


def mesh_to_correspondence(experiment_directory, checkpoint, start_id, end_id):

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

    latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, checkpoint)
    latent_vectors = latent_vectors.cuda()

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    # load template mesh
    template_filename = os.path.join(experiment_directory,
        ws.training_meshes_subdir,
        str(saved_model_epoch), 'template')
    logging.info("Loading from %s.ply" % template_filename)
    template_v, template_f = read_from_plyfile(template_filename)

    # store canonical coordinates as rgb color (in float format)
    verts_color = 255 * (0.5 + 0.5 * template_v)
    verts_color = verts_color.astype(np.uint8)
    write_to_ply(template_filename + '_colored.ply', template_v, template_f, rgb_points=verts_color)

    for i, latent_vector in enumerate(latent_vectors):
        if i < start_id:
            continue

        if sys.platform.startswith('linux'):
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("/")
        else:
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("\\")
        instance_name = instance_name.split(".")[0]
        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)
        if os.path.exists(mesh_filename+ ".ply"):
            logging.info("Loading from %s.ply" % mesh_filename)

            mesh_v, mesh_f = read_from_plyfile(mesh_filename + '.ply')

            queries = torch.from_numpy(mesh_v).cuda()
            num_samples = queries.shape[0]
            latent_repeat = latent_vector.expand(num_samples, -1)
            inputs = torch.cat([latent_repeat, queries], 1)
            warped = []
            head = 0
            max_batch = 2**17
            while head < num_samples:
                with torch.no_grad():
                    warped_, _ = decoder(inputs[head : min(head + max_batch, num_samples)], output_warped_points=True)
                warped_ = warped_.detach().cpu().numpy()
                warped.append(warped_)
                head += max_batch
            warped = np.concatenate(warped, axis=0)

            # store canonical coordinates as rgb color (in float format)
            verts_color = 255 * (0.5 + 0.5 * warped)
            verts_color = verts_color.astype(np.uint8)
            write_to_ply(mesh_filename + '_colored.ply', mesh_v, mesh_f, rgb_points=verts_color)

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
