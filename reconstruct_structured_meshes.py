#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
from tqdm import tqdm

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.io import read_from_plyfile, write_to_ply


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    tensorboard_saver=None
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        # near points loss
        inputs = torch.cat([latent_inputs, xyz.float()], 1).cuda()
        pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        loss = loss_l1(pred_sdf, sdf_gt)

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.item())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.item()

        if tensorboard_saver is not None:
            ws.save_tensorboard_logs(
                tensorboard_saver, e,
                loss_sdf=loss_num, loss_reg=torch.mean(latent.pow(2))
                )

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
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
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=10,
        help="random seed",
    )
    arg_parser.add_argument(
        "--batch_split",
        type=int,
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--pointcloud",
        dest="pointcloud",
        action="store_true",
        help="Set label of using point cloud.",
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
        "--interpolation",
        dest="interpolation",
        action="store_true",
        help="Set label of interpolate latent codes between tow models.",
    )
    arg_parser.add_argument(
        "--first_id",
        dest="first_id",
        type=int,
        default=0,
        help="first_id.",
    )
    arg_parser.add_argument(
        "--second_id",
        dest="second_id",
        type=int,
        default=1,
        help="second_id.",
    )
    arg_parser.add_argument(
        "--num_interpolation",
        dest="num_interpolation",
        type=int,
        default=1,
        help="number of interpolation.",
    )
    arg_parser.add_argument(
        "--modelId",
        "-m",
        dest="modelId",
        default=None,
        help="The instance model with modelId.",
    )

    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()

    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    class_name = specs_filename.split('/', 3)[1].split('_')[0][:-1]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder", "Generator"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    generator = arch.Generator(latent_size, **specs["NetworkSpecs"]["generator_kargs"])

    decoder = torch.nn.DataParallel(decoder)
    generator = torch.nn.DataParallel(generator)

    # load checkpoint parameters
    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])

    saved_generator_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.generator_params_subdir, args.checkpoint + ".pth"
        )
    )
    generator.load_state_dict(saved_generator_state["model_state_dict"])

    decoder = decoder.module.cuda()
    generator = generator.module.cuda()

    # load template mesh
    template_filename = os.path.join(args.experiment_directory,
                                     ws.training_meshes_subdir,
                                     str(saved_model_epoch), 'template_smg')
    template_v, template_f = read_from_plyfile(template_filename)

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.modelId:
        npz_filenames = [args.modelId + '.ply']
        reconstruction_dir = args.data_source
    elif args.pointcloud:
        if args.missing:
            npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split, point_cloud=True, missing=True)
            reconstruction_dir = "{}/master-results/proposed/{}/completion".format(args.data_source, class_name)
        elif args.noise:
            npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split, point_cloud=True, noise=True)
            reconstruction_dir = "{}/master-results/proposed/{}/noise".format(args.data_source, class_name)
        elif args.interpolation:
            npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split, point_cloud=True)
            reconstruction_dir = "{}/master-results/proposed/{}/interpolation".format(args.data_source, class_name)
            latents = []
        else:
            npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split, point_cloud=True)
            reconstruction_dir = "{}/master-results/proposed/{}/reconstruction".format(args.data_source, class_name)
    else:
        npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)
        reconstruction_dir = "{}/master-results/proposed/{}/reconstruction".format(args.data_source, class_name)

    npz_filenames = sorted(npz_filenames)
    if args.interpolation:
        npz_filenames = npz_filenames[[args.first_id, args.second_id]]

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    if args.pointcloud:
        reconstruction_meshes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_meshes_with_points_subdir
        )
        reconstruction_codes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_codes_with_points_subdir
        )
    else:
        reconstruction_meshes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_meshes_subdir
        )
        reconstruction_codes_dir = os.path.join(
            reconstruction_dir, ws.reconstruction_codes_subdir
        )

    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        # clamping_function = lambda x: x * specs["ClampingDistance"]
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 2

    tensorboard_saver = ws.create_tensorboard_saver(reconstruction_dir)

    for ii, npz in enumerate(tqdm(npz_filenames)):
        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.point_cloud_sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, os.path.basename(npz[:-4])
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, os.path.basename(npz[:-4]) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, os.path.basename(npz[:-4]) + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            start_time = time.perf_counter()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.01,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                tensorboard_saver=tensorboard_saver
            )
            logging.info("reconstruct time: {}".format(time.time() - start))
            logging.info("reconstruction error: {}".format(err))
            err_sum += err

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                lat_vecs = latent.expand(template_v.shape[0], -1)
                xyz = torch.chunk(torch.from_numpy(template_v).float(), args.batch_split)
                latent_vector = torch.chunk(lat_vecs.cpu(), args.batch_split)

                generator.eval()

                inverse_warped_xyz = []
                for j in range(args.batch_split):
                    with torch.no_grad():
                        interm_input = torch.cat(
                            [latent_vector[j], xyz[j]], dim=1).cuda()
                        _, _, tmp_inverse_warped_xyz_list = generator(interm_input)
                        inverse_warped_xyz.append(tmp_inverse_warped_xyz_list[-1].cpu())
                inverse_warped_xyz = np.concatenate(inverse_warped_xyz)

                write_to_ply(mesh_filename + '.ply',
                                 inverse_warped_xyz, faces=template_f)
                # print('test end... {} s'.format(time.perf_counter()-start_time))

                logging.debug("total time: {}".format(time.time() - start))


            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)

            if args.interpolation:
                latents.append(latent)

    if args.interpolation:
        latents = torch.stack([latents[0] + (latents[1] - latents[0])*t for t in torch.linspace(0, 1, args.num_interpolation+2)])

        for i, latent in enumerate(latents):
            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()

                lat_vecs = latent.expand(template_v.shape[0], -1)
                xyz = torch.chunk(torch.from_numpy(template_v).float(), args.batch_split)
                latent_vector = torch.chunk(lat_vecs.cpu(), args.batch_split)

                generator.eval()

                inverse_warped_xyz = []
                for j in range(args.batch_split):
                    with torch.no_grad():
                        interm_input = torch.cat(
                            [latent_vector[j], xyz[j]], dim=1).cuda()
                        _, _, tmp_inverse_warped_xyz_list = generator(interm_input)
                        inverse_warped_xyz.append(tmp_inverse_warped_xyz_list[-1].cpu())
                inverse_warped_xyz = np.concatenate(inverse_warped_xyz)

                # store canonical coordinates as rgb color (in float format)

                first_modelId = npz_filenames[args.first_id][:-4]
                second_modelId = npz_filenames[args.second_id][:-4]
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, "{}_{}_interpolation{}.ply".format(first_modelId, second_modelId, i)
                )
                if not os.path.exists(os.path.dirname(mesh_filename)):
                    os.makedirs(os.path.dirname(mesh_filename))

                write_to_ply(mesh_filename, inverse_warped_xyz, template_f)
                # print('test end... {} s'.format(time.perf_counter()-start_time))

                logging.debug("total time: {}".format(time.time() - start))

