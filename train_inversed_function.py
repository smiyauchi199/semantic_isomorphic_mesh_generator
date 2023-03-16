#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn
import torch.utils.data as data_utils

import deep_sdf
import deep_sdf.loss as loss
from deep_sdf.lr_schedule import get_learning_rate_schedules
import deep_sdf.workspace as ws
from metrics.chamfer import ChamferDistance
from metrics.emd_module import emdModule


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def apply_curriculum_reconstruct_loss(inverse_warped_xyz_list, xyz, loss_l1_soft, num_sdf_samples):
    soft_l1_eps_list = [2.5e-2, 1e-2, 2.5e-3, 0]
    soft_l1_lamb_list = [0, 0.1, 0.2, 0.5]
    reconstruct_loss = []
    for k in range(len(inverse_warped_xyz_list)):
        eps = soft_l1_eps_list[k]
        lamb = soft_l1_lamb_list[k]
        l = loss_l1_soft(inverse_warped_xyz_list[k], xyz,
                         eps=eps, lamb=lamb) / num_sdf_samples
        # l = loss_l1(pred_sdf_list[k], sdf_gt[i].cuda()) / num_sdf_samples
        reconstruct_loss.append(l)
    reconstruct_loss = sum(reconstruct_loss) / len(reconstruct_loss)
    return reconstruct_loss


def CD_normal_loss(esti_shapes, shapes):
    dist1, dist2, idx1, idx2 = ChamferDistance()(
        esti_shapes, shapes, bidirectional=True)
    loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    return loss_cd


def EMD_loss(esti_shapes, shapes):
    dist, assigment = emdModule()(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean(1).mean()
    return loss_emd


def main_function(experiment_directory, data_source, continue_from, checkpoint, batch_split):

    logging.info("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    # data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__(
        "networks." + specs["NetworkArch"], fromlist=["Decoder", "Generator", "InverseImplicitFunction"])

    logging.info(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for ckpt in specs["AdditionalSnapshots"]:
        checkpoints.append(ckpt)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        ws.save_generator(experiment_directory, "latest.pth", generator, epoch)
        ws.save_generator_optimizer(experiment_directory,
                                    "latest.pth", optimizer_all, epoch)

    def save_checkpoints(epoch):
        ws.save_generator(experiment_directory, str(
            epoch) + ".pth", generator, epoch)
        ws.save_generator_optimizer(experiment_directory, str(
            epoch) + ".pth", optimizer_all, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist

    # requirements for computing chamfer loss
    assert(scene_per_batch % batch_split == 0)
    scene_per_split = scene_per_batch // batch_split

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    generator = arch.Generator(
        latent_size, **specs["NetworkSpecs"]["generator_kargs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    if torch.cuda.device_count() > 1:
        decoder = torch.nn.DataParallel(decoder)
        generator = torch.nn.DataParallel(generator)

    saved_model_state = torch.load(
        os.path.join(experiment_directory,
                     ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=True
    )

    if sdf_dataset.load_ram:
        num_data_loader_threads = 0
    else:
        num_data_loader_threads = get_spec_with_default(
            specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(
        num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.info(decoder)
    logging.info(generator)

    latent_vectors = ws.load_pre_trained_latent_vectors(
        experiment_directory, checkpoint)
    latent_vectors = latent_vectors.cuda()

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_l1_soft = loss.SoftL1Loss(reduction="sum")
    loss_lp = torch.nn.DataParallel(loss.LipschitzLoss(k=0.5, reduction="sum"))
    huber_fn = loss.HuberFunc(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": generator.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
        ]
    )

    tensorboard_saver = ws.create_tensorboard_saver(experiment_directory)

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:
        if not os.path.exists(os.path.join(experiment_directory, ws.latent_codes_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.model_params_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.optimizer_params_subdir, continue_from + ".pth")):
            logging.warning(
                '"{}" does not exist! Ignoring this argument...'.format(continue_from))
        else:
            logging.info('continuing from "{}"'.format(continue_from))

            model_epoch = ws.load_generator_parameters(
                experiment_directory, continue_from, generator
            )

            optimizer_epoch = ws.load_generator_optimizer(
                experiment_directory, continue_from + ".pth", optimizer_all
            )

            start_epoch = model_epoch + 1

            logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of generator parameters: {}".format(
            sum(p.data.nelement() for p in generator.parameters())
        )
    )

    use_curriculum = get_spec_with_default(specs, "UseCurriculum", False)
    use_chamfer_loss = get_spec_with_default(specs, "UseChamferLoss", False)
    use_emd_loss = get_spec_with_default(specs, "UseEmdLoss", False)

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.eval()
        generator.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        batch_num = len(sdf_loader)
        for bi, (sdf_data, indices) in enumerate(sdf_loader):
            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            batch_loss_reconstruct = 0.0
            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = latent_vectors[indices[i]]

                # NN optimization
                input = torch.cat([batch_vecs, xyz[i].cuda()], dim=1)
                xyz_ = xyz[i].cuda()
                warped_xyz_list, _, _ = decoder(
                    input, output_warped_points=True, output_warping_param=True, generator=True)

                interm_input = torch.cat(
                    [batch_vecs, warped_xyz_list[-1]], dim=1)
                _, _, inverse_warped_xyz_list = generator(interm_input)

                if use_curriculum:
                        reconstruct_loss = apply_curriculum_reconstruct_loss(
                            inverse_warped_xyz_list, xyz_, loss_l1_soft, num_sdf_samples)
                else:
                    if use_chamfer_loss and use_emd_loss:
                        loss_cd = CD_normal_loss(
                            inverse_warped_xyz_list[-1], xyz_)
                        loss_emd = EMD_loss(inverse_warped_xyz_list[-1], xyz_)
                        reconstruct_loss = loss_cd + loss_emd
                    elif use_chamfer_loss:
                        loss_cd = CD_normal_loss(
                            inverse_warped_xyz_list[-1], xyz_)
                        reconstruct_loss = loss_cd
                    else:
                        reconstruct_loss = loss_l1(
                            inverse_warped_xyz_list[-1], xyz_) / num_sdf_samples
                batch_loss_reconstruct += reconstruct_loss.item()
                chunk_loss = reconstruct_loss

                chunk_loss.backward()
                batch_loss += chunk_loss.item()

            logging.debug("reconstruct_loss = {:.9f}".format(
                batch_loss_reconstruct))

            ws.save_tensorboard_logs(
                tensorboard_saver, epoch*batch_num + bi,
                loss_reconstruct=batch_loss_reconstruct)

            loss_log.append(batch_loss)

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

            # release memory
            del warped_xyz_list, reconstruct_loss, batch_loss_reconstruct, batch_loss, chunk_loss

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch)
                      for schedule in lr_schedules])

        append_parameter_magnitudes(param_mag_log, generator)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:
            save_latest(epoch)


if __name__ == "__main__":
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    arg_parser = argparse.ArgumentParser(
        description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--continue",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
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
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.data_source,
                  args.continue_from, args.checkpoint, int(args.batch_split))
