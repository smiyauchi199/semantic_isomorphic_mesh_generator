#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import plyfile
import random
import torch
import torch.utils.data
import tqdm

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split, point_cloud=False, missing=False, noise=False):
    if point_cloud:
        subdir = ws.point_cloud_sdf_samples_subdir
    else:
        subdir = ws.sdf_samples_subdir
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                if missing:
                    missing_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                    for ratio in missing_ratios:
                        instance_filename = os.path.join(
                            dataset, class_name, instance_name + "_missing_{}.npz".format(ratio)
                        )
                        if not os.path.isfile(
                            os.path.join(data_source, subdir, instance_filename)
                        ):
                            # raise RuntimeError(
                            #     'Requested non-existent file "' + instance_filename + "'"
                            # )
                            logging.warning(
                                "Requested non-existent file '{}'".format(instance_filename)
                            )
                        npzfiles += [instance_filename]
                elif noise:
                    noise_ratios = [0.2, 0.4, 0.6]
                    standard_deviations = [0.01, 0.02, 0.03, 0.04, 0.05]
                    for ratio in noise_ratios:
                        for sv in standard_deviations:
                            instance_filename = os.path.join(
                                dataset, class_name, instance_name + "_noised_{}_{}.npz".format(ratio, sv)
                            )
                            if not os.path.isfile(
                                os.path.join(data_source, subdir, instance_filename)
                            ):
                                # raise RuntimeError(
                                #     'Requested non-existent file "' + instance_filename + "'"
                                # )
                                logging.warning(
                                    "Requested non-existent file '{}'".format(instance_filename)
                                )
                            npzfiles += [instance_filename]
                else:
                    instance_filename = os.path.join(
                        dataset, class_name, instance_name + ".npz"
                    )
                    if not os.path.isfile(
                        os.path.join(data_source, subdir, instance_filename)
                    ):
                        # raise RuntimeError(
                        #     'Requested non-existent file "' + instance_filename + "'"
                        # )
                        logging.warning(
                            "Requested non-existent file '{}'".format(instance_filename)
                        )
                    npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def read_surface_samples_into_ram(filename_surface, filename_normalization):
    # load template structured mesh
    surface = plyfile.PlyData.read(filename_surface)
    surface_v = [] #template.elements[0]
    for i in range(surface.elements[0].count):
        v = surface.elements[0][i]
        surface_v.append(np.array((v[0], v[1], v[2])))
    surface_v = np.asarray(surface_v)

    norm_params = np.load(filename_normalization)

    normalized_surface = (surface_v + norm_params['offset']) * norm_params['scale']
    pos_tensor = torch.from_numpy(normalized_surface[:surface_v.shape[0]//2])
    neg_tensor = torch.from_numpy(normalized_surface[surface_v.shape[0]//2:])

    # store sdf to zero
    pos_tensor = torch.cat([pos_tensor, torch.zeros(pos_tensor.shape[0], 1)], dim=1)
    neg_tensor = torch.cat([neg_tensor, torch.zeros(neg_tensor.shape[0], 1)], dim=1)

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(samples.shape[0])
    samples = torch.index_select(samples, 0, randidx)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]
    # print(data[2])  # debug

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(samples.shape[0])
    samples = torch.index_select(samples, 0, randidx)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for i, f in tqdm.tqdm(enumerate(self.npyfiles), ascii=True):
                if i == 1:
                    break
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        # os.path.basename(filename).split('.')[0],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
