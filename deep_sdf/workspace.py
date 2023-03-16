#!/usr/bin/env python3

import json
import os
import torch
from tensorboardX import SummaryWriter

model_params_subdir = "ModelParameters"
generator_params_subdir = "GeneratorParameters"
optimizer_params_subdir = "OptimizerParameters"
generator_optimizer_params_subdir = "GeneratorOptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_meshes_with_points_subdir = "MeshesWithPoints"
reconstruction_codes_subdir = "Codes"
reconstruction_codes_with_points_subdir = "CodesWithPoints"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
point_cloud_sdf_samples_subdir = "PointCloudSdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
obj_samples_subdir = "ShapeNetCore.v1"
training_meshes_subdir = "TrainingMeshes"
training_meshes_deformation_subdir = "TrainingMeshesDeformation"
interpolation_meshes_subdir = "TrainingMeshInterpolation"
tensorboard_log_subdir = "TensorboardLogs"
sdf_slices_sub_dir = "SdfSlices"
synsetId2name = {
    '04379243': 'table',
    '03593526': 'jar',
    '04225987': 'skateboard',
    '02958343': 'car',
    '02876657': 'bottle',
    '04460130': 'tower',
    '03001627': 'chair',
    '02871439': 'bookshelf',
    '02942699': 'camera',
    '02691156': 'airplane',
    '03642806': 'laptop',
    '02801938': 'basket',
    '04256520': 'sofa',
    '03624134': 'knife',
    '02946921': 'can',
    '04090263': 'rifle',
    '04468005': 'train',
    '03938244': 'pillow',
    '03636649': 'lamp',
    '02747177': 'trash bin',
    '03710193': 'mailbox',
    '04530566': 'watercraft',
    '03790512': 'motorbike',
    '03207941': 'dishwasher',
    '02828884': 'bench',
    '03948459': 'pistol',
    '04099429': 'rocket',
    '03691459': 'loudspeaker',
    '03337140': 'file cabinet',
    '02773838': 'bag',
    '02933112': 'cabinet',
    '02818832': 'bed',
    '02843684': 'birdhouse',
    '03211117': 'display',
    '03928116': 'piano',
    '03261776': 'earphone',
    '04401088': 'telephone',
    '04330267': 'stove',
    '03759954': 'microphone',
    '02924116': 'bus',
    '03797390': 'mug',
    '04074963': 'remote',
    '02808440': 'bathtub',
    '02880940': 'bowl',
    '03085013': 'keyboard',
    '03467517': 'guitar',
    '04554684': 'washer',
    '02834778': 'bicycle',
    '03325088': 'faucet',
    '04004475': 'printer',
    '02954340': 'cap',
}
name2synsetId = {
        'table': '04379243',
        'jar': '03593526',
        'skateboard': '04225987',
        'car': '02958343',
        'bottle': '02876657',
        'tower': '04460130',
        'chair': '03001627',
        'bookshelf': '02871439',
        'camera': '02942699',
        'airplane': '02691156',
        'laptop': '03642806',
        'basket': '02801938',
        'sofa': '04256520',
        'knife': '03624134',
        'can': '02946921',
        'rifle': '04090263',
        'train': '04468005',
        'pillow': '03938244',
        'lamp': '03636649',
        'trash bin': '02747177',
        'mailbox': '03710193',
        'watercraft': '04530566',
        'motorbike': '03790512',
        'dishwasher': '03207941',
        'bench': '02828884',
        'pistol': '03948459',
        'rocket': '04099429',
        'loudspeaker': '03691459',
        'file cabinet': '03337140',
        'bag': '02773838',
        'cabinet': '02933112',
        'bed': '02818832',
        'birdhouse': '02843684',
        'display': '03211117',
        'piano': '03928116',
        'earphone': '03261776',
        'telephone': '04401088',
        'stove': '04330267',
        'microphone': '03759954',
        'bus': '02924116',
        'mug': '03797390',
        'remote': '04074963',
        'bathtub': '02808440',
        'bowl': '02880940',
        'keyboard': '03085013',
        'guitar': '03467517',
        'washer': '04554684',
        'bicycle': '02834778',
        'faucet': '03325088',
        'printer': '04004475',
        'cap': '02954340',
}

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            'model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]


def load_generator_parameters(experiment_directory, checkpoint, generator):

    filename = os.path.join(
        experiment_directory, generator_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            'model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    generator.load_state_dict(data["model_state_dict"])

    return data["epoch"]


def build_decoder(experiment_directory, experiment_specs):

    arch = __import__(
        "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
    )

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(
        latent_size, **experiment_specs["NetworkSpecs"]).cuda()

    return decoder


def load_decoder(
    experiment_directory, experiment_specs, checkpoint, data_parallel=True
):

    decoder = build_decoder(experiment_directory, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

    return (decoder, epoch)


def load_pre_trained_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()


def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_generator(experiment_directory, filename, generator, epoch):

    generator_params_dir = get_generatorparams_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": generator.state_dict()},
        os.path.join(generator_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def save_generator_optimizer(experiment_directory, filename, optimizer, epoch):

    generator_optimizer_params_dir = get_generator_optimizer_params_dir(
        experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(generator_optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def load_generator_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        get_generator_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + ".ply",
    )


def get_structured_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + "structured_mesh.ply",
    )


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_generatorparams_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, generator_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_generator_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, generator_optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_tensorboard_logs_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, tensorboard_log_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )


def create_tensorboard_saver(experiment_dir):
    return SummaryWriter(get_tensorboard_logs_dir(experiment_dir, True))


def save_tensorboard_logs(saver, step, **kargs):
    if step % 10 == 0:
        for ln in kargs.keys():
            saver.add_scalar('loss/{}'.format(ln), kargs[ln], step)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(
                    root).as_posix(), recursive=True)
