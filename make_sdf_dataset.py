import argparse
import json
import os

import numpy as np
import point_cloud_utils as pcu
from numpy.linalg import norm
from scipy.spatial import distance
from tqdm import tqdm

import deep_sdf.workspace as ws
from deep_sdf.io import write_to_ply
from sample_points import fps


def smooth_normals(verts, normals):
    dist = distance.cdist(verts, verts, "euclidean")
    indices = np.argsort(dist, axis=1)[:, :10]
    smoothed_normals = np.mean(normals[indices], axis=1)  # 2500 x 5 x 3

    return smoothed_normals


def gaussian_filter(verts, sigma=1.0):
    dist = distance.cdist(verts, verts, "euclidean")
    nn_k = 10
    indices = np.argsort(dist, axis=1)[:, :nn_k+1]
    nn_dist = np.sort(dist, axis=1)[:, :nn_k+1]

    exp_numerator = np.multiply(nn_dist, nn_dist)[:, :, np.newaxis].repeat(3).reshape(nn_dist.shape[0], nn_dist.shape[1], 3)
    weights = 1 / (2 * np.pi * sigma**2) * np.exp(-exp_numerator / (2 * sigma**2))
    weights /= np.sum(weights, axis=1)[:, np.newaxis, :]

    filtered_verts = np.sum(np.multiply(weights, verts[indices]), axis=1)

    return filtered_verts


def jitter_perturbation_point_cloud(verts, ratio=0.2, sigma=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = verts.shape
    jittered_indices = np.random.randint(N, size=int(N*ratio))
    jittered_verts = sigma * 0.5 * np.random.randn(int(N*ratio), C)  # standard deviations of unit sphere's radius
    jittered_verts[:,3:] = 0
    verts[jittered_indices] += jittered_verts
    return verts


def generate_near_surface_sdf(verts, normals, etas=[0.01]):
    # set 0 distance to verts
    surface_dist = np.zeros([verts.shape[0]])
    sdf_surface = np.concatenate([verts, surface_dist[:, np.newaxis]], axis=1)

    # sample near points
    normals = np.asarray([v / norm(v, ord=2) for v in normals])
    
    # is positive normal?
    pos_pts = verts + 1.0 * normals
    neg_pts = verts - 1.0 * normals
    pos_dist = distance.cdist(pos_pts, np.zeros_like(pos_pts), "euclidean")[:, 0]
    neg_dist = distance.cdist(neg_pts, np.zeros_like(neg_pts), "euclidean")[:, 0]
    if pos_dist.sum() < neg_dist.sum():
        normals *= -1

    etas = [0.001*i for i in range(10)] + [0.01]
    # etas = [0.0005*i for i in range(20)] + [0.01]
    sdf_pos = []
    sdf_neg = []
    for i, eta in enumerate(etas):
        pos_pts = verts + eta * normals
        neg_pts = verts - eta * normals
        sdf_pos.append(np.concatenate([pos_pts, eta * np.ones([pos_pts.shape[0], 1])], axis=1))
        sdf_neg.append(np.concatenate([neg_pts, -eta * np.ones([neg_pts.shape[0], 1])], axis=1))
    sdf_pos = np.concatenate(sdf_pos, axis=0)
    sdf_neg = np.concatenate(sdf_neg, axis=0)

    return sdf_surface, sdf_pos, sdf_neg, normals


def predict_sdf(verts):
    verts_normals = pcu.estimate_normals(verts, k=16)  # xyz: 点群データ
    # verts_normals = pcu.estimate_point_cloud_normals(v, k=16)
    verts_normals = smooth_normals(verts, verts_normals)
    surface, pos, neg, verts_normals = generate_near_surface_sdf(verts, verts_normals, etas=[0.005])
    pos = np.concatenate([surface, pos], axis=0)
    neg = np.concatenate([surface, neg], axis=0)

    return verts_normals, pos, neg


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Make SDF dataset with point cloud.")
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The preprocessed data source directory.",
    )
    arg_parser.add_argument(
        "--split_file",
        "-s",
        dest="split_file",
        required=True,
        help="The split file of target data.",
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
    args = arg_parser.parse_args()

    with open(args.split_file, 'r') as f:
        split_data = json.load(f)
    for dataset_name in split_data.keys():
        for synsetId in split_data[dataset_name].keys():
            model_list = split_data[dataset_name][synsetId]
    model_list = sorted(model_list)

    class_name = os.path.basename(args.split_file).split('_', 3)[1][:-1]

    normalization_params_dir = "{}/NormalizationParameters/ShapeNetV2/{}".format(args.data_source, ws.name2synsetId[class_name])
    pointcloud_dir = "{}/ShapeNetV1PointCloud/{}".format(args.data_source, ws.name2synsetId[class_name])

    output_npz_dir = "{}/PointCloudSdfSamples/ShapeNetV2/{}".format(args.data_source, ws.name2synsetId[class_name])
    if not os.path.exists(output_npz_dir):
        os.makedirs(output_npz_dir)

    if args.missing:
        output_points_dir = "{}/master-results/proposed/{}/completion/Points".format(args.data_source, class_name)
    elif args.noise:
        output_points_dir = "{}/master-results/proposed/{}/noise/Points".format(args.data_source, class_name)
    else:
        output_points_dir = "{}/master-results/proposed/{}/reconstruction/Points".format(args.data_source, class_name)
    if not os.path.exists(output_points_dir):
        os.makedirs(output_points_dir)

    for idx, modelId in enumerate(tqdm(model_list)):
        pointcloud_file =  os.path.join(pointcloud_dir, "{}.points.ply.npy".format(modelId))
        v = np.load(pointcloud_file)

        normparams_file =  os.path.join(normalization_params_dir, "{}.npz".format(modelId))
        normparams = np.load(normparams_file)

        # normalize ShapeNetV1PointCloud to unit sphere
        tmp_v = v.copy()
        v[:, 0] = tmp_v[:, 2]
        v[:, 2] = -tmp_v[:, 0]
        v = v * normparams['scale']

        # sampling 2,500 vertices (疎な点群にするためなので頂点数は適当)
        v = fps(v, 2500)

        # sdf samples
        num_samples = 100000  # 5000

        if args.missing:
            missing_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            missing_id = np.random.randint(v.shape[0])
            for ratio in missing_ratios:
                dist = distance.cdist(v[missing_id][np.newaxis,:], v, "euclidean")[0]
                indices = np.argsort(dist)[int(ratio*v.shape[0]):]
                missing_v = v[indices]
                verts_normals, pos, neg = predict_sdf(missing_v)
                write_to_ply("{}/{}_missing_{}.ply".format(output_points_dir, modelId, ratio), missing_v, verts_normals=verts_normals)
                np.savez("{}/{}_missing_{}".format(output_npz_dir, modelId, ratio), pos=pos, neg=neg)
        elif args.noise:
            noise_ratios = [0.2, 0.4, 0.6]
            standard_deviations = [0.01, 0.02, 0.03, 0.04, 0.05]
            for r in noise_ratios:
                for sv in standard_deviations:
                    noised_v = jitter_perturbation_point_cloud(v, ratio=r, sigma=sv)
                    verts_normals, pos, neg = predict_sdf(noised_v)
                    write_to_ply("{}/{}_noised_{}_{}.ply".format(output_points_dir, modelId, r, sv), noised_v, verts_normals=verts_normals)
                    np.savez("{}/{}_noised_{}_{}".format(output_npz_dir, modelId, r, sv), pos=pos, neg=neg)
        else:
            verts_normals, pos, neg = predict_sdf(v)
            write_to_ply("{}/{}.ply".format(output_points_dir, modelId), v, verts_normals=verts_normals)
            np.savez("{}/{}".format(output_npz_dir, modelId), pos=pos, neg=neg)

