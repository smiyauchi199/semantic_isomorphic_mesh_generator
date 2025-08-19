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
    #verts_normals = pcu.estimate_normals(verts, k=16)  # xyz: 点群データ
    verts_normals = pcu.estimate_point_cloud_normals_knn(verts, 16)[1]  # xyz: 点群データ
    # verts_normals = pcu.estimate_point_cloud_normals(v, k=16)
    verts_normals = smooth_normals(verts, verts_normals)
    surface, pos, neg, verts_normals = generate_near_surface_sdf(verts, verts_normals, etas=[0.005])
    pos = np.concatenate([surface, pos], axis=0)
    neg = np.concatenate([surface, neg], axis=0)

    return verts_normals, pos, neg


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Predict SDF for input point cloud.")
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The preprocessed data source directory.",
    )
    arg_parser.add_argument(
        "--file",
        "-f",
        dest="filename",
        required=True,
        help="The filename of input point cloud data.",
    )
    args = arg_parser.parse_args()

    output_npz_dir = "{}/PointCloudSdfSamples".format(args.data_source)
    if not os.path.exists(output_npz_dir):
        os.makedirs(output_npz_dir)

    output_points_dir = "{}/Points".format(args.data_source)
    if not os.path.exists(output_points_dir):
        os.makedirs(output_points_dir)

    # v, _, _, _ = pcu.read_ply(args.filename)
    v = pcu.load_mesh_v(args.filename)

    # sampling 2,500 vertices (疎な点群にするためなので頂点数は適当)
    v = fps(v, 2500)

    # sdf samples
    num_samples = 100000  # 5000
    verts_normals, pos, neg = predict_sdf(v)

    modelId = args.filename[:-4]
    write_to_ply("{}/{}.ply".format(output_points_dir, modelId), v, verts_normals=verts_normals)
    np.savez("{}/{}".format(output_npz_dir, modelId), pos=pos, neg=neg)

