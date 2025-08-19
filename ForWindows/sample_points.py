import argparse

import numpy as np
import open3d as o3d
import torch
from open3d import geometry
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from tqdm import tqdm

from deep_sdf.io import read_from_plyfile, write_to_ply


def fps(points, npoint):
    """
    points: [N, 3] array containing the whole point cloud
    npoint: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(npoint, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of npoint
    for i in tqdm(range(1, npoint)):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]


def farthest_point_sample(points, npoint):
    """
    Input:
        points: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B, N, C = points.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    view_shape = list(centroids.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(centroids.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, centroids, :]

    return new_points


def uniform_sample_points(verts, faces, npoint):
    # use open3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices, mesh.triangles = o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    sampled_mesh_points = geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=npoint)
    sampled_mesh_points = np.asarray(sampled_mesh_points.points)

    return sampled_mesh_points


def uniform_sample_points_from_meshes(verts, faces, npoint):
    # use pytorch3d
    meshes = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0)).cuda()
    sampled_mesh_points = sample_points_from_meshes(meshes, num_samples=npoint)[0]

    return sampled_mesh_points


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Downsample input point cloud.")
    arg_parser.add_argument(
        "--file",
        "-f",
        dest="filename",
        required=True,
        help="The filename of input point cloud data.",
    )
    arg_parser.add_argument(
        "--nsample",
        "-n",
        dest="nsample",
        required=True,
        help="The number of sample points.",
    )
    args = arg_parser.parse_args()

    v, f = read_from_plyfile(args.filename)
    sampled_points = fps(v, int(args.nsample))
    write_to_ply(args.filename[:-4] + '_downsampled.ply', sampled_points)
