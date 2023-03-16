import numpy as np
import trimesh


def mesh_acc(mesh, sampled_points):
    # input - mesh : trimesh.base.Trimesh of the input mesh
    #         sampled_points : sampled points with N x 3 numpy array format
    dist = trimesh.proximity.signed_distance(mesh, sampled_points)
    print(dist)
    dist = np.abs(dist)
    dist.sort()
    print(dist)
    return dist[dist.shape[0] * 9 // 10]


def mesh_comp(mesh, sampled_points, threshold=0.01):
    # input - mesh : trimesh.base.Trimesh of the input mesh
    #         sampled_points : sampled points with N x 3 numpy array format
    dist = trimesh.proximity.signed_distance(mesh, sampled_points)
    dist = np.abs(dist)
    inlier = len(np.where(dist <= 0.01)[0])
    return inlier / dist.shape[0]
