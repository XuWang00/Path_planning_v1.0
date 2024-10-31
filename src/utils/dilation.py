import open3d as o3d
import os
import numpy as np
from scipy import ndimage
import math


def voxelization(objmesh, voxel_size=10):


    # 确保网格的法向量是正确的，这对于体素化过程可能是必要的
    objmesh.compute_vertex_normals()

    # 使用给定的体素尺寸进行体素化
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(objmesh, voxel_size)

    return voxel_grid


def generate_voxel_dilate_xyz(voxel_grid, expan_dis, limit_z):
    # 获取体素索引和体素尺寸
    voxels = np.asarray(voxel_grid.get_voxels())
    voxel_coords = np.asarray([voxel.grid_index for voxel in voxels])
    voxel_size = voxel_grid.voxel_size

    # 计算膨胀的迭代次数和空间扩张
    dilate_iter = math.ceil(expan_dis / voxel_size)
    expan_coord = dilate_iter * 2

    # 初始化膨胀空间
    max_index = np.max(voxel_coords, axis=0) + 1 + expan_coord
    expan_space = np.zeros(max_index)

    # 填充已有体素
    expan_coord_half = expan_coord // 2
    voxel_coords += expan_coord_half
    expan_space[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

    # 生成球形结构元素并进行膨胀
    se = ndimage.generate_binary_structure(3, 1)  # 生成一个球形结构元素
    expan_space = ndimage.binary_dilation(expan_space, iterations=dilate_iter, structure=se)

    # 提取膨胀后的体素坐标
    dilated_indices = np.argwhere(expan_space)
    base_coord = np.array([expan_coord_half, expan_coord_half, expan_coord_half])
    dilated_coords = (dilated_indices - base_coord) * voxel_size + voxel_grid.get_voxel_center_coordinate(
        voxel_coords[0] - expan_coord_half)

    # 过滤低于limit_z的体素
    filtered_coords = dilated_coords[dilated_coords[:, 2] >= limit_z]

    return filtered_coords


def extract_outer_layer(filtered_coords, voxel_size, dilate_iter):
    """
    对扩张后的体素进行腐蚀处理，提取最外层体素。

    参数:
    - filtered_coords: 扩张后体素的坐标数组。
    - voxel_size: 体素的尺寸。
    - dilate_iter: 扩张的迭代次数，也用于腐蚀。

    返回:
    - outer_layer_coords: 最外层体素的坐标数组。
    """
    # 为filtered_coords创建一个稀疏的三维数组
    coords_min = np.min(filtered_coords, axis=0)
    coords_max = np.max(filtered_coords, axis=0)
    shape = np.ceil((coords_max - coords_min) / voxel_size).astype(int) + 3  # 加3为边缘留出空间
    expanded_space = np.zeros(shape)
    indices = ((filtered_coords - coords_min) / voxel_size).astype(int) + 1  # 加1为边缘留出空间
    expanded_space[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    # 腐蚀
    se = ndimage.generate_binary_structure(3, 1)
    eroded_space = ndimage.binary_erosion(expanded_space, structure=se, iterations=dilate_iter)

    # 找出最外层体素
    outer_layer = np.logical_xor(expanded_space, eroded_space)
    outer_layer_indices = np.argwhere(outer_layer)

    # 转换回坐标
    outer_layer_coords = (outer_layer_indices - 1) * voxel_size + coords_min  # 减1因为之前为边缘留出空间时加了1

    return outer_layer_coords

def create_surface_from_voxels(voxel_coords, voxel_size = 10):
    """
    从体素坐标创建表面并返回可视化的几何对象。

    参数:
    - voxel_coords: 体素空间坐标的数组。
    - voxel_size: 体素的尺寸。

    返回:
    - geometry: 可用于open3d.visualization.Visualizer.add_geometry()的几何对象。
    """
    # 将体素坐标转换为点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_coords)

    # 使用点云创建表面（例如，通过Alpha Shape重建）
    # 注意：alpha参数控制表面的紧致程度，可能需要根据数据集进行调整
    alpha = voxel_size * 5
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # 计算顶点法向量以改善渲染效果
    mesh.compute_vertex_normals()

    # 可以选择着色以改善可视化效果
    mesh.paint_uniform_color([0.1, 0.7, 0.7])  # 赋予统一颜色

    return mesh


def visualize_dilated_voxels(voxel_coords):
    """
    将膨胀后的体素坐标可视化为点云。

    参数:
    - voxel_coords: 膨胀后的体素空间坐标数组，形状为(N, 3)。

    返回:
    - point_cloud: 一个open3d.geometry.PointCloud对象，包含膨胀后的体素坐标。
    """
    # 创建一个点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 将膨胀后的体素坐标赋值给点云
    point_cloud.points = o3d.utility.Vector3dVector(voxel_coords)

    # 可以根据需要设置点云的颜色，这里我们设置为蓝色
    point_cloud.paint_uniform_color([0, 0, 1])  # 蓝色

    return point_cloud


def generate_voxel_dilate_surface(voxel_grid, expan_dis, limit_z):
    voxels = np.asarray(voxel_grid.get_voxels())
    voxel_coords = np.asarray([voxel.grid_index for voxel in voxels])
    voxel_size = voxel_grid.voxel_size

    dilate_iter = math.ceil(expan_dis / voxel_size)
    expan_coord = dilate_iter * 2

    max_index = np.max(voxel_coords, axis=0) + 1 + expan_coord
    expan_space = np.zeros(max_index)

    expan_coord_half = expan_coord // 2
    voxel_coords += expan_coord_half
    expan_space[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

    # 创建球形结构元素
    se = ndimage.generate_binary_structure(3, 1)

    # 进行膨胀操作
    dilated_space = ndimage.binary_dilation(expan_space, iterations=dilate_iter, structure=se)

    # 进行腐蚀操作，使用与膨胀相同的结构元素和迭代次数
    eroded_space = ndimage.binary_erosion(dilated_space, iterations=dilate_iter, structure=se)

    # 使用logical_xor找出只在膨胀层而不在腐蚀层的体素
    surface_space = np.logical_xor(dilated_space, eroded_space)

    # 提取最外层体素坐标
    surface_indices = np.argwhere(surface_space)
    surface_coords = (surface_indices - expan_coord_half) * voxel_size + voxel_grid.get_voxel_center_coordinate(
        voxel_coords[0] - expan_coord_half)

    # 过滤低于limit_z的体素
    filtered_coords = surface_coords[surface_coords[:, 2] >= limit_z]

    return filtered_coords


def generate_voxel_dilate_xyz1(voxel_grid, expan_dis, limit_z):
    # 获取体素中心坐标
    voxels = np.asarray(voxel_grid.get_voxels())
    voxel_coords = np.asarray([voxel.grid_index for voxel in voxels])

    voxel_size = voxel_grid.voxel_size
    # 由于只需要膨胀一次，我们将直接计算扩展距离对应的体素数量
    expan_coord = math.ceil(expan_dis / voxel_size)

    # 创建扩展空间的数组
    # 计算数组的新边界
    max_coord = np.max(voxel_coords, axis=0) + expan_coord
    min_coord = np.min(voxel_coords, axis=0) - expan_coord
    dims = max_coord - min_coord + 1

    # 创建用于扩张的空间数组
    expan_space = np.zeros(dims)
    offset = -min_coord

    for voxel in voxel_coords:
        expan_space[tuple(voxel + offset)] = 1  # 标记原始体素位置

    # 生成球形结构元素用于膨胀操作
    se = ndimage.generate_binary_structure(3, 1)  # 球形结构
    expan_space = ndimage.binary_dilation(expan_space, structure=se).astype(expan_space.dtype)

    # 找出膨胀后新增的体素坐标
    expanded_coords = np.argwhere(expan_space) - offset

    # 转换为世界坐标系下的位置
    expanded_voxels_xyz = expanded_coords * voxel_size + voxel_grid.origin

    # 过滤掉z坐标小于limit_z的体素
    filtered_voxels_xyz = expanded_voxels_xyz[expanded_voxels_xyz[:, 2] >= limit_z]

    return filtered_voxels_xyz


def generate_voxel_dilate_xyz2(voxel_grid, expan_dis, limit_z):
    voxels = np.asarray(voxel_grid.get_voxels())
    voxel_coords = np.asarray([voxel.grid_index for voxel in voxels])
    voxel_size = voxel_grid.voxel_size
    dilate_iter = math.ceil(expan_dis / voxel_size)  # 用于确定膨胀的距离

    # 计算膨胀的空间尺寸，考虑到膨胀的范围
    expan_coord = dilate_iter * 2

    # 初始化扩展空间数组，准备进行膨胀操作
    expan_space = np.zeros((np.max(voxel_coords, axis=0) + expan_coord))
    for coord in voxel_coords:
        expan_space[tuple(coord + dilate_iter)] = 1  # 基于原始体素位置进行初始化

    # 生成球形结构元素，进行一次膨胀操作
    se = ndimage.generate_binary_structure(3, 1)  # 生成球形结构元素
    expan_space = ndimage.binary_dilation(expan_space, structure=se, iterations=dilate_iter)

    # 计算膨胀后体素的坐标
    expan_indices = np.argwhere(expan_space > 0)
    voxels_xyz = np.empty((len(expan_indices), 3))
    for i, idx in enumerate(expan_indices):
        voxels_xyz[i] = voxel_grid.origin + idx * voxel_size - (dilate_iter * voxel_size)

    # 过滤掉z坐标小于limit_z的体素
    voxels_xyz = voxels_xyz[voxels_xyz[:, 2] >= limit_z]

    return voxels_xyz


def voxel_expansion(voxel_grid, expan_dis):
    """
    对体素网格进行球体结构的向外膨胀，并返回膨胀后的体素坐标。

    参数:
    - voxel_grid: 输入的体素网格，为open3d.geometry.VoxelGrid对象。
    - expan_dis: 膨胀距离，表示向外膨胀的距离。

    返回:
    - expanded_voxels: 一个包含膨胀后体素中心坐标的列表。
    """
    # 获取原始体素的中心坐标
    original_voxels = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    voxel_size = voxel_grid.voxel_size
    grid_indices = set()

    # 计算膨胀的体素范围（以体素为单位）
    range_voxels = int(np.ceil(expan_dis / voxel_size))

    # 遍历每个体素，计算膨胀范围内的所有体素
    for voxel in original_voxels:
        for x in range(-range_voxels, range_voxels + 1):
            for y in range(-range_voxels, range_voxels + 1):
                for z in range(-range_voxels, range_voxels + 1):
                    # 计算新体素的索引
                    new_voxel_index = (voxel[0] + x, voxel[1] + y, voxel[2] + z)
                    # 如果新体素在球体膨胀范围内，则添加到结果中
                    if np.linalg.norm(np.array(new_voxel_index) - np.array(voxel)) * voxel_size <= expan_dis:
                        grid_indices.add(new_voxel_index)

    # 将索引转换为实际的体素中心坐标
    expanded_voxels = [np.array(index) * voxel_size + voxel_grid.origin for index in grid_indices]

    return expanded_voxels


def offset_mesh_surface(mesh, offset_distance):
    """
    将网格模型的表面向外偏移特定距离。

    参数:
    - mesh: 输入的网格模型，为open3d.geometry.TriangleMesh对象。
    - offset_distance: 向外偏移的距离（单位与模型相同，例如毫米）。

    返回:
    - offset_mesh: 偏移后的网格模型。
    """
    # 确保网格法向量是更新的
    mesh.compute_vertex_normals()

    # 获取原始顶点位置
    vertices = np.asarray(mesh.vertices)

    # 获取顶点法向量
    normals = np.asarray(mesh.vertex_normals)

    # 计算偏移后的顶点位置
    offset_vertices = vertices + normals * offset_distance

    # 创建一个新的网格模型，使用偏移后的顶点和原始的面
    offset_mesh = o3d.geometry.TriangleMesh()
    offset_mesh.vertices = o3d.utility.Vector3dVector(offset_vertices)
    offset_mesh.triangles = mesh.triangles
    offset_mesh.compute_vertex_normals()  # 重新计算法向量

    return offset_mesh


