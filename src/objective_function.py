import numpy as np
import open3d as o3d


def normalize_and_sum_angles(mesh, angles_list):
    # 提取所有三角形面片的顶点索引
    triangles = np.asarray(mesh.triangles)
    # 提取所有顶点坐标
    vertices = np.asarray(mesh.vertices)

    # 初始化求和变量
    angle_weighted_sum = 0.0
    angle_normalized_sum = 0.0
    total_area_of_hit_triangles = 0.0
    # 处理每一个角度和索引
    for index, angle in angles_list:
        # 计算三角形面积
        triangle = triangles[index]
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        total_area_of_hit_triangles += area
        if angle > 90.0:
            normalized_angle = 0
        else:
            # 归一化角度 (0度->1, 90度->0)
            normalized_angle = 1 - (angle / 90.0)*2
            # # 加权求和
            angle_weighted_sum += normalized_angle * area
            # # 不加权求和检验
            # angle_normalized_sum += normalized_angle


    return angle_weighted_sum, total_area_of_hit_triangles


def normalize_and_sum_angles_optimized(mesh, angles_list):
    # 提取所有三角形的顶点索引和顶点坐标
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # 提取角度和索引
    indices, angles = zip(*angles_list)
    indices = np.array(indices)
    angles = np.array(angles)

    # 计算三角形顶点
    v1, v2, v3 = vertices[triangles[indices, 0]], vertices[triangles[indices, 1]], vertices[triangles[indices, 2]]

    # 计算三角形面积
    areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

    # 归一化角度
    normalized_angles = np.clip(1 - (angles / 90.0) * 2, 0, None)

    # 计算加权和
    angle_weighted_sum = np.sum(normalized_angles * areas)

    # 计算总面积
    total_area_of_hit_triangles = np.sum(areas)

    return angle_weighted_sum, total_area_of_hit_triangles


def normalize_and_sum_distances(mesh, distance_list):
    # 提取所有三角形面片的顶点索引
    triangles = np.asarray(mesh.triangles)
    # 提取所有顶点坐标
    vertices = np.asarray(mesh.vertices)

    # 初始化求和变量
    distance_weighted_sum = 0.0
    distance_normalized_sum = 0.0

    # 处理每一个距离和索引
    for index, distance in distance_list:
        # 计算三角形面积
        triangle = triangles[index]
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        # 距离调整
        adjusted_distance = abs(distance - 480)
        # 归一化调整距离 (0mm -> 1, 65mm -> 0)
        if adjusted_distance > 90:
            normalized_distance = 0
        else:
            normalized_distance = 1 - (adjusted_distance / 90.0)*2
        # # 加权求和
        distance_weighted_sum += normalized_distance * area
        # 不加权求和检验
        # distance_normalized_sum += normalized_distance


    return distance_weighted_sum


def normalize_and_sum_distances_optimized(mesh, distance_list):
    # 提取所有三角形的顶点索引和顶点坐标
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # 提取距离和索引
    indices, distances = zip(*distance_list)
    indices = np.array(indices)
    distances = np.array(distances)

    # 计算三角形顶点
    v1, v2, v3 = vertices[triangles[indices, 0]], vertices[triangles[indices, 1]], vertices[triangles[indices, 2]]

    # 计算三角形面积
    areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

    # 距离调整
    adjusted_distances = np.abs(distances - 480)

    # 归一化距离
    normalized_distances = np.clip(1 - (adjusted_distances / 90.0) * 2, 0, None)

    # 计算加权和
    distance_weighted_sum = np.sum(normalized_distances * areas)

    return distance_weighted_sum


def calculate_total_area_of_hit_triangles(mesh, hit_triangle_indices):
    # 确保提供的mesh是一个Open3D的TriangleMesh对象
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("The mesh must be an Open3D TriangleMesh object.")

    # 获取三角形网格的顶点坐标
    vertices = np.asarray(mesh.vertices)

    # 初始化面积总和
    total_area = 0.0

    # 遍历所有击中的面片索引，计算每个面片的面积
    for idx in hit_triangle_indices:
        # 获取定义面片的三个顶点索引
        triangle = mesh.triangles[idx]
        # 获取顶点坐标
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        # 计算该面片的面积
        # 面片面积公式：0.5 * ||(v2-v1) x (v3-v1)||
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        # 累加到总面积
        total_area += area

    return total_area


def calculate_total_mesh_area(mesh):
    """
    计算给定三维网格的所有三角面片的总面积。

    参数:
    - mesh: Open3D的TriangleMesh对象。

    返回:
    - total_area: 所有三角面片的总面积。
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("The mesh must be an Open3D TriangleMesh object.")

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    total_area = 0.0

    for triangle in triangles:
        # 获取三个顶点的坐标
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        # 计算三角形的面积
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        total_area += area

    return total_area


def calculate_costfunction(mesh, total_mesh_area, angles_list, distance_list, hit_triangle_indices, a, b, e, f):
    """
    Calculate the value C using the formula:
    C = a * total_area_of_triangles + b * e * angle_weighted_sum + b * f * distance_weighted_sum

    Parameters:
    mesh (Mesh): The mesh object containing triangle data.
    angles_list (list): List of angle measures.
    distance_list (list): List of distance measures.
    hit_triangle_indices (set): Indices of triangles that have been hit.
    a (float): Constant term in the formula.
    b (float): Coefficient for the weighted sums in the formula.
    e (float): Coefficient for the angle weighted sum.
    f (float): Coefficient for the distance weighted sum.

    Returns:
    float: The calculated value of C.
    """
    if len(hit_triangle_indices) == 0:
        return 0

    # Normalize and sum angles and distances
    angle_weighted_sum, total_area_of_hit_triangles = normalize_and_sum_angles_optimized(mesh, angles_list)
    distance_weighted_sum = normalize_and_sum_distances_optimized(mesh, distance_list)

    # Calculate the total area of hit triangles
    # total_area_of_hit_triangles = calculate_total_area_of_hit_triangles(mesh, hit_triangle_indices)

    # Calculate the value C according to the provided formula
    # c = 1000*(a * total_area_of_hit_triangles/total_mesh_area + b * (e * angle_weighted_sum + f * distance_weighted_sum) / len(hit_triangle_indices))
    #c = 1000 * (a * total_area_of_hit_triangles / total_mesh_area + b * (e * angle_weighted_sum + f * distance_weighted_sum) / total_area_of_hit_triangles)
    c = a*total_area_of_hit_triangles + b * (e * angle_weighted_sum + f * distance_weighted_sum)   ##old costfunction,works well

    return c




