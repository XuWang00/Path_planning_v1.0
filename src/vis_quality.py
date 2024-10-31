import numpy as np
import open3d as o3d


#这里需要修改，最好是以两个camera计算而不是以eye_center计算
# def calculate_view_angles_and_distances(viewpoint, mesh, hit_triangle_indices):
#     # 提取所有三角形面片的顶点索引
#     triangles = np.asarray(mesh.triangles)
#     # 提取所有顶点坐标
#     vertices = np.asarray(mesh.vertices)
#     # 提取所有三角形的法线
#     triangle_normals = np.asarray(mesh.triangle_normals)
#
#     # 初始化返回列表
#     angles_list = []
#     distance_list = []
#
#     # 遍历每个击中的三角形索引
#     for index in hit_triangle_indices:
#         # 获取该三角形的顶点索引
#         triangle = triangles[index]
#         # 获取三个顶点的坐标
#         v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
#         # 计算三角形的中心
#         centroid = (v1 + v2 + v3) / 3
#         # 计算从三角形中心到视点的方向向量
#         direction_to_viewpoint = viewpoint - centroid
#         # 计算距离
#         distance = np.linalg.norm(direction_to_viewpoint)
#         # 单位化方向向量
#         direction_to_viewpoint_normalized = direction_to_viewpoint / np.linalg.norm(direction_to_viewpoint)
#         # 获取三角形的法线向量
#         normal = triangle_normals[index]
#         # 计算方向向量和法线向量的夹角余弦值
#         cos_angle = np.dot(direction_to_viewpoint_normalized, normal)
#         # 计算角度（弧度转换为度）
#         angle = np.arccos(cos_angle) * (180.0 / np.pi)
#         # 存储结果
#         angles_list.append((index, angle))
#         distance_list.append((index, distance))
#
#     return angles_list, distance_list


def calculate_view_angles_and_distances(viewpoint, apex, mesh, hit_triangle_indices):
    # 提取所有三角形面片的顶点索引
    triangles = np.asarray(mesh.triangles)
    # 提取所有顶点坐标
    vertices = np.asarray(mesh.vertices)
    # 提取所有三角形的法线
    triangle_normals = np.asarray(mesh.triangle_normals)

    # 初始化返回列表
    angles_list = []
    distance_list = []

    # 计算光轴方向
    optical_axis = apex - viewpoint
    optical_axis_normalized = optical_axis / np.linalg.norm(optical_axis)

    # 遍历每个击中的三角形索引
    for index in hit_triangle_indices:
        # 获取该三角形的顶点索引
        triangle = triangles[index]
        # 获取三个顶点的坐标
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        # 计算三角形的中心
        centroid = (v1 + v2 + v3) / 3
        # 计算从三角形中心到视点的方向向量
        direction_to_viewpoint = viewpoint - centroid

        # 计算投影距离
        projected_distance = np.dot(direction_to_viewpoint, optical_axis_normalized)
        effective_distance = abs(projected_distance)  # 转换到最佳对焦平面的距离

        # 单位化方向向量
        direction_to_viewpoint_normalized = direction_to_viewpoint / np.linalg.norm(direction_to_viewpoint)
        # 获取三角形的法线向量
        normal = triangle_normals[index]
        # 计算方向向量和法线向量的夹角余弦值
        cos_angle = np.dot(direction_to_viewpoint_normalized, normal)
        # 计算角度（弧度转换为度）
        angle = np.arccos(cos_angle) * (180.0 / np.pi)

        # 存储结果
        angles_list.append((index, angle))
        distance_list.append((index, effective_distance))

    return angles_list, distance_list


def calculate_view_angles_and_distances_optimized(viewpoint, apex, mesh, hit_triangle_indices):
    # 确保 hit_triangle_indices 是 numpy 数组
    if isinstance(hit_triangle_indices, set):
        hit_triangle_indices = np.array(list(hit_triangle_indices))
    elif isinstance(hit_triangle_indices, list):
        hit_triangle_indices = np.array(hit_triangle_indices)

    triangles = np.asarray(mesh.triangles)[hit_triangle_indices]
    vertices = np.asarray(mesh.vertices)
    triangle_normals = np.asarray(mesh.triangle_normals)[hit_triangle_indices]

    # 计算所有质心
    centroids = vertices[triangles].mean(axis=1)
    direction_to_viewpoint = viewpoint - centroids

    optical_axis = apex - viewpoint
    optical_axis_normalized = optical_axis / np.linalg.norm(optical_axis)

    projected_distance = np.dot(direction_to_viewpoint, optical_axis_normalized[:, None]).flatten()
    effective_distance = np.abs(projected_distance)  # 转换到最佳对焦平面的距离

    direction_to_viewpoint_normalized = direction_to_viewpoint / np.linalg.norm(direction_to_viewpoint, axis=1)[:, None]
    cos_angle = np.einsum('ij,ij->i', direction_to_viewpoint_normalized, triangle_normals)

    angles = np.arccos(cos_angle) * (180.0 / np.pi)

    return list(zip(hit_triangle_indices, angles)), list(zip(hit_triangle_indices, effective_distance))


def sum_of_angles(angles_list):
    total_angle = sum(angle for _, angle in angles_list)
    return total_angle

def adjusted_sum_of_distances(distance_list):
    adjustment = 480
    total_distance = sum((distance - adjustment) for _, distance in distance_list)
    return total_distance




def calculate_coverage_ratio(all_hits, objmesh):
    """
    计算选中面片的面积占总面片面积的比例。

    参数:
    - all_hits: 包含被选中的面片索引的集合。
    - objmesh: 包含所有面片的网格对象。

    返回:
    - coverage_ratio: 被选中面片的面积占总面片面积的比例。
    """
    # 计算所有三角形的面积
    vertices = np.asarray(objmesh.vertices)
    triangles = np.asarray(objmesh.triangles)
    total_area = 0
    selected_area = 0

    for i, triangle in enumerate(triangles):
        # 获取三角形顶点
        p1, p2, p3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        # 计算三角形面积 using the cross product area formula
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        total_area += area
        if i in all_hits:
            selected_area += area

    # 计算覆盖比例
    if total_area == 0:  # 防止除以零
        return 0
    coverage_ratio = selected_area / total_area
    return coverage_ratio


def normalize_and_get_scores(mesh, angles_list, distances_list):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    scores = []
    for (angle_index, angle), (distance_index, distance) in zip(angles_list, distances_list):
        if angle_index != distance_index:
            raise ValueError("Indices of angle and distance lists do not match.")

        triangle = triangles[angle_index]
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

        # Normalize angle and distance
        normalized_angle = 1 - (angle / 90.0) if angle <= 90 else 0
        adjusted_distance = abs(distance - 480)
        normalized_distance = 1 - (adjusted_distance / 90.0) if 0 <= adjusted_distance <= 95 else (1 if adjusted_distance < 0 else 0)

        # Calculate combined score
        score = (0.6 * normalized_angle + 0.4 * normalized_distance)
        scores.append((angle_index, score, area))

    return scores


def classify_and_summarize_scores(scores):
    classification_areas = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    classification_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    total_area = 0
    processed_triangles = set()  # 用来记录已经处理过的三角面片索引

    for index, score, area in scores:
        if index in processed_triangles:
            continue  # 如果这个面片已经处理过，就跳过以避免重复计算面积
        processed_triangles.add(index)  # 记录这个面片已被处理

        total_area += area
        if score >= 0.75:
            classification = 'A'
        elif score >= 0.5:
            classification = 'B'
        elif score >= 0.25:
            classification = 'C'
        else:
            classification = 'D'

        classification_areas[classification] += area
        classification_counts[classification] += 1

    # Output the results
    print("Scan Quality Classification:")
    for grade in classification_areas:
        area = classification_areas[grade]
        count = classification_counts[grade]
        area_percentage = (area / total_area) * 100 if total_area > 0 else 0
        count_percentage = (count / sum(classification_counts.values())) * 100
        print(
            f"{grade}-grade scan quality: {count} triangles ({count_percentage:.2f}%), {area:.2f} square units ({area_percentage:.2f}% of total)")

    return classification_areas, classification_counts
