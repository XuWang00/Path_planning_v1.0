import numpy as np
import open3d as o3d
import math




def generate_aabb_grid(mesh, cuboid_size):
    """
    在模型的 AABB 内生成规定尺寸的立方体 AABB 网格。

    参数:
    - mesh: Open3D TriangleMesh 对象。
    - cuboid_size: 每个立方体区块的边长。

    返回:
    - aabb_cuboids: 用于可视化的 AABB 区块对象列表。
    """
    # 计算模型的 AABB 包围盒
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = np.array(aabb.min_bound)
    max_bound = np.array(aabb.max_bound)

    # 计算在每个维度上需要多少个立方体来覆盖整个 AABB
    num_cuboids_x = int(np.ceil((max_bound[0] - min_bound[0]) / cuboid_size))
    num_cuboids_y = int(np.ceil((max_bound[1] - min_bound[1]) / cuboid_size))
    num_cuboids_z = int(np.ceil((max_bound[2] - min_bound[2]) / cuboid_size))

    aabb_cuboids = []

    # 生成立方体网格
    for i in range(num_cuboids_x):
        for j in range(num_cuboids_y):
            for k in range(num_cuboids_z):
                # 计算当前立方体的最小边界点
                min_corner = min_bound + np.array([i * cuboid_size, j * cuboid_size, k * cuboid_size])
                # 创建小立方体的 AABB
                cuboid_aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, min_corner + np.array(
                    [cuboid_size, cuboid_size, cuboid_size]))
                aabb_cuboids.append(cuboid_aabb)

    return aabb_cuboids



def generate_aabb_grid_centered(mesh, cuboid_size):
    """
    以模型的中心为起点，向四周生成规定尺寸的立方体 AABB 网格。

    参数:
    - mesh: Open3D TriangleMesh 对象。
    - cuboid_size: 每个立方体区块的边长。

    返回:
    - aabb_cuboids: 用于可视化的 AABB 区块对象列表。
    """
    # 计算模型的 AABB 包围盒并获取中心点
    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()

    # 从中心向外扩展的立方体数量（向下取整以保持对称）
    half_num_cuboids_x = int((aabb.get_max_extent() / 2) / cuboid_size)
    half_num_cuboids_y = half_num_cuboids_x
    half_num_cuboids_z = half_num_cuboids_x

    aabb_cuboids = []

    # 以中心为起点向四周生成立方体
    for i in range(-half_num_cuboids_x, half_num_cuboids_x + 1):
        for j in range(-half_num_cuboids_y, half_num_cuboids_y + 1):
            for k in range(-half_num_cuboids_z, half_num_cuboids_z + 1):
                # 计算当前立方体的中心点
                cuboid_center = center + np.array([i * cuboid_size, j * cuboid_size, k * cuboid_size])
                # 根据中心点计算立方体的最小和最大边界点
                min_corner = cuboid_center - np.array([cuboid_size / 2, cuboid_size / 2, cuboid_size / 2])
                max_corner = cuboid_center + np.array([cuboid_size / 2, cuboid_size / 2, cuboid_size / 2])
                # 创建小立方体的 AABB
                cuboid_aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
                aabb_cuboids.append(cuboid_aabb)

    return aabb_cuboids



def generate_aabb_grid_base_centered(mesh, cuboid_size):
    """
    以模型的底面中心为起点，向外和向上生成规定尺寸的立方体 AABB 网格。
    保证最下层小立方体的底面与z=0平面重合。

    参数:
    - mesh: Open3D TriangleMesh 对象。
    - cuboid_size: 每个立方体区块的边长。

    返回:
    - aabb_cuboids: 用于可视化的 AABB 区块对象列表。
    """
    # 计算模型的 AABB 包围盒
    aabb = mesh.get_axis_aligned_bounding_box()
    base_center = np.array([(aabb.min_bound[0] + aabb.max_bound[0]) / 2,
                            (aabb.min_bound[1] + aabb.max_bound[1]) / 2, 0])  # 底面中心位于z=0平面

    # 计算在每个维度上需要多少个立方体来覆盖整个AABB
    num_cuboids_x = int(np.ceil((aabb.max_bound[0] - aabb.min_bound[0]) / cuboid_size))
    num_cuboids_y = int(np.ceil((aabb.max_bound[1] - aabb.min_bound[1]) / cuboid_size))
    num_cuboids_z = int(np.ceil((aabb.max_bound[2] - 0) / cuboid_size))  # 从z=0向上扩展

    aabb_cuboids = []

    # 从底面中心开始向外和向上生成立方体
    for i in range(-num_cuboids_x // 2, num_cuboids_x // 2 + 1):
        for j in range(-num_cuboids_y // 2, num_cuboids_y // 2 + 1):
            for k in range(num_cuboids_z):
                cuboid_center = base_center + np.array([i * cuboid_size, j * cuboid_size, k * cuboid_size + cuboid_size / 2])
                min_corner = cuboid_center - np.array([cuboid_size / 2, cuboid_size / 2, cuboid_size / 2])
                max_corner = cuboid_center + np.array([cuboid_size / 2, cuboid_size / 2, cuboid_size / 2])
                cuboid_aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
                aabb_cuboids.append(cuboid_aabb)

    return aabb_cuboids


def filter_cuboids_containing_surface(mesh, cuboids):
    """
    筛选出包含物体表面顶点的立方体区块。

    参数:
    - mesh: Open3D TriangleMesh 对象，代表物体模型。
    - cuboids: 一系列 AxisAlignedBoundingBox 对象的列表，代表之前生成的立方体空间区块。

    返回:
    - surface_cuboids: 包含至少一个模型表面顶点的立方体区块列表。
    """
    # 获取模型的顶点数组
    vertices = np.asarray(mesh.vertices)

    surface_cuboids = []

    for cuboid in cuboids:
        min_bound = np.asarray(cuboid.min_bound)
        max_bound = np.asarray(cuboid.max_bound)

        # 检查每个顶点是否在当前立方体区块内
        for vertex in vertices:
            if np.all(vertex >= min_bound) and np.all(vertex <= max_bound):
                # 如果找到至少一个顶点在区块内，添加到结果列表中，并跳出当前循环
                surface_cuboids.append(cuboid)
                break

    return surface_cuboids




def visualize_average_normals(mesh, cuboids):
    """
    计算并可视化每个小立方体空间内包含的表面的平均法向量。

    参数:
    - mesh: Open3D TriangleMesh 对象，代表物体模型。
    - cuboids: 包含物体表面顶点的小立方体空间列表，由 AxisAlignedBoundingBox 对象组成。

    返回:
    - normal_lines: 表示平均法向量的线段集合，用于可视化。
    """
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    line_points = []  # 存储所有线段的起点和终点
    lines = []  # 线段的索引
    colors = []  # 每条线段的颜色

    for cuboid in cuboids:
        # 找到落在当前cuboid内的顶点索引
        indices = [i for i, v in enumerate(vertices) if np.all(v >= cuboid.min_bound) and np.all(v <= cuboid.max_bound)]
        if not indices:
            continue  # 如果这个cuboid内没有顶点，则跳过

        # 计算这些顶点法向量的平均值
        avg_normal = normals[indices].mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal)  # 归一化法向量

        cuboid_center = (cuboid.min_bound + cuboid.max_bound) / 2.0  # cuboid的中心点
        start_point = cuboid_center
        end_point = cuboid_center + avg_normal * 10  # 将平均法向量延伸以便可视化

        line_points.append(start_point)
        line_points.append(end_point)
        lines.append([len(line_points) - 2, len(line_points) - 1])  # 添加线段的起点和终点索引
        colors.append([1, 0, 0])  # 红色表示法向量

    # 创建LineSet对象来可视化法向量
    normal_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    normal_lines.colors = o3d.utility.Vector3dVector(colors)  # 设置线段颜色为红色

    return normal_lines


def get_cuboids_centers(aabb_cuboids):
    """
    计算并返回每个轴对齐边界框（AABB）立方体的中心点坐标及其编号。

    参数:
    - aabb_cuboids: 一系列 AxisAlignedBoundingBox 对象的列表。

    返回:
    - centers_with_ids: 每个 AABB 立方体的中心点坐标及其编号的列表。每个元素是(center, id)的元组。
    """
    centers_with_ids = []
    for idx, cuboid in enumerate(aabb_cuboids):
        min_bound = np.asarray(cuboid.min_bound)
        max_bound = np.asarray(cuboid.max_bound)
        # 计算中心点坐标
        center = (min_bound + max_bound) / 2.0
        centers_with_ids.append((center, idx))  # 将中心点和编号作为元组添加到列表中

    return centers_with_ids


def highlight_cuboid_frame(cuboids, index):
    """
    创建并返回指定立方体的红色线框模型。

    参数:
    - cuboids: 一系列 AxisAlignedBoundingBox 对象的列表。
    - index: 要高亮显示的立方体的索引。

    返回:
    - line_set: 表示高亮立方体边框的线集对象。
    """
    # 获取指定立方体的AABB
    cuboid = cuboids[index]

    # 计算立方体的8个顶点
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                point = cuboid.get_center() + np.array([i, j, k]) * (cuboid.get_extent() / 2)
                points.append(point)
    points = np.array(points)

    # 定义立方体的12条边
    lines = [[0,1], [1,3], [3,2], [2,0],
             [4,5], [5,7], [7,6], [6,4],
             [0,4], [1,5], [3,7], [2,6]]

    # 创建LineSet对象
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    # 设置线框颜色
    colors = [[0, 0, 1] for i in range(len(lines))]  # 红色
    line_set.colors = o3d.utility.Vector3dVector(colors)
    print(f"will hightlight the cuboid{index}")
    return line_set


def highlight_cuboid_vertices(mesh, cuboids, idx_to_highlight):
    # 创建一个新的TriangleMesh实例
    highlighted_mesh = o3d.geometry.TriangleMesh()
    complement_mesh = o3d.geometry.TriangleMesh()

    # 复制顶点和面片数据
    highlighted_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    highlighted_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    highlighted_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))

    # complement_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # complement_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    # complement_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))

    # 需要确保原始网格有顶点颜色数据，如果没有，初始化为白色
    if len(mesh.vertex_colors) == 0:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh.vertices), 3)))
    highlighted_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))

    # 获取需要高亮的区块的AABB
    target_cuboid = cuboids[idx_to_highlight]

    # 遍历所有顶点，如果顶点在目标区块内，则将颜色设置为红色
    for i, vertex in enumerate(highlighted_mesh.vertices):
        if (np.asarray(vertex) >= target_cuboid.get_min_bound()).all() and \
                (np.asarray(vertex) <= target_cuboid.get_max_bound()).all():
            highlighted_mesh.vertex_colors[i] = [1, 0, 0]
        # else:
        #     complement_mesh.vertex_colors[i] = [0, 0, 1]
    print("ok")
    return highlighted_mesh


def get_cuboid_center(aabb_cuboids, idx):
    """
    计算并返回每个轴对齐边界框（AABB）立方体的中心点坐标及其编号。

    参数:
    - aabb_cuboids: 一系列 AxisAlignedBoundingBox 对象的列表。

    返回:
    - centers_with_ids: 每个 AABB 立方体的中心点坐标及其编号的列表。每个元素是(center, id)的元组。
    """
    if idx < 0 or idx >= len(aabb_cuboids):
        raise IndexError("Idx is out of bounds.")

    cuboid = aabb_cuboids[idx]
    min_bound = np.asarray(cuboid.min_bound)
    max_bound = np.asarray(cuboid.max_bound)
    # 计算中心点坐标
    center = (min_bound + max_bound) / 2.0

    return center


def filter_triangles_by_cuboid(mesh, aabb_cuboids, cuboid_idx):
    """
    根据指定的立方体索引，筛选出处于该立方体内的所有三角面片的索引。

    参数:
    - mesh: Open3D TriangleMesh 对象，包含顶点和三角面片。
    - aabb_cuboids: 包含多个立方体的列表。
    - cuboid_idx: 需要筛选的立方体的索引。

    返回:
    - triangle_indices: 在指定立方体内的三角面片索引列表。
    """
    # 获取指定索引的立方体
    cuboid = aabb_cuboids[cuboid_idx]

    # 获取模型的顶点和三角面片
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 存储处于立方体内部的三角面片的索引
    triangle_indices = []

    # 获取立方体的最小和最大边界
    min_bound = cuboid.get_min_bound()
    max_bound = cuboid.get_max_bound()

    # 检查每个三角面片
    for idx, triangle in enumerate(triangles):
        # 获取三角面片的三个顶点
        p1, p2, p3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        # 检查三角面片的每个顶点是否都在立方体内
        if (all(min_bound <= p1) and all(p1 <= max_bound) and
                all(min_bound <= p2) and all(p2 <= max_bound) and
                all(min_bound <= p3) and all(p3 <= max_bound)):
            triangle_indices.append(idx)

    return triangle_indices


def calculate_total_area_of_triangles(mesh, triangle_indices):
    """
    计算特定三角面片索引列表中所有三角面片的面积总和。

    参数:
    - mesh: Open3D TriangleMesh 对象，包含顶点和三角面片。
    - triangle_indices: 三角面片索引的列表。

    返回:
    - total_area: 指定三角面片的面积总和。
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    total_area = 0.0

    # 计算每个指定三角面片的面积
    for idx in triangle_indices:
        # 获取三角面片的顶点索引
        tri = triangles[idx]
        # 获取顶点
        p1, p2, p3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        # 计算向量
        v1, v2 = p2 - p1, p3 - p1
        # 叉积的模长给出平行四边形的面积，三角形面积是它的一半
        triangle_area = np.linalg.norm(np.cross(v1, v2)) / 2
        total_area += triangle_area

    return total_area


def create_custom_aabb(center, size):
    """
    创建一个自定义的轴对齐包围盒 (AABB)。

    参数:
    - center: 立方体中心的坐标，格式为 [x, y, z]。
    - size: 立方体的边长。

    返回:
    - aabb: 创建的 AxisAlignedBoundingBox 对象。
    """
    half_size = np.array([size, size, size]) / 2
    min_bound = np.array(center) - half_size
    max_bound = np.array(center) + half_size
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    return aabb
