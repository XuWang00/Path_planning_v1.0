import open3d as o3d
import numpy as np





def ray_triangle_intersection(ray_origin, ray_direction, triangle_vertices):
    """
    使用Möller–Trumbore算法检测光线与三角形的相交，并计算交点。

    参数:
    - ray_origin: 光线的起点坐标，形如 [x, y, z]。
    - ray_direction: 光线的方向向量，形如 [x, y, z]。
    - triangle_vertices: 三角形的顶点坐标，形如 [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]。

    返回:
    - intersection_point: 如果相交，则返回交点坐标，形如 [x, y, z]，否则返回 None。
    """
    # 将三角形的顶点坐标分别提取出来
    vertex1, vertex2, vertex3 = triangle_vertices

    # 计算边向量
    edge1 = vertex2 - vertex1
    edge2 = vertex3 - vertex1
    print(f"Edge1: {edge1}")
    print(f"Edge2: {edge2}")

    # 检查输入数组的维度
    print(f"Ray Direction Shape: {ray_direction.shape}")
    print(f"Edge2 Shape: {edge2.shape}")

    # 计算表面法向量
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)

    # 如果a接近于0，则射线与三角形平行或共面
    # if a > -1e-6 and a < 1e-6:
    #     return None
    if a < 1e-6:
        return None


    f = 1.0 / a
    s = ray_origin - vertex1
    u = f * np.dot(s, h)

    # 射线是否在三角形的平面内
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)

    # 射线是否在三角形的平面内
    if v < 0.0 or u + v > 1.0:
        return None

    # 计算射线与三角形的交点
    t = f * np.dot(edge2, q)
    if t > 1e-6:  # 确保交点在射线正方向上
        # intersection_point = ray_origin + t * ray_direction
        # return intersection_point
        return 1

    return None




def process_viewpoints(viewpoints, highlighted_mesh, objmesh):
    """
    遍历视点列表中的每个视点，并打印计算结果。

    参数:
    - viewpoints: 包含视点、方向和视图分类的列表。每个元素是(point, direction, view)的元组。
    - highlighted_mesh: 目标区域的mesh集合。
    - objmesh。

    返回:
    无返回值。
    """
    print("遍历视点列表中的每个视点，打印计算结果:")
    # for viewpoint in viewpoints:
    #     point, _, _ = viewpoint
    #     coun = count_visible_faces(viewpoint, highlighted_mesh, objmesh)
    #     print(f"Viewpoint at {point}: Count = {count}")
    # 只处理第一个视点
    viewpoint = viewpoints[0]
    point, _, _ = viewpoint
    count = count_visible_faces(viewpoint, highlighted_mesh, objmesh)
    print(f"Viewpoint at {point}: Count = {count}")


def count_visible_faces(viewpoint, highlighted_mesh, objmesh):
    """
    计算视点可见的目标区域内的面片个数和面积。

    参数:
    - viewpoint: 视点，形如 (point coordinates, view direction, view category)。
    - highlighted_mesh: 目标区域的mesh。
    - objmesh: 整个模型的mesh。

    返回:
    - count: 可见的面片个数。
    - visible_area: 可见的面片总面积。
    """
    # 解包视点信息
    ray_origin, _ , _ = viewpoint

    count = 0
    visible_area = 0.0

    # 遍历目标区域内的每个面片
    for face_indices in highlighted_mesh.triangles:
        i = 0
        # 获取当前面片的顶点坐标索引
        vertex_indices = np.asarray(face_indices)
        # 将顶点索引转换为 numpy 数组
        vertex_indices = np.array(vertex_indices)
        # 从highlighted_mesh中获取相应的顶点坐标
        triangle_vertices = np.array(highlighted_mesh.vertices)[vertex_indices]
        # 计算面片中心点坐标
        center = np.mean(triangle_vertices, axis=0)
        # 计算视点指向面片中心的射线方向
        ray_direction = center - ray_origin
        ray_direction /= np.linalg.norm(ray_direction)  # 将射线方向向量归一化
        print(f"Ray Origin: {ray_origin}")
        print(f"Ray Direction: {ray_direction}")
        print(f"Triangle Vertices: {triangle_vertices}")

        for face_indices1 in objmesh.triangles:
            # 获取当前面片的顶点坐标索引
            vertex_indices1 = np.asarray(face_indices1)
            # 将顶点索引转换为 numpy 数组
            vertex_indices1 = np.array(vertex_indices1)
            # 从highlighted_mesh中获取相应的顶点坐标
            triangle_vertices1 = np.array(highlighted_mesh.vertices)[vertex_indices1]
            # 检查射线与整个模型的面片是否相交
            intersection_point = ray_triangle_intersection(ray_origin, ray_direction, triangle_vertices1)
            if intersection_point == 1:
                i += 1

        if i == 1:
            # 如果没有相交点，则面片可见，更新计数和面积
            count += 1
            # # 计算面片的面积并累加到总面积中
            # a = vertices[0]
            # b = vertices[1]
            # c = vertices[2]
            # ab = b - a
            # ac = c - a
            # area = 0.5 * np.linalg.norm(np.cross(ab, ac))
            # visible_area += area

    return count


def compute_triangle_area(vertices):
    """
    计算三角形的面积。

    参数:
    - vertices: 三角形的顶点坐标。

    返回:
    - area: 三角形的面积。
    """
    # 使用海伦公式计算三角形面积
    a = np.linalg.norm(vertices[0] - vertices[1])
    b = np.linalg.norm(vertices[1] - vertices[2])
    c = np.linalg.norm(vertices[2] - vertices[0])
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

