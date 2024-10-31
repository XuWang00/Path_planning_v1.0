import open3d as o3d
import numpy as np

def load_obj(model_path):
    mesh = o3d.io.read_triangle_mesh(model_path)
    print("Successfully loaded model.")
    # Ensure the model is correct (not empty)
    if mesh.is_empty():
        raise ValueError("Model is empty. Please check the file path and file integrity.")
    # Compute vertex normals
    mesh.compute_vertex_normals()

    return mesh

def load_obj_t(model_path):
    mesh = o3d.t.io.read_triangle_mesh(model_path)
    print("Successfully loaded model.")
    # Ensure the model is correct (not empty)
    if mesh.is_empty():
        raise ValueError("Model is empty. Please check the file path and file integrity.")
    # Compute vertex normals
    mesh.compute_vertex_normals()

    return mesh

def create_wireframe(mesh):
    # 提取三角形顶点索引
    triangles = np.asarray(mesh.triangles)
    # 提取所有可能的边
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]))
    # 移除重复的边
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    # 创建线集(LineSet)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        lines=o3d.utility.Vector2iVector(edges)
    )
    return line_set

def load_obj_and_create_wireframe(model_path):
    # 读取模型
    mesh = o3d.io.read_triangle_mesh(model_path)
    print("Successfully loaded model.")
    if mesh.is_empty():
        raise ValueError("Model is empty. Please check the file path and file integrity.")

    # 计算顶点法向量
    mesh.compute_vertex_normals()

    # 提取三角形顶点索引
    triangles = np.asarray(mesh.triangles)
    # 提取所有可能的边
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]))
    # 移除重复的边
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    # 创建线集(LineSet)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        lines=o3d.utility.Vector2iVector(edges)
    )

    return mesh, line_set

# def adjust_model_position(mesh):
#     # Get the model's axis-aligned bounding box
#     bbox = mesh.get_axis_aligned_bounding_box()
#     # Calculate the distance needed to move up so that the model bottom aligns with the ground plane
#     translation = -bbox.min_bound
#     # translation[2] = 0  # Assuming the Z-axis is vertical, no need to adjust the Z-axis
#     # Move the model up
#     mesh.translate(translation)
#
#     return mesh
def adjust_model_position(mesh):
    # 获取模型的轴对齐边界框
    bbox = mesh.get_axis_aligned_bounding_box()

    # 计算将边界框底面中心移动到原点所需的平移量
    # 边界框底面中心的x和y坐标是x_min + (x_max - x_min)/2 和 y_min + (y_max - y_min)/2
    bottom_center = np.array([
        (bbox.min_bound[0] + bbox.max_bound[0]) / 2,
        (bbox.min_bound[1] + bbox.max_bound[1]) / 2,
        bbox.min_bound[2]  # z坐标保持底面即可
    ])

    # 计算从当前位置到原点的平移向量
    translation = -bottom_center
    # 平移模型
    mesh.translate(translation)
    print(translation)
    return mesh

def adjust_model_position1(mesh, new_position):
    # 获取模型的轴对齐边界框
    bbox = mesh.get_axis_aligned_bounding_box()

    # 计算边界框底面中心的坐标
    bottom_center = np.array([
        (bbox.min_bound[0] + bbox.max_bound[0]) / 2,
        (bbox.min_bound[1] + bbox.max_bound[1]) / 2,
        bbox.min_bound[2]  # 底面的z坐标
    ])

    # 计算从底面中心到新位置的平移向量
    translation = np.array(new_position) - bottom_center

    # 平移模型
    mesh.translate(translation)

    return mesh


def rotate_model(mesh, rotation_angles):
    """
    Rotate the model to adjust its posture.
    Parameters:
    - mesh: The model to rotate (o3d.geometry.TriangleMesh object).
    - rotation_angles: A list or numpy array containing three elements, representing the angles of rotation around the X, Y, Z axis respectively (unit: radians).
    """
    # Generate rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)
    # Rotate the model
    mesh.rotate(R, center=(0, 0, 0))  # The rotation center point 'center' can be adjusted to the model's center or another point
    return mesh


def visualize_mesh_with_edges(model_path):
    # 加载模型
    mesh = o3d.io.read_triangle_mesh(model_path)

    # 检查模型是否包含法线，如果没有则计算
    if not mesh.has_vertex_normals():
        print("Recomputing vertex normals...")
        mesh.compute_vertex_normals()

    # 可视化设置：显示表面网格
    mesh.paint_uniform_color([0.9, 0.7, 0.7])  # 给模型上色以增强可视化效果，颜色可自定义

    return mesh


def visualize_mesh_normals(mesh):
    # 首先，计算三角面片的法向量
    mesh.compute_triangle_normals()
    # 获取三角面片的中心点
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_centers = vertices[triangles].mean(axis=1)
    # 获取法向量
    normals = np.asarray(mesh.triangle_normals)

    # 创建一个LineSet对象来可视化法向量
    lines = []
    line_colors = []
    for i in range(len(triangle_centers)):
        # 为每个法向量添加一个线段，从三角形的中心指向法向量的方向
        lines.append([i, i + len(triangle_centers)])  # 线段的索引
        line_colors.append([1, 0, 0])  # 红色

    # 法向量的终点是中心点加上法向量自身
    normals_end = triangle_centers + normals * 10  # 乘以一个因子以调整显示的长度
    all_points = np.vstack((triangle_centers, normals_end))  # 合并所有点

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    return line_set