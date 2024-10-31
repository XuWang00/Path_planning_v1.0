import open3d as o3d
import os
import numpy as np
import time
from geometry import *
from model_loader import *
from objective_function import *
from pynput import keyboard
import threading





def get_triangle_indices_from_picked_points(mesh, picked_points):
    # 提取顶点索引
    indices = [p.index for p in picked_points]

    triangle_indices = []
    # 假设每个面片由三个顶点组成
    for i in range(0, len(mesh.triangles), 3):
        triangle = mesh.triangles[i:i+3]
        # 检查这个三角面片的任意一个顶点是否被选中
        if any(vertex in indices for vertex in triangle):
            triangle_indices.append(i // 3)  # 将三角面片的索引加入列表
    return triangle_indices


def filter_triangles_by_bounding_box(picked_points, mesh):
    # 获取模型的顶点和三角面片
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # 计算包围盒
    coordinates = np.array([p.coord for p in picked_points])
    min_bound = np.min(coordinates, axis=0)
    max_bound = np.max(coordinates, axis=0)

    triangle_indices = []
    # 检查每个三角面片
    for idx, triangle in enumerate(triangles):
        # 获取三角面片的三个顶点
        p1, p2, p3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        # 检查三角面片的每个顶点是否都在立方体内
        if (np.all(min_bound <= p1) and np.all(p1 <= max_bound) and
            np.all(min_bound <= p2) and np.all(p2 <= max_bound) and
            np.all(min_bound <= p3) and np.all(p3 <= max_bound)):
            triangle_indices.append(idx)

    # Create a wireframe box for visualization
    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    )

    colors = np.array([[0, 1, 0] for _ in range(len(bbox.lines))])  # 绿色RGB
    bbox.colors = o3d.utility.Vector3dVector(colors)
    return triangle_indices, bbox


def create_highlighted_mesh(mesh, selected_triangles):
    # Create a new mesh that only includes selected triangles
    highlighted_mesh = o3d.geometry.TriangleMesh()
    highlighted_mesh.vertices = mesh.vertices
    highlighted_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles)[selected_triangles])
    highlighted_mesh.compute_vertex_normals()

    return highlighted_mesh

def create_highlighted_mesh_red(mesh, selected_triangles):
    # 创建一个新的TriangleMesh实例
    visualized_mesh = o3d.geometry.TriangleMesh()

    # 复制顶点和面片数据
    visualized_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    visualized_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))

    # 确保每个顶点都有颜色，初始为白色
    vertex_colors = np.ones((len(mesh.vertices), 3))  # 白色
    visualized_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # 根据面片索引设置顶点颜色为红色
    for tri_idx in selected_triangles:
        for vertex_idx in mesh.triangles[tri_idx]:
            visualized_mesh.vertex_colors[vertex_idx] = [1, 0, 0]  # 红色

    return visualized_mesh



## TODO: Intergrate it in the main workflow, here is just a test.

# a tricky way to select features, implemented by o3d get_picked_points function.
def selection():
    ##################################################
    #####             Initialization           #######
    ##################################################
    start_time = time.time()
    # Set the model file path
    model_directory = "D:\\PATH_PLANNING\\pp01\\models"
    model_name = "P00023955-A110-downside_AS4.obj"
    model_path = os.path.join(model_directory, model_name)

    # Create a visualization window using VisualizerWithVertexSelection for interactive selection

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name="Model Visualization", width=1800, height=1600)
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points


    # Load the model
    objmesh = load_obj(model_path)
    rotation_angles = [0, 0, 0]
    objmesh = rotate_model(objmesh, rotation_angles)
    # Adjust model position
    objmesh = adjust_model_position(objmesh)

    # Create wireframe after transformations
    wireframe = create_wireframe(objmesh)


    # Add geometry to the window
    vis.add_geometry(objmesh)
    # vis.add_geometry(wireframe)


    # Run the visualization
    vis.run()

    # Use destroy_window to release all resources associated with the window
    vis.destroy_window()


    picked_points = vis.get_picked_points()
    indices = [p.index for p in picked_points]  # 获取所有选中点的索引
    coordinates = [p.coord for p in picked_points]  # 获取所有选中点的坐标

    for idx, coord in zip(indices, coordinates):
        print(f"Index: {idx}, Coordinate: {coord}")

    selected_triangles, bbox = filter_triangles_by_bounding_box(picked_points, objmesh)

    highlighted_mesh = create_highlighted_mesh(objmesh, selected_triangles)
    # highlighted_mesh = create_highlighted_mesh_red(objmesh, selected_triangles)

    # Create a visualization window
    visu = o3d.visualization.Visualizer()
    visu.create_window(window_name="Model Visualization", width=1800, height=1600)
    # Set rendering options
    render_option = visu.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0.75, 0.75, 0.75])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points

    visu.add_geometry(wireframe)
    # visu.add_geometry(objmesh)
    visu.add_geometry(highlighted_mesh)
    # visu.add_geometry(bbox)
    visu.run()
    visu.destroy_window()

# if __name__ == "__main__":
#     selection()
