import open3d as o3d
import os
import numpy as np
import time
from uniform_viewpoint_generator import *
from geometry import *
from window_visualizer import *
from model_loader import *
from viewpoints_generator import *
from ray_casting import *
from vis_quality import *
from objective_function import *
from diagram import *
import sys


def setup_initial_view(vis):
    """
    设置初始视角，使相机朝向 x 轴的负方向。

    参数:
    - vis: Open3D 可视化窗口对象
    """
    ctr = vis.get_view_control()
    # 相机的焦点，即相机看向的点
    lookat = np.array([0, 0, 0])  # 相机将聚焦在原点
    # 相机的位置点，此设置使得相机在 x 正方向，朝向原点
    camera_pos = np.array([-1, -1.5, -1/2])  # 放置在 x 轴负方向
    # 相机的上方向，这里设置为 z 轴方向，保持相机竖直
    up = np.array([0, 0, 1])
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_front(camera_pos - lookat)  # front 是从 lookat 指向 camera_pos 的向量



def main():

    ##################################################
    #####             Initialization           #######
    ##################################################

    start_time = time.time()
    # Set the model file path
    model_directory = "D:\\PATH_PLANNING\\pp01\\models"
    model_name = "P00023955-A110-downside_AS4.obj"
    model_path = os.path.join(model_directory, model_name)



    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Model Visualization", width=2500, height=1500)


    # Load the model
    # objmesh, wireframe = load_obj_and_create_wireframe(model_path)
    objmesh = load_obj(model_path)
    rotation_angles = [0, 0, 0]  # x-axis 90 degree
    objmesh = rotate_model(objmesh, rotation_angles)
    # Adjust model position
    objmesh = adjust_model_position(objmesh)
    total_mesh_area = calculate_total_mesh_area(objmesh)



    aabb, obb = get_bbox(objmesh)
    max_dimension,length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
    print(f"base_center: {base_center} center: {center}")



    # Create wireframe after transformations
    wireframe = create_wireframe(objmesh)
    # normals = visualize_mesh_normals(objmesh)

    print("Initialization completed in {:.2f} seconds".format(time.time() - start_time))



    ############################################
    #               ray casting               ##
    ############################################
    viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 3, b=480 + width / 3, c=490 ,
                                                   center=center)
    filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min= 100)


    arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints,[1,0,0])

    # Print viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view= viewpoint
        print(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")

    # Create RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    scene.add_triangles(objmesh_t )

    all_hits = set()
    # viewpoint_indices = [339, 207, 290, 58, 41, 349, 22, 355, 186, 6, 36]  #  TRANSMISSION
    # viewpoint_indices = [82, 248, 295, 239, 186] #base
    # viewpoint_indices = [347, 258, 357, 302, 110, 351, 123, 271, 358] #PiKACHU
    # viewpoint_indices = [82, 460, 475, 343, 415, 396, 408, 573, 282, 559, 542, 419, 366, 548, 252, 375, 260, 239]
    viewpoint_indices = [83]

    # for idx in viewpoint_indices:
    #     viewpoint = viewpoints[idx]
    #     eye_center = np.array(viewpoint[0])
    #     eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=101)
    #     rays_viz = visualize_rays_from_viewpoints(eye_center, eye_left, eye_right, apex, scene)
    #     for line_set in rays_viz:
    #         vis.add_geometry(line_set)
    #     line_set3 = create_line_set3(eye_center, eye_left, eye_right)
    #     line_set2 = create_line_set2(eye_center, eye_left, eye_right, apex)
    #     vis.add_geometry(line_set3)
    #     vis.add_geometry(line_set2)
    #     vis.poll_events()
    #     vis.update_renderer()
    #
    #     time.sleep(1)



    # rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)
    #
    # valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)
    # unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
    # hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)
    #
    # filtered_hits_intersection = filter_triangles_by_depth(eye_center, center, objmesh, hits_intersection, 420, 600)
    # all_hits.update(filtered_hits_intersection)
    # print("Number of filtered_hits_intersection:", len(filtered_hits_intersection))

    #############################################
    ##            Optical-quality              ##
    #############################################

    # coverage_ratio = calculate_total_area_of_hit_triangles(objmesh, all_hits) / total_mesh_area
    #
    # print(f"Current coverage ratio: {coverage_ratio:.2%}")
    #
    # print("Optical-quality caculation completed in {:.2f} seconds".format(time.time() - start_time))
    #
    #


    #############################################
    ## Add elements to the visualization window##
    #############################################


    vis.add_geometry(wireframe)
    vis.add_geometry(objmesh)

    vis.add_geometry(aabb)

    #
    # for arrow in arrow_list:
    #     vis.add_geometry(arrow)


    # # # # 可视化被击中的面片
    visualized_mesh = visualize_hit_faces(objmesh, all_hits)
    # vis.add_geometry(visualized_mesh)

    # Set rendering options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0, 0, 0])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points


    # Add XYZ coordinate axes
    coordinate_frame = coordinate()
    vis.add_geometry(coordinate_frame)

    #Add 底平面
    ground_plane = create_ground_plane()
    vis.add_geometry(ground_plane)

    # #Add 辅助线
    # line_set = axis(base_center)
    # vis.add_geometry(line_set)
    # ray = generate_ray(np.pi / 2, np.pi / 2)
    # vis.add_geometry(ray)

    for idx in viewpoint_indices:
        viewpoint = viewpoints[idx]
        eye_center = np.array(viewpoint[0])
        eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=101)
        rays_viz = visualize_rays_from_viewpoints(eye_center, eye_left, eye_right, apex, scene)
        for line_set in rays_viz:
            vis.add_geometry(line_set)
        line_set3 = create_line_set3(eye_center, eye_left, eye_right)
        line_set2 = create_line_set2(eye_center, eye_left, eye_right, apex)
        vis.add_geometry(line_set3)
        vis.add_geometry(line_set2)
        # vis.reset_view_point(True)
        setup_initial_view(vis)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)

    # Run the visualization
    vis.run()


if __name__ == "__main__":
    main()
