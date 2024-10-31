import open3d as o3d
import os
import numpy as np
import time
from uniform_viewpoint_generator import *
from geometry import *
from window_visualizer import *
from model_loader import *
from partitioner import *
# from dilation import *
from viewpoints_generator import *
from ray_casting import *
from vis_quality import *
from objective_function import *
from diagram import *
from logger import *
import sys

def estimate_data_size(rays, ans):
    size_rays = sys.getsizeof(rays)
    size_ans = sys.getsizeof(ans)

    # 如果Open3D返回的是复杂对象，可能需要深入到对象属性中去计算
    if hasattr(ans, '__dict__'):
        for key, value in ans.__dict__.items():
            size_ans += sys.getsizeof(value)

    return size_rays, size_ans


#main workflow
def main():
    ##################################################
    #####             Initialization           #######
    ##################################################
    # logger = setup_logging(test)
    start_time = time.time()
    # Set the model file path
    model_directory = "D:\\PATH_PLANNING\\pp01\\models"
    model_name = "P00023955-A110-downside15processed_AS4.obj"
    model_path = os.path.join(model_directory, model_name)



    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Model Visualization", width=800, height=600)


    # Load the model
    # objmesh, wireframe = load_obj_and_create_wireframe(model_path)
    objmesh = load_obj(model_path)
    rotation_angles = [0, 0, 0]  # x-axis 90 degree
    objmesh = rotate_model(objmesh, rotation_angles)
    # Adjust model position
    objmesh = adjust_model_position(objmesh)


    # ### TC模型可视化，适配第25个视点
    # model_name1 = "TranscanC1.obj"
    # model_path1 = os.path.join(model_directory, model_name1)
    # objmesh1 = load_obj(model_path1)
    # rotation_angles1 = [np.pi/2, np.pi/2+np.pi/3, -np.pi/4]  # x-axis 90 degree
    # objmesh1 = rotate_model(objmesh1, rotation_angles1)
    # objmesh1 = adjust_model_position1(objmesh1, (302.66291794, -172.09147777,  372.02090856))
    # vis.add_geometry(objmesh1)
    #



    aabb, obb = get_bbox(objmesh)
    max_dimension,length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
    # center[1] += 30  # 针对该物体特殊处理
    # center[0] += 10
    # center[2] -= 18.1
    print(f"base_center: {base_center} center: {center}")

    # #划分物体为立方体区块
    # aabb_cuboids = generate_aabb_grid_base_centered(objmesh, cuboid_size = 45)
    # aabb_cuboids = filter_cuboids_containing_surface(objmesh, aabb_cuboids)
    # # # 添加每个 AABB 区块到可视化
    # for aabb_box in aabb_cuboids:
    #     vis.add_geometry(aabb_box)

    # # average_normals = visualize_average_normals(objmesh, aabb_cuboids)
    #
    # centers_with_ids = get_cuboids_centers(aabb_cuboids)
    # # 迭代 centers_with_ids 来分别获取中心点和编号
    # for center, idx in centers_with_ids:
    #     print(f"Cuboid {idx} center: {center}")
    # # centers, idx = get_cuboids_centers(aabb_cuboids)
    # # print(f"Center: {centers}, id:{idx}")
    #


    # ##选区功能
    # custom_aabb = create_custom_aabb([10, 30, 17], 28)
    # highlight_line_set = highlight_cuboid_frame([custom_aabb], 0)
    # cuboid_triangle_indices = filter_triangles_by_cuboid(objmesh, [custom_aabb], 0)
    # vis.add_geometry(highlight_line_set)



    # view_stats = analyze_mesh_by_view(highlighted_mesh)


    # Create wireframe after transformations
    wireframe = create_wireframe(objmesh)
    # normals = visualize_mesh_normals(objmesh)

    print("Initialization completed in {:.2f} seconds".format(time.time() - start_time))
    ################################################
    ##Uniformlly distributed viewpoints generation##
    ################################################

    # # Generate equally distributed viewpoints on a sphere
    # if max_dimension <= 500:
    #     radius = 500 + max_dimension
    #     viewpoints = generate_viewpoints_on_hemisphere(radius, base_center)
    #     arrow_list = visualize_viewpoints_as_arrows(viewpoints)
    # else:
    #     print("not  designed yet")


    # # Generate equally distributed viewpoints on a cylinder, use angle when max_dimension <= 300
    # if max_dimension <= 500:
    #     radius = 500 + length
    #     height = 500 + height
    #     viewpoints = generate_viewpoints_on_cylinder2(radius, height, base_center)
    #     arrow_list = visualize_viewpoints_as_arrows(viewpoints)
    # else:
    #     print("not  designed yet")

    # viewpoints = generate_viewpoints_on_hemiellipsoid(a = 500 + length, b = 500 + width, c = 500 + height, base_center = base_center, angle_step=20)
    # arrow_list = visualize_viewpoints_as_arrows(viewpoints)



    ############################################
    #               ray casting               ##
    ############################################
    # viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 2, b=480 + width / 2, c=480,
    #                                                center=center)
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



    if filtered_viewpoints:
        first_viewpoint_coordinates = filtered_viewpoints[1][0]
        eye_center = np.array(first_viewpoint_coordinates)
        print(" ")
        print("#########################################################################################################")
        print("Coordinates of the viewpoint for inspection:", eye_center)
    else:
        print("No viewpoints generated.")


    eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement =101)
    rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)



    # #估算视点信息占用内存
    # size_rays, size_ans = estimate_data_size(rays, ans)
    # print(f"Estimated size of rays: {size_rays / 1024} KB")
    # print(f"Estimated size of ans: {size_ans / 1024} KB")

    ####可视化视锥射线，目前最优解决方案########
    rays_viz = visualize_rays_from_viewpoints(eye_center, eye_left, eye_right, apex, scene)
    for line_set in rays_viz:
        vis.add_geometry(line_set)
    print(" ")
    valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)

    # 打印结果，查看每个视锥有效击中的面片数量
    for idx, hits in enumerate(valid_hit_triangle_indices):
        print(f"Number of valid hits for view {idx + 1}: {len(hits)}")
    # 获取每个视锥中不重复的有效击中面片索引
    unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
    # 打印结果，查看每个视锥的不重复有效击中面片数量
    for idx, hits in enumerate(unique_valid_hits_per_view):
        print(f"Number of unique valid hits for view {idx + 1}: {len(hits)}")
    # 获取三个视锥的面片索引交集
    hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)
    print("Number of intersecting hits:", len(hits_intersection))


    filtered_hits_intersection = filter_triangles_by_depth(eye_center, center, objmesh, hits_intersection, 405, 600)
    print(" ")
    print("Number of depth&angle-filtered hits intersection:", len(filtered_hits_intersection))

    # # # 计算该区块内面片与已过滤交集的三角面片索引，实现选区功能
    # filtered_hits_intersection = filtered_hits_intersection.intersection(cuboid_triangle_indices)

    #############################################
    ##            Optical-quality              ##
    #############################################


    # # 计算并打印所选面片的总面积
    # total_area = calculate_total_area_of_triangles(objmesh, hits_intersection)
    # print(f"Total area of the selected triangles: {total_area} square units")
    total_mesh_area = calculate_total_mesh_area(objmesh)
    angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh, filtered_hits_intersection)
    c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, filtered_hits_intersection, a=0, b=1, e=0.6, f=0.4)
    print(" ")
    print(f"Result of Objective Function:  {c}")
    print(" ")
    scores = normalize_and_get_scores(objmesh, angles_list, distance_list)
    classify_and_summarize_scores(scores)

    # # 可视化被击中的面片
    visualized_mesh = visualize_hit_faces(objmesh, filtered_hits_intersection)
    vis.add_geometry(visualized_mesh)


    line_set3 = create_line_set3(eye_center, eye_left, eye_right)
    line_set2 = create_line_set2(eye_center, eye_left, eye_right, apex)
    vis.add_geometry(line_set3)
    vis.add_geometry(line_set2)


    #
    # print("Ray casting completed in {:.2f} seconds".format(time.time() - start_time))
    #

    coverage_ratio = calculate_coverage_ratio(filtered_hits_intersection, objmesh)
    # print("Number of objmesh.triangles:", len(objmesh.triangles))
    # coverage_ratio = len(hits_intersection) / len(objmesh.triangles)
    print(" ")
    print(f"Current coverage ratio: {coverage_ratio:.2%}")

    # print("Optical-quality caculation completed in {:.2f} seconds".format(time.time() - start_time))


#################################################################################################################

    #############################################
    ##          TEST  Viewposition             ##
    #############################################
    # viewpoints = generate_viewpoints_on_ellipsoid1(a=560 + length / 2, b=560 + width / 2, c=560 + height / 2,
    #                                                center=center)
    # filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min=100)
    # # filtered_viewpoints = filter_viewpoints_by_area(filtered_viewpoints, view_stats, area_threshold=50.0)
    #
    # arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints)
    #
    # # Create RaycastingScene
    # scene = o3d.t.geometry.RaycastingScene()
    # objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    # scene.add_triangles(objmesh_t)
    #
    # views = ['Topview', 'Frontview', 'Leftview', 'Backview', 'Rightview']
    # best_viewpoints = []
    #
    # for view in views:
    #     current_viewpoints = filter_viewpoints_by_name(filtered_viewpoints, view)
    #     best_score = -np.inf
    #     best_viewpoint_data = None
    #     print("Now is", view)
    #
    #     for idx, viewpoint in enumerate(current_viewpoints):
    #         eye_center = np.array(viewpoint[0])  # 转换视点坐标为NumPy数组
    #         eye_left, eye_right = generate_eye_positions(eye_center, center, displacement=90)
    #         rays, ans, cameras = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, center, scene)
    #
    #         valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)
    #         unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
    #         hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)
    #
    #         angles_list, distance_list = calculate_view_angles_and_distances(eye_center, objmesh, hits_intersection)
    #         c = calculate_costfunction(objmesh, angles_list, distance_list, hits_intersection, a=2, b=3, e=2, f=3)
    #
    #         if c > best_score:
    #             best_score = c
    #             best_viewpoint_data = (idx, viewpoint[0], viewpoint[1], view, c)
    #
    #     if best_viewpoint_data:
    #         best_viewpoints.append(best_viewpoint_data)
    #         print(f"Best viewpoint for {view}: Index {best_viewpoint_data[0] + 1}, Score: {best_viewpoint_data[4]}")
    #
    #     print("Optical-quality calculation completed in {:.2f} seconds".format(time.time() - start_time))
    #
    # # 打印所有视角的最佳视点
    # for vp in best_viewpoints:
    #     print(f"Viewpoint {vp[0] + 1}: Position {vp[1]}, Direction {vp[2]}, View {vp[3]}, Score {vp[4]}")
    #
    # print("Optical-quality calculation completed in {:.2f} seconds".format(time.time() - start_time))
    #

    # #############################################
    # ##          TEST  All Viewpoints           ##
    # #############################################
    # viewpoints = generate_viewpoints_on_longitude_line(a=480 + length / 2, b=480 + width / 2, c=480 + height / 2,
    #                                                center=center)
    # filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min=0)
    #
    # arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints)
    #
    # # Create RaycastingScene
    # scene = o3d.t.geometry.RaycastingScene()
    # objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    # scene.add_triangles(objmesh_t)
    #
    # viewpoint_details = []  # 用于存储每个视点的详细信息
    #
    # # 检查视点列表是否不为空
    # if filtered_viewpoints:
    #     for idx, viewpoint in enumerate(filtered_viewpoints):
    #         eye_center = np.array(viewpoint[0])  # 转换视点坐标为NumPy数组
    #         eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=230)
    #         rays, ans, cameras = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)
    #
    #         valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)
    #         unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
    #         hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)
    #
    #         angles_list, distance_list = calculate_view_angles_and_distances(eye_center, objmesh, hits_intersection)
    #         c = calculate_costfunction(objmesh, angles_list, distance_list, hits_intersection, a=0, b=5, e=2, f=0)
    #
    #         # 存储视点详细信息
    #         viewpoint_details.append({
    #             "index": idx,
    #             "position": viewpoint[0],
    #             "direction": viewpoint[1],
    #             "view": viewpoint[2],
    #             "score": c
    #         })
    #
    #         print(f"Processed viewpoint {idx + 1}: Score {c}")
    #
    # else:
    #     print("No viewpoints generated.")
    #
    # # 打印所有视点的详细信息
    # for vp in viewpoint_details:
    #     print(
    #         f"Viewpoint {vp['index'] + 1}: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")
    #
    # # 调用函数，绘制图表
    # plot_viewpoint_scores(viewpoint_details)
    #
    # print("Optical-quality calculation completed in {:.2f} seconds".format(time.time() - start_time))

    #############################################
    ## Add elements to the visualization window##
    #############################################


    vis.add_geometry(wireframe)
    vis.add_geometry(objmesh)
    # ##法线可视化
    # vis.add_geometry(average_normals)

    vis.add_geometry(aabb)
    # vis.add_geometry(obb)


    for arrow in arrow_list:
        vis.add_geometry(arrow)


    # 可视化block和block内面片

    # vis.add_geometry(highlighted_mesh)


    # Set rendering options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points


    # # Add XYZ coordinate axes
    # coordinate_frame = coordinate()
    # vis.add_geometry(coordinate_frame)
    # centerpoint = create_point(center)
    # vis.add_geometry(centerpoint)

    # #Add 底平面
    # ground_plane = create_ground_plane()
    # vis.add_geometry(ground_plane)

    #
    # #Add 辅助线
    # line_set = axis(base_center)
    # vis.add_geometry(line_set)
    # ray = generate_ray(np.pi / 2, np.pi / 2)
    # vis.add_geometry(ray)



    # Run the visualization
    vis.run()


if __name__ == "__main__":
    main()
