import configparser
from logger import setup_logging
import os
import time
from uniform_viewpoint_generator import *
from geometry import *
from window_visualizer import *
from model_loader import *
from viewpoints_generator import *
from ray_casting import *
from vis_quality import *
from objective_function import *
from coo_transformer import *
from feature_selection_workflow import *


def _get_system_settings(section, key):
    config_system = configparser.ConfigParser()
    config_system.read('config_system.ini')
    value = config_system.get(section, key)
    if ',' in value:
        return np.array([float(x.strip()) for x in value.split(',')])
    try:
        return float(value)
    except ValueError:
        return value


def initialize_settings():
    settings = {
        'model_directory': _get_system_settings('Paths', 'model_directory'),
        'model_name': _get_system_settings('Paths', 'model_name'),
        'log_filename': _get_system_settings('Paths', 'log_filename'),
        'rotation_angles': _get_system_settings('Viewpoints', 'rotation_angles'),
        'angle_step_phi_large': _get_system_settings('Viewpoints', 'angle_step_phi_large'),
        'angle_step_phi_small': _get_system_settings('Viewpoints', 'angle_step_phi_small'),
        'angle_step_theta': _get_system_settings('Viewpoints', 'angle_step_theta'),
        'z_min': _get_system_settings('Viewpoints', 'z_min'),
        'min_depth': _get_system_settings('Scanner', 'min_depth'),
        'max_depth': _get_system_settings('Scanner', 'max_depth'),
        'displacement': _get_system_settings('Scanner', 'displacement'),
        'angle': _get_system_settings('Scanner', 'angle'),
        'a':_get_system_settings('Costfunction', 'a'),
        'b':_get_system_settings('Costfunction', 'b'),
        'e':_get_system_settings('Costfunction', 'e'),
        'f':_get_system_settings('Costfunction', 'f'),
        # 'log_text':_get_system_settings('Transformation', 'log_text'),
        # 'input_string':_get_system_settings('Transformation', 'input_string'),
        # 'output_filename': _get_system_settings('Transformation', 'output_filename')
    }
    return settings


 ##TODO: Use o3d.visualization.gui to combine them as a demo-GUI window

def integrated_workflow(index=10):

    ##################################################
    #####             Initialization           #######
    ##################################################
    start_time = time.time()
    settings = initialize_settings()
    model_path = os.path.join(settings['model_directory'], settings['model_name'])
    logger = setup_logging(filename=settings['log_filename'])

###TODO: Here you can use o3d.visualization.gui develop it as a demo-software
    # app = o3d.visualization.gui.Application.instance
    # app.initialize()

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Model Visualization", width=2000, height=1500)

    # Load and adjust the model
    objmesh = load_obj(model_path)
    objmesh = rotate_model(objmesh, settings['rotation_angles'])
    # Adjust model position
    objmesh = adjust_model_position(objmesh)
    aabb, obb = get_bbox(objmesh)
    max_dimension, length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
    # center[1] += 30  # If you dont want to put it at the center,change here
    # center[0] += 10
    # center[2] -= 18.1
    print(f"base_center: {base_center} center: {center}")

    ####Demo block, TC模型可视化，需调整位置
    # model_name1 = "TranscanC1.obj"
    # model_path1 = os.path.join(model_directory, model_name1)
    # objmesh1 = load_obj(model_path1)
    # rotation_angles1 = [np.pi/2, np.pi/2+np.pi/3, -np.pi/4]  # x-axis 90 degree
    # objmesh1 = rotate_model(objmesh1, rotation_angles1)
    # objmesh1 = adjust_model_position1(objmesh1, (302.66291794, -172.09147777,  372.02090856))
    # vis.add_geometry(objmesh1)


    # Create wireframe after transformations
    wireframe = create_wireframe(objmesh)

    print("Initialization completed in {:.2f} seconds".format(time.time() - start_time))

    ############################################
    #      Ray-casting Simulation             ##
    ############################################
    ##generate the candidate viewpoints for further use
    viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 3, b=480 + width / 3, c=480 + height / 3,
                                                   center=center,
                                                   angle_step_phi_large=settings['angle_step_phi_large'],
                                                   angle_step_phi_small=settings['angle_step_phi_small'],
                                                   angle_step_theta=settings['angle_step_theta'])
    filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min=settings['z_min'])
    arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints, [1, 0, 0])

    # Print viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view = viewpoint
        print(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")

    # Create Raycasting Scene
    scene = o3d.t.geometry.RaycastingScene()
    objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    scene.add_triangles(objmesh_t)

    if filtered_viewpoints:
        first_viewpoint_coordinates = filtered_viewpoints[index][0]
        eye_center = np.array(first_viewpoint_coordinates)
        print(" ")
        print(
            "#########################################################################################################")
        print("Coordinates of the viewpoint for inspection:", eye_center)
    else:
        print("No viewpoints generated.")

    eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=settings['displacement'],
                                                       angle=settings['angle'])
    rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)

    ##Visualize the ray as lineset
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

    filtered_hits_intersection = filter_triangles_by_depth(eye_center, center, objmesh, hits_intersection,
                                                           min_depth=settings['min_depth'],
                                                           max_depth=settings['max_depth'])
    print(" ")
    print("Number of depth&angle-filtered hits intersection:", len(filtered_hits_intersection))


    ##   Optical-quality caculation


    total_mesh_area = calculate_total_mesh_area(objmesh)
    angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh,
                                                                     filtered_hits_intersection)
    c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, filtered_hits_intersection,
                               a=settings['a'], b=settings['b'], e=settings['e'], f=settings['f'])
    print(" ")
    print(f"Result of Objective Function:  {c}")
    print(" ")
    scores = normalize_and_get_scores(objmesh, angles_list, distance_list)
    classify_and_summarize_scores(scores)

    # # visualize the hited faces
    visualized_mesh = visualize_hit_faces(objmesh, filtered_hits_intersection)
    vis.add_geometry(visualized_mesh)

    line_set3 = create_line_set3(eye_center, eye_left, eye_right)
    line_set2 = create_line_set2(eye_center, eye_left, eye_right, apex)
    vis.add_geometry(line_set3)
    vis.add_geometry(line_set2)

    print("Ray casting completed in {:.2f} seconds".format(time.time() - start_time))

    coverage_ratio = calculate_coverage_ratio(filtered_hits_intersection, objmesh)
    # print("Number of objmesh.triangles:", len(objmesh.triangles))
    # coverage_ratio = len(hits_intersection) / len(objmesh.triangles)
    print(" ")
    print(f"Current coverage ratio: {coverage_ratio:.2%}")

    print("Optical-quality caculation completed in {:.2f} seconds".format(time.time() - start_time))


    ## Add elements to the visualization window

    vis.add_geometry(wireframe)
    vis.add_geometry(objmesh)

    vis.add_geometry(aabb)

    for arrow in arrow_list:
        vis.add_geometry(arrow)

    # Set rendering options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points

    # Add XYZ coordinate axes
    coordinate_frame = coordinate()
    vis.add_geometry(coordinate_frame)
    centerpoint = create_point(center)
    vis.add_geometry(centerpoint)

    # Run the visualization
    vis.run()
    vis.destroy_window()

    # 等待用户输入以继续
    input("Press Enter to start Optimal-viewpoints Generation...")


    #############################################
    ##     Optimal-viewpoints Generation       ##
    #############################################


    # Print viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view = viewpoint
        print(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")
    # Log viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view = viewpoint
        logger.info(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")

    # arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints)

    # # Create RaycastingScene
    # scene = o3d.t.geometry.RaycastingScene()
    # objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    # scene.add_triangles(objmesh_t)

    ############################################

    results = {}
    hits_intersections = {}

    for idx, viewpoint in enumerate(filtered_viewpoints):
        eye_center = np.array(viewpoint[0])
        eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=settings['displacement'],
                                                           angle=settings['angle'])
        rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)
        # print("Raycasting completed in {:.2f} seconds".format(time.time() - start_time))
        valid_hit_triangle_indices = filter_hits_by_angle_for_three_views_optimized(objmesh, rays, ans)
        hits_intersection = compute_unique_intersection_of_hits(valid_hit_triangle_indices)
        hits_intersection = filter_triangles_by_depth_optimized(eye_center, center, objmesh, hits_intersection,
                                                                min_depth=settings['min_depth'],
                                                                max_depth=settings['max_depth'])
        # print("Filter completed in {:.2f} seconds".format(time.time() - start_time))
        results[idx] = {'eye_center': eye_center, 'center': center}
        hits_intersections[idx] = hits_intersection

    print("finish simulation")
    print("Raycasting simulationcompleted in {:.2f} seconds".format(time.time() - start_time))

    best_viewpoints = []
    all_hits = set()
    # Pair each viewpoint with its original index
    remaining_viewpoints = [(idx, vp) for idx, vp in enumerate(filtered_viewpoints)]
    previous_coverage_ratio = 0.0

    while remaining_viewpoints:
        current_best = None
        current_best_score = -np.inf
        best_idx = -1

        for idx, (original_idx, viewpoint) in enumerate(remaining_viewpoints):

            hits_intersection = hits_intersections[original_idx]  # Directly retrieve the set
            assert isinstance(hits_intersection, set), "hits_intersection must be a set"

            hits_intersection = hits_intersection.difference(all_hits)

            eye_center = results[original_idx]['eye_center']
            center = results[original_idx]['center']

            hits_intersection = hits_intersection.difference(all_hits)

            angles_list, distance_list = calculate_view_angles_and_distances_optimized(eye_center, center, objmesh,
                                                                                       hits_intersection)

            c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, hits_intersection,
                                       a=settings['a'], b=settings['b'], e=settings['e'], f=settings['f'])
            print(f" Costfunction result of Viewpoint {idx} costfunction:  {c}")

            if c > current_best_score:
                current_best = {
                    "index": original_idx,  # Use the original index
                    "position": viewpoint[0],
                    "direction": viewpoint[1],
                    "view": viewpoint[2],
                    "score": c,
                    "hits_intersection": hits_intersection
                }
                current_best_score = c
                best_idx = idx  # Track the best index to remove

        if current_best:
            best_viewpoints.append(current_best)
            print(
                f"Added best viewpoint: Index {current_best['index']}, Position {current_best['position']}, Direction {current_best['direction']}, View {current_best['view']}, Score {current_best['score']}")
            logger.info(
                f"Added best viewpoint: Index {current_best['index']}, Position {current_best['position']}, Direction {current_best['direction']}, View {current_best['view']}, Score {current_best['score']}")
            all_hits.update(current_best["hits_intersection"])
            del remaining_viewpoints[best_idx]  # Remove the best viewpoint from the list

            coverage_ratio = calculate_total_area_of_hit_triangles(objmesh, all_hits) / total_mesh_area
            growth_rate = coverage_ratio - previous_coverage_ratio

            if coverage_ratio >= 0.97 or growth_rate < 0.0002:
                print(f"Current coverage ratio: {coverage_ratio:.2%}")
                print("Coverage target reached or growth too small.")
                logger.info(f"Coverage target reached or growth too small. Final coverage ratio: {coverage_ratio:.2%}")
                break
            else:
                print(f"Current coverage ratio: {coverage_ratio:.2%} ")
                logger.info(f"Current coverage ratio: {coverage_ratio:.2%}")

            previous_coverage_ratio = coverage_ratio  # 更新上一次的覆盖率
        else:
            break

    print("Optical-quality calculation completed in {:.2f} seconds".format(time.time() - start_time))

    # Print all best viewpoints
    for vp in best_viewpoints:
        print(
            f"Viewpoint {vp['index']}: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")
    # Log all best viewpoints
    for vp in best_viewpoints:
        logger.info(
            f"Viewpoint {vp['index']}: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")



    # 在关闭视窗并等待用户输入后继续执行最优视点的可视化
    input("Press Enter to start Optimal-viewpoints Visualization...")

    #############################################
    ##    Optimal-viewpoints Visualization     ##
    #############################################

    # 提取前10个最优视点的索引
    viewpoint_indices = [vp['index'] for vp in best_viewpoints[:10]]

    # Create a visualization window
    visu = o3d.visualization.Visualizer()
    visu.create_window(window_name="Model Visualization", width=2500, height=1500)

    # Set rendering options
    render_option = visu.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0, 0, 0])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points

    visu.add_geometry(wireframe)
    visu.add_geometry(objmesh)

    visu.add_geometry(aabb)

    def setup_initial_view(visu):
        """
        设置初始视角，使相机朝向 x 轴的负方向。

        参数:
        - vis: Open3D 可视化窗口对象
        """
        ctr = visu.get_view_control()
        # 相机的焦点，即相机看向的点
        lookat = np.array([0, 0, 0])  # 相机将聚焦在原点
        # 相机的位置点，此设置使得相机在 x 正方向，朝向原点
        camera_pos = np.array([-1, -1.5, -1 / 2])  # 放置在 x 轴负方向
        # 相机的上方向，这里设置为 z 轴方向，保持相机竖直
        up = np.array([0, 0, 1])
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_front(camera_pos - lookat)  # front 是从 lookat 指向 camera_pos 的向量

    for idx in viewpoint_indices:
        viewpoint = viewpoints[idx]
        eye_center = np.array(viewpoint[0])
        eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=settings['displacement'], angle =settings['angle'])
        rays_viz = visualize_rays_from_viewpoints(eye_center, eye_left, eye_right, apex, scene)
        for line_set in rays_viz:
            visu.add_geometry(line_set)
        line_set3 = create_line_set3(eye_center, eye_left, eye_right)
        line_set2 = create_line_set2(eye_center, eye_left, eye_right, apex)
        visu.add_geometry(line_set3)
        visu.add_geometry(line_set2)
        # vis.reset_view_point(True)
        setup_initial_view(visu)
        visu.poll_events()
        visu.update_renderer()
        time.sleep(1)

    # Run the visualization
    visu.run()
    visu.destroy_window()


def integrated_workflow_with_selection(index=10):
    ##################################################
    #####             Initialization           #######
    ##################################################
    start_time = time.time()
    settings = initialize_settings()
    model_path = os.path.join(settings['model_directory'], settings['model_name'])
    logger = setup_logging(filename=settings['log_filename'])

    ###TODO: Here you can use o3d.visualization.gui develop it as a demo-software
    # app = o3d.visualization.gui.Application.instance
    # app.initialize()

    # Create a visualization window
    vis1 = o3d.visualization.VisualizerWithVertexSelection()
    vis1.create_window(window_name="Model Visualization", width=2000, height=1500)
    # Load the model
    objmesh = load_obj(model_path)
    objmesh = rotate_model(objmesh, settings['rotation_angles'])
    # Adjust model position
    objmesh = adjust_model_position(objmesh)

    ####Demo block, TC模型可视化，需调整位置
    # model_name1 = "TranscanC1.obj"
    # model_path1 = os.path.join(model_directory, model_name1)
    # objmesh1 = load_obj(model_path1)
    # rotation_angles1 = [np.pi/2, np.pi/2+np.pi/3, -np.pi/4]  # x-axis 90 degree
    # objmesh1 = rotate_model(objmesh1, rotation_angles1)
    # objmesh1 = adjust_model_position1(objmesh1, (302.66291794, -172.09147777,  372.02090856))
    # vis.add_geometry(objmesh1)

    aabb, obb = get_bbox(objmesh)
    max_dimension, length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
    # center[1] += 30  # 针对该物体特殊处理
    # center[0] += 10
    # center[2] -= 18.1
    print(f"base_center: {base_center} center: {center}")

    # Create wireframe after transformations
    wireframe = create_wireframe(objmesh)

    print("Initialization completed in {:.2f} seconds".format(time.time() - start_time))
    # Add geometry to the window
    vis1.add_geometry(objmesh)
    vis1.add_geometry(wireframe)


    # Run the visualization
    vis1.run()

    # Use destroy_window to release all resources associated with the window
    vis1.destroy_window()


    picked_points = vis1.get_picked_points()
    indices = [p.index for p in picked_points]  # 获取所有选中点的索引
    coordinates = [p.coord for p in picked_points]  # 获取所有选中点的坐标

    for idx, coord in zip(indices, coordinates):
        print(f"Index: {idx}, Coordinate: {coord}")

    selected_triangles, bbox = filter_triangles_by_bounding_box(picked_points, objmesh)
    selected_triangles = set(selected_triangles)
    # highlighted_mesh = create_highlighted_mesh(objmesh, selected_triangles)

    ############################################
    #      Ray-casting Simulation             ##
    ############################################
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Model Visualization", width=2000, height=1500)

    viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 3, b=480 + width / 3, c=480 + height / 3,
                                                   center=center,
                                                   angle_step_phi_large=settings['angle_step_phi_large'],
                                                   angle_step_phi_small=settings['angle_step_phi_small'],
                                                   angle_step_theta=settings['angle_step_theta'])
    filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min=settings['z_min'])

    arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints, [1, 0, 0])

    # Print viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view = viewpoint
        print(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")

    # Create RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    scene.add_triangles(objmesh_t)

    if filtered_viewpoints:
        first_viewpoint_coordinates = filtered_viewpoints[index][0]
        eye_center = np.array(first_viewpoint_coordinates)
        print(" ")
        print(
            "#########################################################################################################")
        print("Coordinates of the viewpoint for inspection:", eye_center)
    else:
        print("No viewpoints generated.")

    eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=settings['displacement'],
                                                       angle=settings['angle'])
    rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)

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

    filtered_hits_intersection = filter_triangles_by_depth(eye_center, center, objmesh, hits_intersection,
                                                           min_depth=settings['min_depth'],
                                                           max_depth=settings['max_depth'])
    print(" ")
    print("Number of depth&angle-filtered hits intersection:", len(filtered_hits_intersection))
    filtered_hits_intersection = filtered_hits_intersection.intersection(selected_triangles)

    #############################################
    ##            Optical-quality              ##
    #############################################

    total_mesh_area = calculate_total_mesh_area(objmesh)
    angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh,
                                                                     filtered_hits_intersection)
    c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, filtered_hits_intersection,
                               a=settings['a'], b=settings['b'], e=settings['e'], f=settings['f'])
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

    print("Ray casting completed in {:.2f} seconds".format(time.time() - start_time))

    coverage_ratio = calculate_coverage_ratio(filtered_hits_intersection, objmesh)
    # print("Number of objmesh.triangles:", len(objmesh.triangles))
    # coverage_ratio = len(hits_intersection) / len(objmesh.triangles)
    print(" ")
    print(f"Current coverage ratio: {coverage_ratio:.2%}")

    print("Optical-quality caculation completed in {:.2f} seconds".format(time.time() - start_time))

    #############################################
    ## Add elements to the visualization window##
    #############################################
    vis.add_geometry(bbox)
    vis.add_geometry(wireframe)
    vis.add_geometry(objmesh)
    # ##法线可视化
    # vis.add_geometry(average_normals)

    vis.add_geometry(aabb)

    for arrow in arrow_list:
        vis.add_geometry(arrow)

    # Set rendering options
    render_option = vis.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points

    # Add XYZ coordinate axes
    coordinate_frame = coordinate()
    vis.add_geometry(coordinate_frame)
    centerpoint = create_point(center)
    vis.add_geometry(centerpoint)

    # Run the visualization
    vis.run()
    vis.destroy_window()

    vis.remove_geometry(visualized_mesh, reset_bounding_box=False)  # 移除被击中的面片的可视化，但不重置视图框架

    # 等待用户输入以继续
    input("Press Enter to start Optimal-viewpoints Generation...")

    #############################################
    ##     Optimal-viewpoints Generation       ##
    #############################################

    # Print viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view = viewpoint
        print(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")
    # Log viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view = viewpoint
        logger.info(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")

    # arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints)

    # # Create RaycastingScene
    # scene = o3d.t.geometry.RaycastingScene()
    # objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    # scene.add_triangles(objmesh_t)

    ############################################

    results = {}
    hits_intersections = {}

    for idx, viewpoint in enumerate(filtered_viewpoints):
        eye_center = np.array(viewpoint[0])
        eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=settings['displacement'],
                                                           angle=settings['angle'])
        rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)
        # print("Raycasting completed in {:.2f} seconds".format(time.time() - start_time))
        valid_hit_triangle_indices = filter_hits_by_angle_for_three_views_optimized(objmesh, rays, ans)
        hits_intersection = compute_unique_intersection_of_hits(valid_hit_triangle_indices)
        hits_intersection = hits_intersection.intersection(selected_triangles)
        hits_intersection = filter_triangles_by_depth_optimized(eye_center, center, objmesh, hits_intersection,
                                                                min_depth=settings['min_depth'],
                                                                max_depth=settings['max_depth'])

        # print("Filter completed in {:.2f} seconds".format(time.time() - start_time))
        results[idx] = {'eye_center': eye_center, 'center': center}
        hits_intersections[idx] = hits_intersection

    print("finish simulation")
    print("Raycasting simulationcompleted in {:.2f} seconds".format(time.time() - start_time))

    best_viewpoints = []
    all_hits = set()
    # Pair each viewpoint with its original index
    remaining_viewpoints = [(idx, vp) for idx, vp in enumerate(filtered_viewpoints)]
    previous_coverage_ratio = 0.0

    while remaining_viewpoints:
        current_best = None
        current_best_score = -np.inf
        best_idx = -1

        for idx, (original_idx, viewpoint) in enumerate(remaining_viewpoints):

            hits_intersection = hits_intersections[original_idx]  # Directly retrieve the set
            assert isinstance(hits_intersection, set), "hits_intersection must be a set"

            hits_intersection = hits_intersection.difference(all_hits)

            eye_center = results[original_idx]['eye_center']
            center = results[original_idx]['center']

            angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh,
                                                                                       hits_intersection)

            c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, hits_intersection,
                                       a=settings['a'], b=settings['b'], e=settings['e'], f=settings['f'])
            print(f" Costfunction result of Viewpoint {idx} costfunction:  {c}")

            if c > current_best_score:
                current_best = {
                    "index": original_idx,  # Use the original index
                    "position": viewpoint[0],
                    "direction": viewpoint[1],
                    "view": viewpoint[2],
                    "score": c,
                    "hits_intersection": hits_intersection
                }
                current_best_score = c
                best_idx = idx  # Track the best index to remove

        if current_best:
            best_viewpoints.append(current_best)
            print(
                f"Added best viewpoint: Index {current_best['index']}, Position {current_best['position']}, Direction {current_best['direction']}, View {current_best['view']}, Score {current_best['score']}")
            logger.info(
                f"Added best viewpoint: Index {current_best['index']}, Position {current_best['position']}, Direction {current_best['direction']}, View {current_best['view']}, Score {current_best['score']}")
            all_hits.update(current_best["hits_intersection"])
            del remaining_viewpoints[best_idx]  # Remove the best viewpoint from the list

            coverage_ratio = calculate_total_area_of_hit_triangles(objmesh, all_hits) / total_mesh_area
            growth_rate = coverage_ratio - previous_coverage_ratio

            if coverage_ratio >= 0.97 or growth_rate < 0.0002:
                print(f"Current coverage ratio: {coverage_ratio:.2%}")
                print("Coverage target reached or growth too small.")
                logger.info(f"Coverage target reached or growth too small. Final coverage ratio: {coverage_ratio:.2%}")
                break
            else:
                print(f"Current coverage ratio: {coverage_ratio:.2%} ")
                logger.info(f"Current coverage ratio: {coverage_ratio:.2%}")

            previous_coverage_ratio = coverage_ratio  # 更新上一次的覆盖率
        else:
            break

    # print("Optical-quality calculation completed in {:.2f} seconds".format(time.time() - start_time))

    # Print all best viewpoints
    for vp in best_viewpoints:
        print(
            f"Viewpoint {vp['index']}: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")
    # Log all best viewpoints
    for vp in best_viewpoints:
        logger.info(
            f"Viewpoint {vp['index']}: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")

    # 在关闭视窗并等待用户输入后继续执行最优视点的可视化
    input("Press Enter to start Optimal-viewpoints Visualization...")

    #############################################
    ##    Optimal-viewpoints Visualization     ##
    #############################################

    # 提取前10个最优视点的索引
    viewpoint_indices = [vp['index'] for vp in best_viewpoints[:10]]

    # Create a visualization window
    visu = o3d.visualization.Visualizer()
    visu.create_window(window_name="Model Visualization", width=2500, height=1500)

    # Set rendering options
    render_option = visu.get_render_option()
    render_option.light_on = True
    render_option.background_color = np.array([0, 0, 0])  # Set background color to dark grey
    render_option.point_size = 5  # If it is a point cloud, you can set the size of the points

    visu.add_geometry(wireframe)
    visu.add_geometry(objmesh)

    visu.add_geometry(aabb)

    def setup_initial_view(visu):
        """
        设置初始视角，使相机朝向 x 轴的负方向。

        参数:
        - vis: Open3D 可视化窗口对象
        """
        ctr = visu.get_view_control()
        # 相机的焦点，即相机看向的点
        lookat = np.array([0, 0, 0])  # 相机将聚焦在原点
        # 相机的位置点，此设置使得相机在 x 正方向，朝向原点
        camera_pos = np.array([-1, -1.5, -1 / 2])  # 放置在 x 轴负方向
        # 相机的上方向，这里设置为 z 轴方向，保持相机竖直
        up = np.array([0, 0, 1])
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_front(camera_pos - lookat)  # front 是从 lookat 指向 camera_pos 的向量

    for idx in viewpoint_indices:
        viewpoint = viewpoints[idx]
        eye_center = np.array(viewpoint[0])
        eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=settings['displacement'],
                                                           angle=settings['angle'])
        rays_viz = visualize_rays_from_viewpoints(eye_center, eye_left, eye_right, apex, scene)
        for line_set in rays_viz:
            visu.add_geometry(line_set)
        line_set3 = create_line_set3(eye_center, eye_left, eye_right)
        line_set2 = create_line_set2(eye_center, eye_left, eye_right, apex)
        visu.add_geometry(line_set3)
        visu.add_geometry(line_set2)
        # vis.reset_view_point(True)
        setup_initial_view(visu)
        visu.poll_events()
        visu.update_renderer()
        time.sleep(1)

    # Run the visualization
    visu.run()
    visu.destroy_window()


if __name__ == "__main__":
    while True:
        command = input("Enter 'whole' to run ray-casting on whole object, 'part' to run feature selection first, or 'exit' to quit: ").strip().lower()
        if command == 'whole':
            integrated_workflow()
        elif command == 'part':
            integrated_workflow_with_selection()
        elif command == 'exit':
            print("Exiting the program.")
            break