import open3d as o3d
import os
import numpy as np
import time
from uniform_viewpoint_generator import *
from geometry import *
from window_visualizer import *
from model_loader import *
from partitioner import *
from viewpoints_generator import *
from ray_casting import *
from vis_quality import *
from objective_function import *
from diagram import *
from logger import *
import concurrent.futures
import threading

#main workflow
def main():
    ##################################################
    #####             Initialization           #######
    ##################################################
    logger = setup_logging(filename='Viewpointsgeneration_oral0807.txt')
    start_time = time.time()
    # Set the model file path
    model_directory = "D:\\PATH_PLANNING\\pp01\\models"
    model_name = "P00023955-A110-downside_AS4.obj"
    model_path = os.path.join(model_directory, model_name)


    app = o3d.visualization.gui.Application.instance
    app.initialize()
    logger.info("##########################################################################################")
    # # Create a visualization window
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Model Visualization", width=800, height=600)


    # Load the model
    # objmesh, wireframe = load_obj_and_create_wireframe(model_path)
    objmesh = load_obj(model_path)


    rotation_angles = [0 , 0, 0]
    objmesh = rotate_model(objmesh, rotation_angles)
    # Adjust model position
    objmesh = adjust_model_position(objmesh)



    aabb, obb = get_bbox(objmesh)
    max_dimension,length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
    # center[2] += 100 #针对该物体特殊处理
    # center[0] += 30  # 针对该物体特殊处理
    print(f"base_center: {base_center} center: {center}")
    logger.info(f"base_center: {base_center} center: {center}")
    total_mesh_area = calculate_total_mesh_area(objmesh)

    # # Create wireframe after transformations
    # wireframe = create_wireframe(objmesh)
    # # normals = visualize_mesh_normals(objmesh)

    print("Initialization completed in {:.2f} seconds".format(time.time() - start_time))

    #############################################
    ##              Iteration                  ##
    #############################################
    viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 3, b=480 + width / 3, c=490 ,
                                                   center=center)
    # viewpoints = generate_viewpoints_on_ellipsoid1(a=490, b=490, c=490 ,
    #                                                center=center)
    filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min=400)

    # Print viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view= viewpoint
        print(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")
    # Log viewpoints and view directions for verification
    for idx, viewpoint in enumerate(filtered_viewpoints):
        point, direction, view= viewpoint
        logger.info(f"Viewpoint{idx}: {point}, Direction: {direction}, View: {view}")

    # arrow_list = visualize_viewpoints_as_arrows(filtered_viewpoints)

    # Create RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    scene.add_triangles(objmesh_t)

########################################################################################################################
    # 创建一个全局锁，用于同步对共享资源的访问
    results_lock = threading.Lock()
    results = {}
    hits_intersections = {}

    # 使用多线程并行处理每个视点的光线投射
    def process_viewpoint(idx, viewpoint):
        try:
            eye_center = np.array(viewpoint[0])
            eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=101)
            print(0)
            # 调用已有的光线投射函数，不需要更改
            rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)
            print(1)
            valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)
            print(2)
            unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
            hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)
            hits_intersection = filter_triangles_by_depth(eye_center, center, objmesh, hits_intersection, 420, 590)
            print(3)

            # 返回结果
            return idx, eye_center, hits_intersection
        except Exception as e:
            print(f"Error in viewpoint {idx}: {e}")
            return idx, None, None

    # 使用线程池并行处理视点
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有视点任务到线程池中
        futures = {executor.submit(process_viewpoint, idx, viewpoint): idx for idx, viewpoint in
                   enumerate(filtered_viewpoints)}

        # 获取每个完成任务的结果并存储
        for future in concurrent.futures.as_completed(futures):
            idx, eye_center, hits_intersection = future.result()

            # 确保写入 results 和 hits_intersections 字典时加锁
            with results_lock:
                if eye_center is not None and hits_intersection is not None:
                    results[idx] = {'eye_center': eye_center, 'center': center}
                    hits_intersections[idx] = hits_intersection

    print("finish simulation")



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

            angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh, hits_intersection)

            c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, hits_intersection, a=3, b=1, e=0.7, f=0.3)
            print(f"Result of costfunction:  {c}")

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
            print(f"Added best viewpoint: Index {current_best['index']}, Position {current_best['position']}, Direction {current_best['direction']}, View {current_best['view']}, Score {current_best['score']}")
            logger.info(f"Added best viewpoint: Index {current_best['index']}, Position {current_best['position']}, Direction {current_best['direction']}, View {current_best['view']}, Score {current_best['score']}")
            all_hits.update(current_best["hits_intersection"])
            del remaining_viewpoints[best_idx]  # Remove the best viewpoint from the list

            coverage_ratio = calculate_total_area_of_hit_triangles(objmesh, all_hits) / total_mesh_area
            growth_rate = coverage_ratio - previous_coverage_ratio

            if coverage_ratio >= 0.99 or growth_rate < 0.0002:
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
            f"Viewpoint {vp['index'] }: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")
    # Log all best viewpoints
    for vp in best_viewpoints:
        logger.info(
            f"Viewpoint {vp['index'] }: Position {vp['position']}, Direction {vp['direction']}, View {vp['view']}, Score {vp['score']}")





if __name__ == "__main__":
    main()
