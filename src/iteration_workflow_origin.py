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

#main workflow
def main():
    ##################################################
    #####             Initialization           #######
    ##################################################
    logger = setup_logging()
    start_time = time.time()
    # Set the model file path
    model_directory = "D:\\PATH_PLANNING\\pp01\\models"
    model_name = "DemoPart_EMEA_processed2.obj"
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


    rotation_angles = [np.pi, 0, 0]
    objmesh = rotate_model(objmesh, rotation_angles)
    # Adjust model position
    objmesh = adjust_model_position(objmesh)



    aabb, obb = get_bbox(objmesh)
    max_dimension,length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
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
    viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 3, b=480 + width / 3, c=480 + height / 3,
                                                   center=center)
    filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min= 30)

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
            eye_center = np.array(viewpoint[0])
            eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=101)
            rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)

            valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)
            unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
            hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)

            hits_intersection = filter_triangles_by_depth(eye_center, center, objmesh, hits_intersection, 420,570)

            # # Remove already selected hits
            # hits_intersection = hits_intersection.difference(all_hits)
            hits_intersection = hits_intersection.difference(all_hits)

            angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh, hits_intersection)

            c = calculate_costfunction(objmesh, total_mesh_area, angles_list, distance_list, hits_intersection, a=1, b=1, e=0.6, f=0.4)
            print(f"Result of costfunction:  {c}")

            # #vis-quality anasys
            # scores = normalize_and_get_scores(objmesh, angles_list, distance_list)
            # classify_and_summarize_scores(scores)

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

            if coverage_ratio >= 0.99 or growth_rate < 0.002:
                print(f"Current coverage ratio: {coverage_ratio:.2%}")
                print("Coverage target reached or growth too small.")
                logger.info(f"Coverage target reached or growth too small. Final coverage ratio: {coverage_ratio:.2%}")
                break
            else:
                print(f"Current coverage ratio: {coverage_ratio:.2%} (Growth: {growth_rate:.2f}%)")
                logger.info(f"Current coverage ratio: {coverage_ratio:.2%} (Growth: {growth_rate:.2f}%)")

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


    #############################################
    ## Add elements to the visualization window##
    #############################################

    # vis.add_geometry(objmesh)
    # vis.add_geometry(wireframe)
    # # ##法线可视化
    # # vis.add_geometry(average_normals)
    #
    # vis.add_geometry(aabb)
    # # vis.add_geometry(obb)
    #
    #
    # for arrow in arrow_list:
    #     vis.add_geometry(arrow)
    #
    #
    # # Set rendering options
    # render_option = vis.get_render_option()
    # render_option.light_on = True
    # render_option.background_color = np.array([0.05, 0.05, 0.05])  # Set background color to dark grey
    # render_option.point_size = 5  # If it is a point cloud, you can set the size of the points
    #
    #
    # # Add XYZ coordinate axes
    # coordinate_frame = coordinate()
    # vis.add_geometry(coordinate_frame)
    #
    # #Add 底平面
    # ground_plane = create_ground_plane()
    # vis.add_geometry(ground_plane)
    #
    # #Add 辅助线
    # line_set = axis(base_center)
    # vis.add_geometry(line_set)
    # ray = generate_ray(np.pi / 2, np.pi / 2)
    # vis.add_geometry(ray)
    #
    #
    # # Run the visualization
    # vis.run()


if __name__ == "__main__":
    main()
