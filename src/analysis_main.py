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


def main():
    ##################################################
    #####             Initialization           #######
    ##################################################
    logger = setup_logging()
    start_time = time.time()
    model_directory = "D:\\PATH_PLANNING\\pp01\\models"
    model_name = "pikachu_4scale_remesh.obj"
    model_path = os.path.join(model_directory, model_name)

    objmesh = load_obj(model_path)
    objmesh = adjust_model_position(objmesh)
    aabb, obb = get_bbox(objmesh)
    max_dimension, length, width, height = bbox_dimensions(aabb)
    center, base_center = compute_object_center(aabb)
    total_mesh_area = calculate_total_mesh_area(objmesh)
    total_mesh_count = len(np.asarray(objmesh.triangles))
    print(f"total_mesh_area: {total_mesh_area}, total_mesh_count: {total_mesh_count}")
    scene = o3d.t.geometry.RaycastingScene()
    objmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(objmesh)
    scene.add_triangles(objmesh_t)

    #############################################
    ##               Viewpoint Setup           ##
    #############################################
    viewpoints = generate_viewpoints_on_ellipsoid1(a=480 + length / 3, b=480 + width / 3, c=480 + height / 3, center=center)
    filtered_viewpoints = filter_viewpoints_by_z(viewpoints, z_min=-100)
    indices_to_analyze = [45, 69, 57, 60, 20, 50, 71, 74]

    # 初始化总分类统计和面积统计
    total_classification_areas = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    total_classification_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    all_hits = set()

    # 统计视点的循环
    for idx, viewpoint in enumerate([filtered_viewpoints[i] for i in [45, 69, 57, 60, 20, 50, 71, 74]]):
        eye_center = np.array(viewpoint[0])
        eye_left, eye_right, apex = generate_eye_positions(eye_center, center, displacement=230)
        rays, ans = ray_casting_for_visualization_3eyes(eye_center, eye_left, eye_right, apex, scene)

        valid_hit_triangle_indices = filter_hits_by_angle_for_three_views(objmesh, rays, ans)
        unique_valid_hits_per_view = get_unique_valid_hits(valid_hit_triangle_indices)
        hits_intersection = compute_intersection_of_hits(unique_valid_hits_per_view)
        hits_intersection = hits_intersection.difference(all_hits)  # 确保只考虑未被扫描过的面片

        # 更新已处理面片集合
        all_hits.update(hits_intersection)

        angles_list, distance_list = calculate_view_angles_and_distances(eye_center, center, objmesh, hits_intersection)
        scores = normalize_and_get_scores(objmesh, angles_list, distance_list)
        classification_areas, classification_counts = classify_and_summarize_scores(scores)

        # 累加每个分类的面积和计数
        for grade in total_classification_areas:
            total_classification_areas[grade] += classification_areas[grade]
            total_classification_counts[grade] += classification_counts[grade]

    #############################################
    ##              Results Summary            ##
    #############################################
    # 计算总面积和面片数百分比
    total_area = sum(total_classification_areas.values())
    total_count = sum(total_classification_counts.values())

    print()
    print("Final Scan Quality Classification Summary:")
    for grade in total_classification_areas:
        area = total_classification_areas[grade]
        count = total_classification_counts[grade]
        area_percentage = (area / total_mesh_area) * 100 if total_mesh_area > 0 else 0
        count_percentage = (count / total_mesh_count) * 100
        print(f"{grade}-grade scan quality: {count} triangles ({count_percentage:.2f}%), {area:.2f} square units ({area_percentage:.2f}% of total)")

    print()
    print("Final Scan Quality Classification Summary test:")
    for grade in total_classification_areas:
        area = total_classification_areas[grade]
        count = total_classification_counts[grade]
        area_percentage = (area / total_area) * 100 if total_area > 0 else 0
        count_percentage = (count / total_count) * 100
        print(f"{grade}-grade scan quality: {count} triangles ({count_percentage:.2f}%), {area:.2f} square units ({area_percentage:.2f}% of total)")


    # print("Optical-quality calculation completed in {:.2f} seconds".format(time.time() - start_time))




if __name__ == "__main__":
    main()
