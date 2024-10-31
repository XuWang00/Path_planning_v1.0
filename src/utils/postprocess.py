import pymeshlab


def remove_isolated_pieces(mesh_file, output_file, min_component_size=25):
    """
    移除网格中的孤立面片。

    参数:
    - mesh_file: 输入网格文件路径。
    - output_file: 清理后的网格输出文件路径。
    - min_component_size: 定义孤立组件的最小尺寸，低于此值的组件将被移除。
    """
    # 加载网格
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)

    # 应用移除孤立面片的过滤器
    ms.apply_filter('meshing_remove_connected_component_by_face_number',
                    mincomponentsize=min_component_size,
                    removeunref=True)
    # 在保存前，移除未引用的顶点
    ms.apply_filter('meshing_remove_unreferenced_vertices')
    # 保存清理后的网格
    ms.save_current_mesh(output_file)


if __name__ == "__main__":
    input_mesh = "D:\\PATH_PLANNING\\pp01\\models\\1upprocessed.obj"  # 输入网格文件路径
    output_mesh = "D:\\PATH_PLANNING\\pp01\\models\\output_mesh.obj"  # 清理后的输出文件路径
    remove_isolated_pieces(input_mesh, output_mesh, min_component_size=25)
