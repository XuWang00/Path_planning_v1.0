import open3d as o3d
import numpy as np



def get_bbox(objmesh):
    aabb = objmesh.get_axis_aligned_bounding_box()
    # Set the bounding box color
    aabb.color = (1, 0, 0)  #  Red
    # Get the oriented bounding box (OBB) of the model
    obb = objmesh.get_oriented_bounding_box()
    # Set the bounding box color
    obb.color = (0, 1, 0)  # Green
    return aabb, obb

def bbox_dimensions(bbox):
    """
    Calculate and return the dimensions of a given bounding box, where length is the larger dimension on the XY plane.

    Parameters:
    - bbox: The bounding box object.

    Returns:
    - length, width, height: The dimensions of the bounding box.
    """
    # Calculate the dimensions of the bounding box in the X, Y, Z axes direction
    x_size = np.abs(bbox.get_max_bound()[0] - bbox.get_min_bound()[0])
    y_size = np.abs(bbox.get_max_bound()[1] - bbox.get_min_bound()[1])
    height = np.abs(bbox.get_max_bound()[2] - bbox.get_min_bound()[2])

    # Determine length and width
    length, width = max(x_size, y_size), min(x_size, y_size)

    # Print dimension information (or otherwise display)
    print(f"Length: {length:.2f}, Width: {width:.2f}, Height: {height:.2f}")
    max_dimension = max(length, width, height)
    print(f"The maximum dimension is: {max_dimension:.2f}")
    return max_dimension, length, width, height


def compute_object_center(bbox):
    # Calculate the object's center coordinates
    center = bbox.get_center()

    # Calculate the base center coordinates (assuming the base is at the minimum Z value)
    base_center = np.array(center)  # Create a copy of the center coordinates
    base_center[2] = bbox.get_min_bound()[2]  # Set the Z coordinate to the bounding box's minimum Z value

    return center, base_center


def create_hemisphere(radius, center, resolution=20):
    """
    Create an upper hemisphere mesh.
    Parameters:
    - radius: The radius of the sphere.
    - center: The center coordinates of the sphere.
    - resolution: The resolution of the sphere, where a larger value means more triangles.
    Returns:
    - hemisphere: The mesh of the upper hemisphere.
    """
    # Create a complete sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.translate(center)

    # Find all vertices where z > 0
    vertices = np.asarray(sphere.vertices)
    faces = np.asarray(sphere.triangles)
    valid_faces = []

    for face in faces:
        ## If all vertices of a face have a z coordinate > 0, keep the face
        if np.all(vertices[face, 2] > center[2]):
            valid_faces.append(face)

    # Create a new hemisphere mesh from valid faces
    hemisphere = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(valid_faces)
    )

    hemisphere.compute_vertex_normals()  # Recalculate vertex normals

    return hemisphere


def generate_ray(theta, phi, length=1000):
    """
    生成并返回一条射线的LineSet，射线从原点发出，具有指定的天顶角和方位角。

    参数:
    - theta_deg: 天顶角，以度为单位。
    - phi_deg: 方位角，以度为单位。
    - length: 射线的长度。

    返回:
    - lineset: 一个Open3D LineSet对象，表示射线。
    """
    # # 将角度转换为弧度
    # theta = np.deg2rad(theta_deg)
    # phi = np.deg2rad(phi_deg)

    # 计算方向向量
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    # 计算射线的终点
    end_point = np.array([dx, dy, dz]) * length

    # 创建LineSet
    points = [np.array([0, 0, 0]), end_point]  # 射线的起点和终点
    lines = [[0, 1]]  # 线段的索引
    colors = [[1, 0, 0]]  # 射线的颜色（红色）

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set




def create_line_set3(eye_center, eye_left, eye_right):
    """
    Creates a LineSet object in Open3D to visualize lines connecting the given three points:
    eye_center, eye_left, and eye_right.

    Parameters:
    - eye_center: array-like, the coordinates of the eye center.
    - eye_left: array-like, the coordinates of the left eye position.
    - eye_right: array-like, the coordinates of the right eye position.

    Returns:
    - o3d.geometry.LineSet: An Open3D LineSet object that visualizes the lines.
    """
    # Create an array of points
    points = np.array([eye_left, eye_center, eye_right])

    # Define the lines based on the indices of the points array
    # Connect eye_left to eye_center, and eye_center to eye_right
    lines = [[0, 1], [1, 2]]  # Indices in the points array

    # Create a LineSet object from Open3D
    line_set = o3d.geometry.LineSet()

    # Set the points
    line_set.points = o3d.utility.Vector3dVector(points)

    # Set the lines
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optionally, set colors for each line
    colors = [[1, 1, 0], [1, 1, 0]]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_line_set2(eye_center, eye_left, eye_right, center):
    """
    Creates a LineSet object in Open3D to visualize lines connecting the given three points:
    eye_center, eye_left, and eye_right.

    Parameters:
    - eye_center: array-like, the coordinates of the eye center.
    - eye_left: array-like, the coordinates of the left eye position.
    - eye_right: array-like, the coordinates of the right eye position.

    Returns:
    - o3d.geometry.LineSet: An Open3D LineSet object that visualizes the lines.
    """
    # Create an array of points
    points = np.array([eye_center, eye_left, eye_right, center])

    # Define the lines based on the indices of the points array
    # Connect eye_left to eye_center, and eye_center to eye_right
    lines = [[0, 3], [1, 3], [2, 3]]  # Indices in the points array

    # Create a LineSet object from Open3D
    line_set = o3d.geometry.LineSet()

    # Set the points
    line_set.points = o3d.utility.Vector3dVector(points)

    # Set the lines
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optionally, set colors for each line
    colors = [[1, 1, 0],[1, 1, 0],[1, 1, 0]]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_point(center, color=[1, 0, 1], radius=5):
    """
    创建一个代表点的小球（Sphere）。

    参数:
    - center: 点的坐标，形式为[x, y, z]。
    - color: 球体的颜色，形式为[R, G, B]，默认为红色。
    - radius: 球体的半径，默认为0.05。

    返回:
    - sphere: Open3D的TriangleMesh对象，表示一个小球。
    """
    # 创建一个小球来表示点
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)  # 将球体移动到指定的位置

    # 设置球体的颜色
    sphere.paint_uniform_color(color)  # 设置统一的颜色

    return sphere
