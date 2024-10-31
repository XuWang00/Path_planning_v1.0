import numpy as np


def generate_viewpoints_on_sphere1(radius, center, angle_step_large=45, angle_step_small=15):
    """
    Generates viewpoints and corresponding view directions on the surface of a sphere with varying density.

    Parameters:
    - radius: Radius of the sphere.
    - center: Center coordinates of the sphere.
    - angle_step_large: The angular interval (in degrees) for generating points in sparse regions (top and bottom).
    - angle_step_small: The angular interval (in degrees) for generating points in dense regions (middle).

    Returns:
    - viewpoints: A list of generated viewpoints, view directions, and view categories on the sphere surface.
                  Each item is a tuple: (point coordinates, view direction, view category).
    """
    viewpoints = []
    # Add special cases for zenith and nadir points
    viewpoints.append((center + np.array([0, 0, radius]), np.array([0, 0, -1]), 'Topview'))
    viewpoints.append((center + np.array([0, 0, -radius]), np.array([0, 0, 1]), 'Bottomview'))

    # Convert angles to radians
    angle_step_rad_large = np.deg2rad(angle_step_large)
    angle_step_rad_small = np.deg2rad(angle_step_small)

    # Generate viewpoints in different regions with different densities
    for theta in np.arange(angle_step_rad_small, np.pi, angle_step_rad_small):
        if theta <= np.deg2rad(45) or theta > np.deg2rad(135):
            angle_step_rad_phi = angle_step_rad_large
        else:
            angle_step_rad_phi = angle_step_rad_small

        for phi in np.arange(0, 2 * np.pi, angle_step_rad_phi):
            x = radius * np.sin(theta) * np.cos(phi) + center[0]
            y = radius * np.sin(theta) * np.sin(phi) + center[1]
            z = radius * np.cos(theta) + center[2]
            point = np.array([x, y, z])
            direction = center - point
            direction_normalized = direction / np.linalg.norm(direction)

            # Determine the view category
            view = classify_viewpoint_by_angle(theta, phi)
            viewpoints.append((point, direction_normalized, view))

    return viewpoints


def generate_viewpoints_on_ellipsoid1(a, b, c, center, angle_step_phi_large=20, angle_step_phi_small=15, angle_step_theta=10):
    """
    Generates viewpoints and corresponding view directions on the surface of an ellipsoid with varying density.

    """
    viewpoints = []
    # Convert angles to radians
    angle_step_rad_large = np.deg2rad(angle_step_phi_large)
    angle_step_rad_small = np.deg2rad(angle_step_phi_small)
    angle_step_rad_theta = np.deg2rad(angle_step_theta)

    # Add special cases for zenith and nadir points
    # 为顶点和底点设置合适的视点方向
    viewpoints.append((center + np.array([0, 0, c]), np.array([0, 0, -1]), 'Topview'))
    viewpoints.append((center + np.array([0, 0, -c]), np.array([0, 0, 1]), 'Bottomview'))

    # Iterate over zenith angles
    for theta in np.arange(angle_step_rad_theta, np.pi, angle_step_rad_theta):
        if theta <= np.deg2rad(45) or theta > np.deg2rad(135):
            angle_step_rad_phi = angle_step_rad_large
        else:
            angle_step_rad_phi = angle_step_rad_small

        for phi in np.arange(0, 2 * np.pi, angle_step_rad_phi):  # Azimuth angle
            x = a * np.sin(theta) * np.cos(phi) + center[0]
            y = b * np.sin(theta) * np.sin(phi) + center[1]
            z = c * np.cos(theta) + center[2]
            point = np.array([x, y, z])

            # Calculate view direction: vector pointing from point to base center
            direction = center - point
            # Normalize direction vector
            direction_normalized = direction / np.linalg.norm(direction)

            # Classify the viewpoint based on angle
            view = classify_viewpoint_by_angle(theta, phi)
            viewpoints.append((point, direction_normalized, view))

    return viewpoints

def generate_viewpoints_on_longitude_line(a, b, c, center, fixed_phi=np.pi/2):
    """
    Generates viewpoints on the surface of an ellipsoid along a single meridian line.

    Parameters:
    - a, b, c: Semi-major axis lengths of the ellipsoid along the x, y, and z axes, respectively.
    - center: Center coordinates of the ellipsoid.
    - fixed_phi: Fixed azimuth angle in radians to generate viewpoints along a specific meridian.

    Returns:
    - viewpoints: A list of generated viewpoints, view directions, and view categories on the ellipsoid surface.
                  Each item is a tuple: (point coordinates, view direction, view category).
    """
    viewpoints = []

    # # Add special cases for zenith and nadir points with specific view directions
    # viewpoints.append((center + np.array([0, 0, c]), np.array([0, 0, -1]), 'Topview'))
    # viewpoints.append((center + np.array([0, 0, -c]), np.array([0, 0, 1]), 'Bottomview'))

    # Iterate over zenith angles from the pole to the equator and back to the other pole
    for theta in np.arange(0, np.pi, np.deg2rad(5)):  # Increment zenith angle
        x = a * np.sin(theta) * np.cos(fixed_phi) + center[0]
        y = a * np.sin(theta) * np.sin(fixed_phi) + center[1]
        z = a * np.cos(theta) + center[2]
        point = np.array([x, y, z])

        # Calculate view direction: vector pointing from point to base center
        direction = center - point
        direction_normalized = direction / np.linalg.norm(direction)

        # Classify the viewpoint based on angle
        view = classify_viewpoint_by_angle(theta, fixed_phi)
        viewpoints.append((point, direction_normalized, view))

    return viewpoints

def generate_viewpoints_on_latitude_line(a, b, c, center, fixed_theta=45, angle_step_phi=15):
    """
    Generates viewpoints along a fixed latitude line on the surface of an ellipsoid.

    Parameters:
    - a, b, c: Semi-major axis lengths of the ellipsoid along the x, y, and z axes, respectively.
    - center: Center coordinates of the ellipsoid.
    - fixed_theta: Fixed zenith angle in radians to generate viewpoints along a specific latitude.
    - angle_step_phi: The angular interval (in degrees) for generating points along the latitude line.

    Returns:
    - viewpoints: A list of generated viewpoints, view directions, and view categories on the ellipsoid surface.
                  Each item is a tuple: (point coordinates, view direction, view category).
    """
    viewpoints = []
    fixed_theta_rad = np.deg2rad(fixed_theta)  # Convert fixed zenith angle to radians

    # Iterate over azimuth angles to generate points along the latitude
    for phi in np.arange(0, 2 * np.pi, np.deg2rad(angle_step_phi)):
        x = a * np.sin(fixed_theta_rad) * np.cos(phi) + center[0]
        y = a * np.sin(fixed_theta_rad) * np.sin(phi) + center[1]
        z = a * np.cos(fixed_theta_rad) + center[2]
        point = np.array([x, y, z])

        # Calculate view direction: vector pointing from point to base center
        direction = center - point
        direction_normalized = direction / np.linalg.norm(direction)

        # Classify the viewpoint based on angle
        view = classify_viewpoint_by_angle(fixed_theta_rad, phi)
        viewpoints.append((point, direction_normalized, view))

    return viewpoints

def classify_viewpoint_by_angle(theta, phi):

    """
    Classify the viewpoint based on theta and phi angles.

    Parameters:
    - theta: Zenith angle in radians.
    - phi: Azimuthal angle in radians.

    !!! Here theta, phi is used to describe the position of viewpoint, not direction.

    Returns:
    - A string indicating the category of the viewpoint.
    """
    if theta <= np.deg2rad(45):
        return 'Topview'
    elif theta > np.deg2rad(135):
        return 'Bottomview'
    elif 0 <= phi < np.pi / 2:
        return 'Frontview'
    elif np.pi / 2 <= phi < np.pi:
        return 'Rightview'
    elif np.pi <= phi < 3 * np.pi / 2:
        return 'Backview'
    else:
        return 'Leftview'


def count_viewpoints_by_view(viewpoints):
    """
    Counts the number of viewpoints in each view and prints the count.

    Parameters:
    - viewpoints: A list of viewpoints, where each viewpoint is a tuple containing the point coordinates,
      view direction, and view category.
    """
    # Initialize a dictionary to hold the count of viewpoints for each view
    view_counts = {
        'Topview': 0,
        'Bottomview': 0,
        'Frontview': 0,
        'Leftview': 0,
        'Backview': 0,
        'Rightview': 0
    }

    # Iterate through the viewpoints and increment the count for the corresponding view
    for _, _, view in viewpoints:
        if view in view_counts:
            view_counts[view] += 1

    # Print the count of viewpoints for each view
    for view, count in view_counts.items():
        print(f"{view} view has {count} viewpoints.")



def filter_viewpoints_by_z(viewpoints, z_min):
    """
    过滤掉z坐标小于指定值的视点。

    参数:
    - viewpoints: 视点列表，每个视点为一个元组(point, direction, view)。
    - z_min: z坐标的最小值，默认为50。

    返回:
    - filtered_viewpoints: 过滤后的视点列表。
    """
    filtered_viewpoints = []
    for point, direction, view in viewpoints:
        if point[2] >= z_min:
            filtered_viewpoints.append((point, direction, view))
    return filtered_viewpoints




def classify_mesh_by_normal(theta, phi):
    """
    Classify the viewpoint based on theta and phi angles.

    Parameters:
    - theta: Zenith angle in radians.
    - phi: Azimuthal angle in radians.

    Returns:
    - A string indicating the category of the viewpoint.
    """
    if theta <= np.deg2rad(45):
        return 'Topview'
    elif theta > np.deg2rad(135):
        return 'Bottomview'
    elif 0 <= phi < np.pi / 2:
        return 'Frontview'
    elif np.pi / 2 <= phi < np.pi:
        return 'Rightview'
    elif np.pi <= phi < 3 * np.pi / 2:
        return 'Backview'
    else:
        return 'Leftview'



def analyze_mesh_by_view(mesh):
    """
    分析给定mesh内的每个三角面片，计算法线向量，面积，以及面片所对应的视图分类。

    参数:
    - mesh: 需要分析的mesh，假设已经包含了法线。

    返回:
    - view_stats: 每个视图分类的面片数量及面积总和的字典。
    """
    # 确保mesh有法线
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # 获取三角形面片的中心点
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_normals = np.asarray(mesh.triangle_normals)

    triangle_centers = np.mean(vertices[triangles], axis=1)

    # 计算每个面片的面积
    triangle_areas = np.linalg.norm(np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                                             vertices[triangles[:, 2]] - vertices[triangles[:, 0]]), axis=1) * 0.5

    view_stats = {'Topview': {'count': 0, 'area': 0},
                  'Bottomview': {'count': 0, 'area': 0},
                  'Frontview': {'count': 0, 'area': 0},
                  'Backview': {'count': 0, 'area': 0},
                  'Leftview': {'count': 0, 'area': 0},
                  'Rightview': {'count': 0, 'area': 0}}

    # 分析每个三角形面片
    for i, normal in enumerate(triangle_normals):
        view = classify_mesh_by_normal(np.arccos(normal[2]), np.arctan2(normal[1], normal[0]))

        if view in view_stats:
            view_stats[view]['count'] += 1
            view_stats[view]['area'] += triangle_areas[i]

    print("View Statistics:")
    for view, stats in view_stats.items():
        print(f"{view}: Count = {stats['count']}, Total Area = {stats['area']:.2f}")

    return view_stats



def filter_viewpoints_by_area(viewpoints, view_stats, area_threshold):
    """
    根据视图的总面积阈值来过滤视点。

    参数:
    - viewpoints: 包含视点、方向和视图分类的列表。每个元素是(point, direction, view)的元组。
    - view_stats: 包含视图分类的面片数量及面积总和的字典。
    - area_threshold: 总面积阈值。

    返回:
    - filtered_viewpoints: 过滤后的视点列表。
    """
    filtered_viewpoints = []
    for point, direction, view in viewpoints:
        # 如果该视图的总面积大于或等于阈值，则保留该视点
        if view_stats[view]['area'] >= area_threshold:
            filtered_viewpoints.append((point, direction, view))

    return filtered_viewpoints


def filter_viewpoints_by_name(viewpoints, target_view):
    """
    根据指定的视图名称来过滤视点。

    参数:
    - viewpoints: 包含视点、方向和视图分类的列表。每个元素是(point, direction, view)的元组。
    - target_view: 要保留的视图名称，如'Topview', 'Bottomview', 等。

    返回:
    - filtered_viewpoints: 过滤后的视点列表。
    """
    filtered_viewpoints = []
    for point, direction, view in viewpoints:
        # 如果该视图的是俯视图，则保留
        if view == target_view:
            filtered_viewpoints.append((point, direction, view))

    return filtered_viewpoints


