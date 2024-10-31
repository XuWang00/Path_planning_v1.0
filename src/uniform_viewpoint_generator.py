import open3d as o3d
import numpy as np


def generate_viewpoints_on_hemisphere(radius, base_center, angle_step=20):
    """
    Generates viewpoints and corresponding view directions on the surface of a hemisphere.

    Parameters:
    - radius: Radius of the hemisphere.
    - base_center: Center coordinates of the hemisphere's base.
    - angle_step: The angular interval (in degrees) for generating points.

    Returns:
    - viewpoints: A list of generated viewpoints and view directions on the hemisphere surface.
                  Each viewpoint is a tuple: (point coordinates, view direction).
    """

    viewpoints = []
    # Convert angle to radians
    angle_step_rad = np.deg2rad(angle_step)
    print(base_center)

    # Add the zenith point and its view direction
    zenith_point = np.array([base_center[0], base_center[1], radius])
    direction = base_center - zenith_point
    direction_normalized = direction / np.linalg.norm(direction)
    viewpoints.append((zenith_point, direction_normalized))

    # Iterate over zenith and azimuth angles
    for theta in np.arange(angle_step_rad, 4*np.pi /9, angle_step_rad):  # 天顶角
        for phi in np.arange(0, 2 * np.pi, angle_step_rad):  # 方位角
            x = radius * np.sin(theta) * np.cos(phi)+ base_center[0]
            y = radius * np.sin(theta) * np.sin(phi)+ base_center[1]
            z = radius * np.cos(theta)+ base_center[2]
            point = np.array([x, y, z])

            # Calculate view direction: vector pointing from point to base center
            direction = base_center - point
            # Normalize direction vector
            direction_normalized = direction / np.linalg.norm(direction)

            viewpoints.append((point, direction_normalized))

    return viewpoints



def generate_viewpoints_on_hemiellipsoid(a, b, c, base_center, angle_step=20):
    """
    Generates viewpoints and corresponding view directions on the surface of a hemisphere of an ellipsoid.

    Parameters:
    - a, b, c: Semi-major axis lengths of the ellipsoid along the x, y, and z axes, respectively.
    - base_center: Center coordinates of the ellipsoid's base.
    - angle_step: The angular interval (in degrees) for generating points.

    Returns:
    - viewpoints: A list of generated viewpoints and view directions on the ellipsoid surface.
                  Each viewpoint is a tuple: (point coordinates, view direction).
    """

    viewpoints = []
    # Convert angle to radians
    angle_step_rad = np.deg2rad(angle_step)

    # Add the zenith point and its view direction
    zenith_point = np.array([base_center[0], base_center[1], c])
    direction = base_center - zenith_point
    direction_normalized = direction / np.linalg.norm(direction)
    viewpoints.append((zenith_point, direction_normalized))


    # Iterate over zenith and azimuth angles
    for theta in np.arange(angle_step_rad, np.pi / 2, angle_step_rad):  # Zenith angle
        for phi in np.arange(0, 2 * np.pi, angle_step_rad):  # Azimuth angle
            x = a * np.sin(theta) * np.cos(phi) + base_center[0]
            y = b * np.sin(theta) * np.sin(phi) + base_center[1]
            z = c * np.cos(theta) + base_center[2]
            point = np.array([x, y, z])

            # Calculate view direction: vector pointing from point to base center
            direction = base_center - point
            # Normalize direction vector
            direction_normalized = direction / np.linalg.norm(direction)

            viewpoints.append((point, direction_normalized))

    return viewpoints




def generate_viewpoints_on_cylinder(radius, height, base_center, angle_step=30, height_step=300, top_step=400):
    """
    Generates viewpoints and corresponding view directions on the side surface of a cylinder.

    Parameters:
    - radius: Radius of the cylinder.
    - height: Height of the cylinder.
    - base_center: Center coordinates of the cylinder's base.
    - angle_step: The angular interval (in degrees) between viewpoints.
    - height_step: The vertical interval between viewpoints.

    Returns:
    - viewpoints: A list of generated viewpoints and view directions on the cylinder surface.
                  Each viewpoint is a tuple: (point coordinates, view direction).
    """

    viewpoints = []
    angle_step_rad = np.deg2rad(angle_step)

    # Add the zenith point and its view direction
    zenith_point = np.array([base_center[0], base_center[1], height])
    direction = base_center - zenith_point
    direction_normalized = direction / np.linalg.norm(direction)
    viewpoints.append((zenith_point, direction_normalized))


    for l in np.arange(top_step, radius, top_step):
        for alpha in np.arange(0, 2 * np.pi, angle_step_rad):
            x = l * np.cos(alpha) + base_center[0]
            y = l * np.sin(alpha) + base_center[1]
            z = height
            point = np.array([x, y, z])
            direction = np.array([0, 0, -1])
            viewpoints.append((point, direction))


    # Start from the top of the cylinder, generating viewpoints down the side
    for z in np.arange(height, 0, -height_step):
        for theta in np.arange(0, 2 * np.pi, angle_step_rad):
            x = radius * np.cos(theta) + base_center[0]
            y = radius * np.sin(theta) + base_center[1]
            point = np.array([x, y, z])

            # Direction vector points towards the center line of the cylinder at the current height
            direction = np.array([0, 0, z]) - point
            direction_normalized = direction / np.linalg.norm(direction)

            viewpoints.append((point, direction_normalized))


    return viewpoints



def generate_viewpoints_on_cylinder2(radius, height, base_center, angle_step=30, height_step=300, top_step=400):
    """
    Generates viewpoints and corresponding view directions on the side surface of a cylinder.

    Parameters:
    - radius: Radius of the cylinder.
    - height: Height of the cylinder.
    - base_center: Center coordinates of the cylinder's base.
    - angle_step: The angular interval (in degrees) between viewpoints.
    - height_step: The vertical interval between viewpoints.

    Returns:
    - viewpoints: A list of generated viewpoints and view directions on the cylinder surface.
                  Each viewpoint is a tuple: (point coordinates, view direction).
    """

    viewpoints = []
    angle_step_rad = np.deg2rad(angle_step)

    # Add the zenith point and its view direction
    zenith_point = np.array([base_center[0], base_center[1], height])
    direction = base_center - zenith_point
    direction_normalized = direction / np.linalg.norm(direction)
    viewpoints.append((zenith_point, direction_normalized))

    ##The viewpoints on the top surface
    for l in np.arange(top_step, radius, top_step):
        for alpha in np.arange(0, 2 * np.pi, angle_step_rad):
            x = l * np.cos(alpha) + base_center[0]
            y = l * np.sin(alpha) + base_center[1]
            z = height
            point = np.array([x, y, z])
            # Calculate view direction: vector pointing from point to base center
            direction = base_center - point
            # Normalize direction vector
            direction_normalized = direction / np.linalg.norm(direction)
            viewpoints.append((point, direction_normalized))


    # Start from the top of the cylinder, generating viewpoints down the side
    for z in np.arange(height-400, 0, -height_step):
        for theta in np.arange(0, 2 * np.pi, angle_step_rad):
            x = radius * np.cos(theta) + base_center[0]
            y = radius * np.sin(theta) + base_center[1]
            point = np.array([x, y, z])

            # Direction vector points towards the center line of the cylinder at the current height
            direction = np.array([0, 0, z]) - point
            direction_normalized = direction / np.linalg.norm(direction)

            viewpoints.append((point, direction_normalized))


    return viewpoints




def generate_viewpoints_on_elliptical_cylinder(a, b, height, base_center, angle_step=30, height_step=300, top_step=200):
    """
    Generates viewpoints and corresponding view directions on the side surface of an elliptical cylinder.

    Parameters:
    - a: Semi-major axis of the ellipse along the x-axis.
    - b: Semi-minor axis of the ellipse along the y-axis.
    - height: Height of the cylinder.
    - base_center: Center coordinates of the cylinder's base.
    - angle_step: The angular interval (in degrees) between viewpoints.
    - height_step: The vertical interval between viewpoints.

    Returns:
    - viewpoints: A list of generated viewpoints and view directions on the elliptical cylinder surface.
                  Each viewpoint is a tuple: (point coordinates, view direction).
    """

    viewpoints = []
    angle_step_rad = np.deg2rad(angle_step)

    # Directly add the center point of the top surface
    top_center = np.array([base_center[0], base_center[1], height + base_center[2]])
    direction = base_center - top_center
    direction_normalized = direction / np.linalg.norm(direction)
    viewpoints.append((top_center, direction_normalized))

    # Viewpoints on the top surface, excluding the center by starting from l > 0
    for l in np.linspace(1 / (a / top_step), 1, int(a / top_step) - 1):  # Exclude the first step to skip the center
        for alpha in np.arange(0, 2 * np.pi, angle_step_rad):
            x = l * a * np.cos(alpha) + base_center[0]
            y = l * b * np.sin(alpha) + base_center[1]
            z = height + base_center[2]
            point = np.array([x, y, z])
            direction = base_center - point
            direction_normalized = direction / np.linalg.norm(direction)
            viewpoints.append((point, direction_normalized))


    # Generate viewpoints along the height of the elliptical cylinder
    for z in np.arange(height-height_step, 0, -height_step):
        for theta in np.arange(0, 2 * np.pi, angle_step_rad):
            x = a * np.cos(theta) + base_center[0]
            y = b * np.sin(theta) + base_center[1]
            point = np.array([x, y, z])

            # Direction vector points towards the base center at the same height
            direction = np.array([base_center[0], base_center[1], z]) - point
            direction_normalized = direction / np.linalg.norm(direction)

            viewpoints.append((point, direction_normalized))



    return viewpoints



def visualize_viewpoints_as_arrows(viewpoints, arrow_color, arrow_length=20 ):
    """
        Create and visualize arrows based on a set of viewpoints.

        Parameters:
        - viewpoints: A collection of viewpoints, each including coordinates and a direction vector.
        - arrow_length: The length of the arrows.

        Returns:
        - arrow_list: A list of the created arrows.
        """
    arrow_list = []
    for point, direction, _ in viewpoints:
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=2, cone_radius=4,
                                                       cylinder_height=arrow_length * 0.8,
                                                       cone_height=arrow_length * 0.2)

        # Check if the direction is a special case of being perpendicular to the z-axis
        if np.allclose(direction, [0, 0, 1]):
            # No rotation needed
            R = np.eye(3)
        elif np.allclose(direction, [0, 0, -1]):
            # 180 degrees rotation around the x-axis
            R = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])
        else:
            # General case, calculate rotation
            arrow_frame = np.array([[0, 0, 1],  # Arrow's default direction
                                    [0, 1, 0],
                                    [1, 0, 0]])
            target_frame = np.column_stack((direction, np.cross([0, 0, 1], direction), [0, 0, 1]))
            R = target_frame @ np.linalg.inv(arrow_frame)

        # Apply rotation and translation
        arrow.rotate(R, center=arrow.get_center())
        arrow.translate(point)

        # Set the arrow color
        arrow.paint_uniform_color(arrow_color)

        arrow_list.append(arrow)

    return arrow_list
