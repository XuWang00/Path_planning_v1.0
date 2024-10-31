import open3d as o3d
import numpy as np

def create_ground_plane(size=30, height=0.01, color=[0.5, 0.5, 0.5]):
    """Create a large plane to represent the ground."""
    # Create a mesh plane
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=height)

    # Move the plane up along the Y-axis so that the top surface center aligns with (0, 0, 0)
    plane_mesh.translate((0, -height / 2, 0))

    # Reset the plane's origin to its geometric center
    plane_mesh.translate((-size / 2, -size / 2, 0))

    # Set the plane's color
    plane_mesh.paint_uniform_color(color)

    return plane_mesh





def coordinate():
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])

    return coordinate_frame

def axis(base_center):
    # Add a line segment to represent the model's height
    scale=1500
    points = [base_center, [base_center[0], base_center[1], scale]]  # The start and end points of the line
    lines = [[0, 1]]  # The indices of the line
    colors = [[0, 1, 1] for i in range(len(lines))]  # The color of the line
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


