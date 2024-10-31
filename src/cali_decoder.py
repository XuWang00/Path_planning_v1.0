import numpy as np

# 定义旋转矩阵和平移向量
T_right = np.array([-1.692644680264465, -7.159189314620370, 566.128796138577510])
T_left = np.array([0.429104752273365, 1.405577935741759, 558.729515415347240])
R_right = np.array([0.980340282201326, -0.009485668325607, -0.197086156768659,
                    -0.014418077023466, -0.999617344658513, -0.023606848853878,
                    -0.196786813959598, 0.025984348255843, -0.980101914852404]).reshape(3,3)
R_left = np.array([0.984952136328966, -0.012650740756177, 0.172363708185124,
                   -0.006088375048936, -0.999238186078382, -0.038548400382477,
                   0.172720064932451, 0.036918914408529, -0.984278808533754]).reshape(3,3)

# 计算相对位置
delta_T = T_right - T_left
distance = np.linalg.norm(delta_T)
print("Relative Position (Vector):", delta_T)
print("Distance between cameras:", distance)

# 计算相对角度
R_relative = R_left.T @ R_right
print("Relative Rotation Matrix:", R_relative)

# 计算角度（这里简化为使用arccos提取Z轴旋转，对简单场景足够）
angle_z = np.arccos(R_relative[0, 0]) * (180 / np.pi)  # 从旋转矩阵的对角元素提取角度
print("Relative Rotation Angle about Z-axis:", angle_z)

def rotation_matrix_to_euler_angles(R):
    assert(R.shape == (3, 3))

    sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) * (180 / np.pi)  # Convert to degrees




# 使用示例
R_relative = R_left.T @ R_right
euler_angles = rotation_matrix_to_euler_angles(R_relative)
print("Euler anglesZYX (Roll, Pitch, Yaw):", euler_angles)
