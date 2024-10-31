import numpy as np
import csv
import re

def parse(log_text):
    # 正则表达式，用于匹配日志中的视点信息
    pattern = re.compile(
         # r"Viewpoint (\d+): Position \[([\d\.\s\-e]+)\], Direction \[([\d\.\s\-e]+)\], View (\w+), Score ([\d\.]+)"
         r"Viewpoint (\d+): Position \[([-\d\.eE+\s]+)\], Direction \[([-\d\.eE+\s]+)\], View (\w+), Score ([\d\.]+)"

    )

    # 解析日志文本并构建目标列表
    viewpoints = []
    for match in re.finditer(pattern, log_text):
        index = int(match.group(1))
        position = [float(x) for x in match.group(2).split()]
        direction = [float(x) for x in match.group(3).split()]
        view = match.group(4)
        score = float(match.group(5))

        viewpoints.append({
            'index': index,
            'position': position,
            'direction': direction,
            'view': view,
            'score': score
        })

    return viewpoints

def calculate_rotation_angle(position):
    """
    计算从视点位置到X轴正方向的旋转角度。

    参数:
    position: 视点的XY平面坐标，形式为[x, y]。

    返回:
    gamma: 旋转角度，单位为弧度。
    """
    x, y = position[0], position[1]
    gamma = np.arctan2(y, x)  # atan2返回的是[-pi, pi]的范围，适合这里的旋转需求
    if gamma > 0 :
        gamma = 2*np.pi - gamma
    elif gamma < 0 :
        gamma = -gamma
    return gamma  # 转台逆时针旋转



def rotate_around_z(point, angle):
    """ 绕 z 轴旋转点或向量 """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, point)


def adjust_viewpoints(viewpoints):
    adjusted_viewpoints = []
    for vp in viewpoints:
        position = np.array(vp['position'])
        direction = np.array(vp['direction'])

        # 计算旋转到 x<0 的 xz 平面的角度
        angle_to_rotate = calculate_rotation_angle(position)

        # # 旋转视点位置和方向
        new_position = rotate_around_z(position, angle_to_rotate)
        # new_direction = rotate_around_z(direction, angle_to_rotate)
        new_direction = angle(direction)

        angle_to_rotate = np.rad2deg(angle_to_rotate)
        # 存储调整后的视点信息
        adjusted_viewpoints.append({
            'index': vp['index'],
            'new_position': new_position.tolist(),
            'new_direction': new_direction.tolist(),
            'rotation_angle': angle_to_rotate,
            'score': vp['score']
        })
    return adjusted_viewpoints




def reverse_direction_vector(direction):
    """
    计算当前方向向量的反向方向向量。

    参数:
    direction (list of float): 当前方向向量，例如 [-0.9858526920681038, 3.3428285139969205e-10, -0.16761407]

    返回:
    list of float: 反向方向向量
    """
    return [-component for component in direction]

def calculate_rotation_around_y(direction):
    """
    计算绕y轴旋转使z轴对齐到给定方向的角度。

    参数:
    direction: 新的方向向量，格式为 [dx, dy, dz]。

    返回:
    rotation_angle_y: 绕y轴的旋转角度，单位为度。

    # 示例视点方向向量
    direction = [-0.351206563, 0.0, -0.936298003]
    """
    # 全局Z轴向量
    global_z = np.array([0, 0, 1])

    # # 计算新方向向量的归一化
    # direction_normalized = np.array(direction)
    # direction_normalized /= np.linalg.norm(direction_normalized)

    reversed_direction = reverse_direction_vector(direction)
    # 计算夹角的余弦值
    cos_theta = np.dot(global_z, reversed_direction)

    # 通过余弦值计算夹角（转换为度）
    angle = np.degrees(np.arccos(cos_theta))

    # 确定旋转的方向（通过观察X分量的符号）
    # 如果dx为正，则顺时针旋转；如果dx为负，则逆时针旋转
    if reversed_direction[0] > 0:
        rotation_angle_y = -angle
    else:
        rotation_angle_y = angle

    return rotation_angle_y



def angle(vector):
    # # 定义两个点的坐标
    # point1 = np.array([74.60128028085659, -4.973799150320701e-14, 626.90897643])
    # # point2 = np.array([-38.53177982, -64.80434153, 364.32442822])
    # point2 = np.array([0.0,          0.0,         108.13248452])
    # # 计算两点间的向量
    # vector = point1 - point2

    # vector = [-0.999999999930507, -5.470533470663952e-10, -4.8964453e-17]

    # 定义z轴向量
    z_axis = np.array([0, 0, -1])

    # 计算两向量的点积
    dot_product = np.dot(vector, z_axis)

    # 计算两向量的模长
    vector_norm = np.linalg.norm(vector)
    z_axis_norm = np.linalg.norm(z_axis)

    # 计算两向量的夹角（弧度）
    angle_radians = np.arccos(dot_product / (vector_norm * z_axis_norm))

    # 转换为度数
    angle_degrees = np.degrees(angle_radians)
    # print(angle_degrees)
    return angle_degrees



def formatting(input_string):

    # data = [
    #     [-195.546327, 6.582990, 97.926672, 59.601065, -19.195523, 54.379302, 44.15582975609704],
    #     [-195.570071, 6.504626, 97.814397, 59.809760, -19.128578, 54.175970, 59.26348837923722],
    #     [-195.587481, 6.447121, 97.727765, 59.990443, -19.074605, 53.999474, 285.42756377841005],
    #     [-191.942798, 16.081291, 113.215119, 40.280802, -30.913023, 73.030300, 120.73651161974331],
    #     [-191.768364, 16.465545, 113.709146, 40.017150, -31.331940, 73.242900, 194.58321024765235],
    #     [-187.098895, 23.976452, 127.329161, 32.929311, -45.188966, 79.422789, 224.1558297562074],
    #     [-178.626817, -2.503680, 85.142020, -156.886808, -108.146750, 172.780957, 290.83347948941423],
    #     [-179.358427, -3.133680, 84.555217, -156.845745, -108.539475, 172.130331, 80.29279565975882],
    #     [-153.883176, 4.894389, 116.363915, -173.017706, -70.030475, 190.843644, 330.2847984216172],
    #     [-149.128138, 0.180285, 122.537056, -180.000000, -57.643229, 198.071862, 290.83347948926485],
    #     [-148.651187, 0.232434, 122.589003, -180.000000, -57.643430, 198.548813, 199.17077417650114],
    #     [-147.604619, 0.333676, 122.689710, -180.000000, -57.643966, 199.595381, 150.28479842150836],
    #     [-149.577683, 0.127675, 122.484597, -180.000000, -57.643078, 197.622317, 100.73651161861447],
    #     [-149.577683, 0.127675, 122.484597, -180.000000, -57.643078, 197.622317, 39.26348838138554]
    # ]

    data = convert_data(input_string)
    # 四舍五入并添加额外的0
    formatted_data = [[round(value, 2) for value in sublist] + [0, 0] for sublist in data]

    # for line in formatted_data:
    #     print(line)
    return formatted_data

def save_csv(input_string, filename):
    # data = [
    #     [-195.55, 6.58, 97.93, 59.6, -19.2, 54.38, 44.16, 0, 0],
    #     [-195.57, 6.5, 97.81, 59.81, -19.13, 54.18, 59.26, 0, 0],
    #     [-195.59, 6.45, 97.73, 59.99, -19.07, 54.0, 285.43, 0, 0],
    #     [-191.94, 16.08, 113.22, 40.28, -30.91, 73.03, 120.74, 0, 0],
    #     [-191.77, 16.47, 113.71, 40.02, -31.33, 73.24, 194.58, 0, 0],
    #     [-187.1, 23.98, 127.33, 32.93, -45.19, 79.42, 224.16, 0, 0],
    #     [-178.63, -2.5, 85.14, -156.89, -108.15, 172.78, 290.83, 0, 0],
    #     [-179.36, -3.13, 84.56, -156.85, -108.54, 172.13, 80.29, 0, 0],
    #     [-153.88, 4.89, 116.36, -173.02, -70.03, 190.84, 330.28, 0, 0],
    #     [-149.13, 0.18, 122.54, -180.0, -57.64, 198.07, 290.83, 0, 0],
    #     [-148.65, 0.23, 122.59, -180.0, -57.64, 198.55, 199.17, 0, 0],
    #     [-147.6, 0.33, 122.69, -180.0, -57.64, 199.6, 150.28, 0, 0],
    #     [-149.58, 0.13, 122.48, -180.0, -57.64, 197.62, 100.74, 0, 0],
    #     [-149.58, 0.13, 122.48, -180.0, -57.64, 197.62, 39.26, 0, 0],
    # ]
    formatted_data = formatting(input_string)
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(formatted_data)


def convert_data(input_string):
    # 分割输入字符串为每一行数据
    lines = input_string.strip().split('\n')

    # 初始化一个空的列表来存储最终的数据
    data = []

    for line in lines:
        # 分割每一行的数据
        numbers = line.split(',')

        # 将字符串转换为浮点数
        numbers = [float(num) for num in numbers]

        # 将数据添加到data列表中
        data.append(numbers)

    return data

##Step 1

log_text = """
2024-08-06 13:21:15,311 - Pathplanning - INFO - Viewpoint 82: Position [8.45086976e-15 1.33790965e+02 4.91409916e+02], Direction [-1.71818047e-17 -2.72015816e-01 -9.62292781e-01], View Topview, Score 51619.07236872222
2024-08-06 13:21:15,311 - Pathplanning - INFO - Viewpoint 460: Position [-8.87773433e-14 -4.68496410e+02  2.25189210e+02], Direction [ 1.73317816e-16  9.14633978e-01 -4.04282929e-01], View Leftview, Score 25782.826967635883
2024-08-06 13:21:15,311 - Pathplanning - INFO - Viewpoint 475: Position [250.54177099 420.67529118 185.69613172], Direction [-0.48412152 -0.81287029 -0.32383368], View Frontview, Score 16230.586050941729
2024-08-06 13:21:15,311 - Pathplanning - INFO - Viewpoint 415: Position [-4.61801090e+02  5.48241667e-14  2.63106261e+02], Direction [ 8.83378572e-01 -1.04873061e-16 -4.68660111e-01], View Backview, Score 11311.748093022414
2024-08-06 13:21:15,311 - Pathplanning - INFO - Viewpoint 396: Position [430.17017939 -73.53012518 299.1587153 ], Direction [-0.82871731  0.14165484 -0.54144393], View Leftview, Score 7807.7330445726375
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 408: Position [-157.94527507  420.67529118  263.10626149], Direction [ 0.30860731 -0.82195224 -0.47870246], View Rightview, Score 5528.072281344456
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 573: Position [ 402.28120996 -327.22728693  103.19386855], Direction [-0.76552429  0.62269982 -0.16191815], View Leftview, Score 3528.3915371536236
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 282: Position [ 117.23127911 -312.23664305  393.46803862], Direction [-0.23346946  0.62182823 -0.74754375], View Topview, Score 2960.230941487353
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 559: Position [-5.25140824e+02  6.23437421e-14  1.03193869e+02], Direction [ 9.87126368e-01 -1.17189807e-16 -1.59942280e-01], View Backview, Score 2641.575362514666
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 542: Position [517.16275475  88.3999959  103.19386855], Direction [-0.97299131 -0.16631598 -0.16008404], View Frontview, Score 2366.6421840832045
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 419: Position [-353.760159   -287.75884677  263.10626149], Direction [ 0.68337745  0.55587918 -0.47327962], View Backview, Score 1790.6532969454477
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 366: Position [280.77364392 324.37624483 299.1587153 ], Direction [-0.54744679 -0.63246227 -0.54799041], View Frontview, Score 1712.71014529697
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 548: Position [179.60873984 478.37429069 103.19386855], Direction [-0.34672393 -0.92347296 -0.16425654], View Frontview, Score 842.0070301983144
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 252: Position [301.20840232 -51.48634793 419.49076319], Direction [-0.59708258  0.1020609  -0.79566071], View Topview, Score 823.4318964762314
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 375: Position [-334.61299884  272.1839874   299.1587153 ], Direction [ 0.64996013 -0.52869656 -0.54592287], View Rightview, Score 755.7580246832953
2024-08-06 13:21:15,312 - Pathplanning - INFO - Viewpoint 260: Position [117.23127911 312.23664305 393.46803862], Direction [-0.23346946 -0.62182823 -0.74754375], View Topview, Score 597.7359588274459
2024-08-06 13:21:15,313 - Pathplanning - INFO - Viewpoint 239: Position [-234.29854417 -190.58527976  419.49076319], Direction [ 0.46643035  0.37940807 -0.7990571 ], View Topview, Score 524.5816325800977
"""

##Step 2
# 输入字符串
input_string = """
-189.092056, 6.345250, 122.203778, 30.909109, -47.325857, 84.691254, 270.0 
-175.774116, -0.291980, 148.101981, 28.810701, -100.084947, 94.974551, 140.87408652700051 
-175.357165, -0.311982, 148.448890, 29.060261, -100.817638, 94.942768, 9.699972800960209 
-172.764866, -10.661054, 150.262413, 31.904296, -116.042476, 100.040243, 69.42102704105517 
-172.763817, -10.658227, 150.263246, 31.905144, -116.039020, 100.038972, 290.5789729589448 
-185.075348, -14.248312, 83.442821, 204.558174, -104.925372, 162.129207, 290.87881416088754 
-184.289567, -13.271072, 84.488061, 204.491228, -104.246593, 162.744006, 349.699972802108796 
-184.830453, -13.939274, 83.774619, 204.536335, -104.710359, 162.319497,199.1259134725384 
-181.447951, -10.847293, 91.123234, 201.910552, -96.556440, 163.090619, 229.42102704164296 
-179.990905, -9.380650, 92.649838, 201.846032, -95.552703, 164.344833, 160.0 
-180.673751, -10.054780, 91.951747, 201.875142, -96.013850, 163.754272, 120.87408652729505 
-177.385927, -8.231679, 98.285067, 199.307314, -87.897881, 165.036609, 70.00000000000001 
-172.465107, -6.292139, 105.174321, 196.566834, -78.793543, 168.426058, 280.77677803222116 
-162.107319, -6.724262, 116.133450, 190.049542, -60.718700, 178.294037, 330.30002719809715 
-163.965121, -7.499930, 115.400758, 190.043673, -61.009210, 176.457405, 270.57897295866064 
-161.895320, -6.724564, 116.266403, 189.962783, -60.509303, 178.525671, 160.0 
-162.863108, -7.075867, 115.875556, 190.001623, -60.738247, 177.557418, 19.1259134722605 
"""


if __name__ == "__main__":
    #Step 1
    viewpoints = parse(log_text)
    # # 调整视点
    adjusted_viewpoints = adjust_viewpoints(viewpoints)

    # 对 adjusted_viewpoints 按照 'index' 排序
    adjusted_viewpoints = sorted(adjusted_viewpoints, key=lambda x: x['index'])

    # 打印结果
    for vp in adjusted_viewpoints:
        print(
            f"Viewpoint {vp['index']}: Position {vp['new_position']}, Direction {vp['new_direction']}, Rotation Angle (radians): {vp['rotation_angle']} ")

    # ##Step 2
    # save_csv(input_string, filename='output_orscan0.csv')
    #
