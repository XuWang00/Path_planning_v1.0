import numpy as np
import matplotlib.pyplot as plt


def plot_multiple_lines(data, labels):
    """
    Plots multiple lines of data on a single graph with custom labels.

    Parameters:
    - data: A list of lists, where each sublist contains percentage values for each series.
    - labels: A list of strings for the labels of each line.
    """
    # Define x-axis labels
    x_labels = range(1, 41)  # Adjust to the correct range if necessary

    # Line styles and markers for differentiation
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', 'd', '^', 'v']

    # Plotting the data
    plt.figure(figsize=(10, 6))
    for idx, series in enumerate(data):
        plt.plot(x_labels, series, line_styles[idx % len(line_styles)], marker=markers[idx % len(markers)],
                 label=labels[idx])

    # Adding chart details
    # plt.title('Comparison of Percentages Across Series')
    plt.xlabel('Viewpoint Number')
    plt.ylabel('Scan Coverage (%)')
    plt.grid(True)
    plt.legend()
    plt.xticks(x_labels)  # Set x-axis ticks
    plt.yticks(np.arange(0, 101, 5))  # 设置y轴刻度
    plt.ylim(0, 100)  # 扩展y轴的显示范围以分散线条
    plt.show()



def plot_viewpoint_scores(viewpoint_details):
    """
    Plots the scores of viewpoints against their indices.

    Parameters:
    - viewpoint_details: List of dictionaries containing viewpoint index, position, direction, view, and score.
    """
    # 提取索引和得分
    indices = [vp['index'] + 1 for vp in viewpoint_details]  # 索引从1开始
    scores = [vp['score'] for vp in viewpoint_details]

    # 创建折线图
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(indices, scores, marker='o', linestyle='-', color='b')  # 折线图，蓝色线条和圆圈标记
    plt.title('Viewpoint Scores')  # 图表标题
    plt.xlabel('Viewpoint Index')  # x轴标签
    plt.ylabel('Score')  # y轴标签
    plt.grid(True)  # 显示网格
    plt.xticks(indices)  # 设置x轴刻度标签
    plt.show()  # 显示图表



if __name__ == "__main__":
    # # Example data provided as lists of percentages
    # data = [
    #     [31.02, 60.29, 83.75, 89.61, 92.67, 95.80, 97.80, 99.00],
    #     [30.59, 59.10, 82.25, 87.84, 92.40, 94.92, 96.92, 97.90],
    #     [28.27, 57.36, 80.65, 87.98, 91.90, 95.34, 96.76, 97.67],
    #     [28.27, 57.36, 80.65, 87.98, 90.87, 94.49, 95.46, 96.84],
    #     [28.27, 57.36, 77.54, 85.21, 90.69, 92.28, 93.24, 94.13]
    # ]
    #
    # # Call the function with the example data
    # labels = ["a=1, b=0", "a=0.75, b=0.25", "a=0.5, b=0.5", "a=0.25, b=0.75", "a=0, b=1"]
    # Example data provided as lists of percentages
    data = [
        [25.16, 38.53, 47.22, 53.11, 57.20, 60.21, 62.08, 63.70, 65.14, 66.39, 67.37, 68.34, 68.76, 69.22, 69.64, 69.96,
         70.25, 70.53, 70.78, 71.00, 71.18, 71.32, 71.47, 71.61, 71.73, 71.85, 71.94, 72.03, 72.12, 72.21, 72.29, 72.35,
         72.43, 72.49, 72.55, 72.60, 72.65, 72.70, 72.75, 72.80]
    ]

    # Call the function with the example data
    labels = ["a=3, b=1"]
    # Call the function with the example data and custom labels
    plot_multiple_lines(data, labels)
