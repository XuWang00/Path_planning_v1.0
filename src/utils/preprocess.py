import vtk


class PickerCallback:
    def __init__(self):
        self.poly_data = None  # 被操作的多边形数据
        self.renderer = None  # 渲染器实例
        # 设置用于显示框选面片的Mapper和Actor
        self.mapper = vtk.vtkDataSetMapper()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

    # 当框选事件结束时执行的方法
    def execute(self, caller, event):
        area_picker = caller
        frustum = area_picker.GetFrustum()

        extract_geometry = vtk.vtkExtractPolyDataGeometry()
        extract_geometry.SetInputData(self.poly_data)
        extract_geometry.SetImplicitFunction(frustum)
        extract_geometry.Update()

        if not extract_geometry.GetOutput().GetNumberOfCells():
            return

        # 定位被选中的cell
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(extract_geometry.GetOutput())
        locator.BuildLocator()

        # 获取摄像头位置和视图方向
        camera = self.renderer.GetActiveCamera()
        camera_pos = camera.GetPosition()
        view_focal_point = camera.GetFocalPoint()
        view_direction = [view_focal_point[i] - camera_pos[i] for i in range(3)]

        # 初始化光线投射的结果变量
        t = vtk.mutable(0.0)  # 光线投射的参数t
        xyz = [0.0, 0.0, 0.0]  # 交点坐标
        pcoords = [0.0, 0.0, 0.0]  # 交点的参数坐标
        subId = vtk.mutable(0)  # 相交单元格的子Id
        cellId = vtk.mutable(-1)  # 相交的单元格Id

        # 从摄像头位置向视图中心发射光线，找到第一个与模型相交的面片
        hit = locator.IntersectWithLine(camera_pos, view_direction, 0.0001, t, xyz, pcoords, subId, cellId)

        if hit:
            # 使用cellId来过滤和显示靠近摄像头的面片
            connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
            connectivity_filter.SetInputConnection(extract_geometry.GetOutputPort())
            connectivity_filter.SetExtractionModeToSpecifiedRegions()
            connectivity_filter.AddSpecifiedRegion(cellId)
            connectivity_filter.Update()

            # 设置Mapper和Actor以显示选中的面片
            self.mapper.SetInputConnection(connectivity_filter.GetOutputPort())
            self.mapper.ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(1, 0, 0)  # Red color
            self.actor.GetProperty().SetRepresentationToWireframe()

            self.renderer.AddActor(self.actor)
            self.renderer.GetRenderWindow().Render()

    # 设置需要处理的多边形数据
    def set_poly_data(self, poly_data):
        self.poly_data = poly_data

    # 设置渲染器
    def set_renderer(self, renderer):
        self.renderer = renderer

# 主函数，设置VTK环境和框选逻辑
def main():
    sphere_source = vtk.vtkSphereSource() # 创建一个球形数据源
    sphere_source.Update()

    mapper = vtk.vtkPolyDataMapper()  # 创建Mapper
    mapper.SetInputConnection(sphere_source.GetOutputPort())

    actor = vtk.vtkActor()  # 创建Actor
    actor.SetMapper(mapper)

    # 显示网格
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetColor(0, 1, 0)  # 设置网格颜色为绿色
    actor.GetProperty().SetLineWidth(1)  # 设置线宽

    renderer = vtk.vtkRenderer()  # 创建Renderer
    renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()  # 创建RenderWindow
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()   # 创建交互器
    interactor.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleRubberBandPick()  # 设置交互方式为框选
    interactor.SetInteractorStyle(style)

    area_picker = vtk.vtkAreaPicker()  # 创建区域拾取器
    interactor.SetPicker(area_picker)

    callback = PickerCallback()   # 创建框选回调实例
    callback.set_poly_data(sphere_source.GetOutput())   # 设置多边形数据
    callback.set_renderer(renderer)   # 设置渲染器
    area_picker.AddObserver("EndPickEvent", callback.execute)  # 绑定框选结束事件到回调函数
    # 开始交互
    interactor.Start()


if __name__ == "__main__":
    main()
