#coding = utf-8
import cv2
import numpy as np

class VisionDetection:
    """
    图像处理类
    function：
        1. 透视变换
        2. 寻找四边形顶点位置
        3. 寻找红点位置
        4. 寻找绿点位置
    Attributes:
        rec_loc: 矩形顶点位置及面积列表（按面积从小到大排序）
        redpoint_loc: 红点位置列表
        greenpoint_loc: 红点位置列表
    Agruments:
        mode: 模式，test表示测试模式（展示效果图），run表示运行模式（不展示效果图）
    """
    def __init__(self, mode='test'):
        self.area_threshold = 10000
        self.mode = mode
        self.rec_loc = []
        self.redpoint_loc  = []
        self.greenpoint_loc  = []
        self.center_loc = []
        self.result = None
        self.M = None
        self.WH = None

    def float2int(self, point):
        for i in range(len(point)):
            point[i] = int(point[i] *1000)
        return point
    def order_points(self, pts):
        """
        排序坐标点
        :param pts: 待排序的坐标点
        :return: 排序后的坐标点
        """
        # 初始化坐标点
        rect = np.zeros((4, 2), dtype="float32")

        # 顶部左角的点具有最小的和，
        # 底部右角的点具有最大的和
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 计算点的差值，
        # 顶部右角的点将具有最小的差值，
        # 底部左角的点将具有最大的差值
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # 返回排序后的坐标点
        return rect

    def compute_M(self, image):
        """
        计算透视变换矩阵
        :param image: 待处理的图片
        :return: 透视变换矩阵
        """
        warped = None
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化处理，将图像转换为黑白
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # 边缘检测，降低阈值以捕获更多细节
        edges = cv2.Canny(blurred, 50, 150)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历所有轮廓
        for contour in contours:
            # 近似轮廓，减少顶点数量
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果轮廓近似后有4个顶点，且面积大于某个阈值，则认为是矩形
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                # 获取四边形的四个顶点坐标
                box = approx.reshape(4, 2).astype("float32")

                # 重新排序顶点
                box = self.order_points(box)

                # 计算原始四边形的宽度和高度
                widthA = np.sqrt(((box[0][0] - box[1][0]) ** 2) + ((box[0][1] - box[1][1]) ** 2))
                widthB = np.sqrt(((box[2][0] - box[3][0]) ** 2) + ((box[2][1] - box[3][1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((box[0][0] - box[3][0]) ** 2) + ((box[0][1] - box[3][1]) ** 2))
                heightB = np.sqrt(((box[1][0] - box[2][0]) ** 2) + ((box[1][1] - box[2][1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                self.WH = [maxWidth, maxHeight]

                # 定义目标点，这里我们增加一些额外的空间，但要确保比例正确
                # 假设你想在每个边上增加10%的空间
                extraSpace = -0.05  # 5% extra space
                dst_width = maxWidth * (1 + 2 * extraSpace)
                dst_height = maxHeight * (1 + 2 * extraSpace)

                # 重新计算目标点
                dst_pts = np.array([
                    [0 - maxWidth * extraSpace, 0 - maxHeight * extraSpace],
                    [maxWidth + maxWidth * extraSpace, 0 - maxHeight * extraSpace],
                    [maxWidth + maxWidth * extraSpace, maxHeight + maxHeight * extraSpace],
                    [0 - maxWidth * extraSpace, maxHeight + maxHeight * extraSpace]
                ], dtype="float32")

                # 计算透视变换矩阵
                self.M = cv2.getPerspectiveTransform(box, dst_pts)

    def warp_image(self, image):
        """
        寻找轮廓
        :param image: 待处理的图片
        :return: 透视变换后的图片
        """
        # 执行透视变换
        warped = cv2.warpPerspective(image, self.M, (self.WH[0], self.WH[1]))
        
        return warped

    def find_rec(self, image):
        """
        寻找四边形顶点位置
        :param image: 待处理的图片
        :return: 所有四边形顶点位置列表
        """
        locations = []
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化处理，将图像转换为黑白
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # 边缘检测，降低阈值以捕获更多细节
        edges = cv2.Canny(blurred, 50, 150)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历所有轮廓
        for i, contour in enumerate(contours):
            # 近似轮廓，减少顶点数量
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果轮廓近似后有4个顶点，且面积大于某个阈值，则认为是矩形
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                # 获取四边形的四个顶点坐标
                box = approx.reshape(4, 2).astype("float32")

                # 重新排序顶点
                box = self.order_points(box)

                #将检测到的四边形的四个顶点坐标存储到lacations列表内
                location = []
                area = cv2.contourArea(contour)
                for j in range(4):
                    location.append((box[j][1] / image.shape[1], box[j][0] / image.shape[0]))
                locations.append([location,area])
                #绘制轮廓
                if self.mode == 'test':
                    cv2.drawContours(image, [approx], 0, (255, 0, 0), 1)
                # cv2.imshow('Image with Rec', image)
                # cv2.waitKey(0)

        locations.sort(key=lambda x:x[1])
        # 初始化一个空列表来保存最终保留的矩形
        final_locations = []

        # 遍历排序后的矩形列表
        for i, (coords, area) in enumerate(locations):
            # 检查当前矩形是否与之前保留的矩形面积相近
            if not any(abs(area - other_area) < self.area_threshold for _, other_area in final_locations):
                # 如果不相近，则保留当前矩形
                final_locations.append((coords, area))

        flattened_coordinates = [num for loc in final_locations for coord in loc[0] for num in coord]
        self.rec_loc = flattened_coordinates
 
    def find_redpoint(self, image):
        """
        寻找红色点
        :param image: 待处理的图片
        :return: 红点平均百分比坐标值
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # 合并两个红色范围的掩模
        mask = mask1 + mask2

        # 查找红色像素的位置
        red_points = np.column_stack(np.where(mask > 0))

        if red_points.size == 0:
            self.redpoint_loc = (0.0, 0.0)
        else:
            #此处已经进行x，y坐标交换
            x_mean = np.mean(red_points[:, 1])
            y_mean = np.mean(red_points[:, 0])
            center = (int(x_mean), int(y_mean))
            if self.mode == 'test':
                cv2.circle(image, center, radius=5, color=(255, 0, 0), thickness=-1)

            # 返回所有红点的平均位置
            x_percent = x_mean / image.shape[1]
            y_percent = y_mean / image.shape[0]
            self.redpoint_loc = [x_percent, y_percent]
            self.result = image

    def find_greenpoint(self, image):
        """
        寻找绿色点
        :param image: 待处理的图片
        :return: 绿点平均百分比坐标值
        """

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义绿色的HSV范围
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 查找红色像素的位置
        green_points = np.column_stack(np.where(mask > 0))

        if green_points.size == 0:
            self.greenpoint_loc = (0.0, 0.0)
        else:
            #此处已经进行x，y坐标交换
            x_mean = np.mean(green_points[:, 1])
            y_mean = np.mean(green_points[:, 0])
            center = (int(x_mean), int(y_mean))
            if self.mode == 'test':
                cv2.circle(image, center, radius=5, color=(255, 0, 0), thickness=-1)

            # 返回所有红点的平均位置
            x_percent = x_mean / image.shape[1]
            y_percent = y_mean / image.shape[0]
            self.greenpoint_loc = [x_percent, y_percent]
    
    def find_center(self):
        if len(self.rec_loc) >= 7:
           self.center_loc = [(self.rec_loc[0]+self.rec_loc[5])/2, (self.rec_loc[1]+self.rec_loc[6])/2]
        else:
           print("rec_loc does not have enough elements.")
           self.center_loc = None