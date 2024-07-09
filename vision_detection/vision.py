#coding = utf-8
import cv2
import numpy as np

def order_points(pts):
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

def find_contours(image):
    """
    寻找轮廓
    :param image: 待处理的图片
    :return: 透视变换后的图片
    """

    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理，将图像转换为黑白
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            box = order_points(box)

            # 计算原始四边形的宽度和高度
            widthA = np.sqrt(((box[0][0] - box[1][0]) ** 2) + ((box[0][1] - box[1][1]) ** 2))
            widthB = np.sqrt(((box[2][0] - box[3][0]) ** 2) + ((box[2][1] - box[3][1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((box[0][0] - box[3][0]) ** 2) + ((box[0][1] - box[3][1]) ** 2))
            heightB = np.sqrt(((box[1][0] - box[2][0]) ** 2) + ((box[1][1] - box[2][1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            # 根据原始四边形的宽度和高度定义目标点
            dst_pts = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(box, dst_pts)

            # 执行透视变换
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

            #绘制轮廓
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Detected Rectangles', image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return warped

def find_redpoint(image):
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
        return 0.0, 0.0
    #此处已经进行x，y坐标交换
    x_mean = np.mean(red_points[:, 1])
    y_mean = np.mean(red_points[:, 0])
    center = (int(x_mean), int(y_mean))
    cv2.circle(image, center, radius=5, color=(0, 0, 255), thickness=-1)

    # 显示结果
    cv2.imshow('Original Image with Red Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 返回所有红点的平均位置
    x_percent = x_mean / image.shape[1]
    y_percent = y_mean / image.shape[0]
    return x_percent, y_percent

def find_greenpoint(image):
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
        return 0.0, 0.0
    #此处已经进行x，y坐标交换
    x_mean = np.mean(green_points[:, 1])
    y_mean = np.mean(green_points[:, 0])
    center = (int(x_mean), int(y_mean))
    cv2.circle(image, center, radius=5, color=(0, 0, 255), thickness=-1)

    # 显示结果
    cv2.imshow('Original Image with Green Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 返回所有红点的平均位置
    x_percent = x_mean / image.shape[1]
    y_percent = y_mean / image.shape[0]
    return x_percent, y_percent

def run():
    # 读取图片
    image = cv2.imread('./datas/1.png')

    # 寻找轮廓
    warped = find_contours(image)

    # 寻找红绿色点
    x_redpoint, y_redpoint = find_redpoint(warped)
    x_greenpoint, y_greenpoint = find_greenpoint(warped)
    print(f"redpoint: {x_redpoint}, {y_redpoint} \ngreenpoint: {x_greenpoint}, {y_greenpoint}")

if __name__ == '__main__':
    run()