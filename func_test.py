import cv2
import numpy as np

def draw_contours(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return
    
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 10, 50, apertureSize=3)

    # 创建一个结构元素，通常是一个矩形或圆形
    kernel = np.ones((3, 3), np.uint8)

    # 使用dilate函数加粗边缘
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历每个轮廓
    for contour in contours:
        # 近似轮廓，使其更接近多边形
        epsilon = 0.003 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 检查轮廓是否为四边形
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            # 绘制轮廓
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    
    # 显示结果
    cv2.imshow('Detected Quadrilaterals', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用图像路径调用函数
draw_contours('./datas/3.jpg')