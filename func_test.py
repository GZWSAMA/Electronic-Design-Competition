import cv2
import numpy as np

def find_contours(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选四边形轮廓
    quadrilaterals = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            quadrilaterals.append(approx)
    return quadrilaterals

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 找到四边形并绘制轮廓
        quadrilaterals = find_contours(frame)
        for quad in quadrilaterals:
            cv2.drawContours(frame, [quad], -1, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Quadrilateral Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()