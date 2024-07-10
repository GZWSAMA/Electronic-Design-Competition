import cv2
import numpy as np

# 读取图像
image = cv2.imread('./datas/1.png')
cv2.namedWindow('Color Detection')

# 创建滑动条
def nothing(x):
    pass

cv2.createTrackbar('H Min', 'Color Detection', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Color Detection', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Color Detection', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Color Detection', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Color Detection', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Color Detection', 255, 255, nothing)

while True:
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 获取滑动条的当前值
    h_min = cv2.getTrackbarPos('H Min', 'Color Detection')
    h_max = cv2.getTrackbarPos('H Max', 'Color Detection')
    s_min = cv2.getTrackbarPos('S Min', 'Color Detection')
    s_max = cv2.getTrackbarPos('S Max', 'Color Detection')
    v_min = cv2.getTrackbarPos('V Min', 'Color Detection')
    v_max = cv2.getTrackbarPos('V Max', 'Color Detection')

    # 根据HSV范围创建掩模
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 应用掩模
    result = cv2.bitwise_and(image, image, mask=mask)

    # 显示原图、掩模和结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理窗口
cv2.destroyAllWindows()