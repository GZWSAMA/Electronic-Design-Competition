#coding = utf-8
import cv2
import keyboard
from vision_detection.vision_detection import VisionDetection as VS 

def capture_image():
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 读取一帧图像
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("无法获取帧")
        exit()

    return frame

# 初始化VS类的一个实例，传入特定的参数
vs = VS(
    mode='test',          # 第一个参数，模式参数。当设置为'test'时，系统将生成并展示中间处理结果或最终的效果图，
                          # 这对于调试和可视化处理流程非常有用。如果设置为'run'，则可能不会生成额外的输出，
                          # 以避免中断正常的程序流程或节省资源。

    blurred_para=5,      # 第二个参数，模糊参数。这可能用于控制图像模糊的程度，例如在进行边缘检测前，
                          # 使用高斯模糊或其他类型的模糊来减少图像噪声。数值越大，模糊效果越强。
                          #可以去除噪声

    edge_para=120        # 第三个参数，边缘参数。这可能用于控制边缘检测的敏感度或阈值。在Canny边缘检测等算法中，
                          # 较高的值意味着只有非常明显的边缘才会被检测到，较低的值则会检测到更多的边缘细节。
                          #越大中间的矩形越容易被检测到，但过大会导致边缘细节消失
)

def run():
    while vs.WH is None:
        image = capture_image()
        cv2.imshow("image", image)
        cv2.waitKey(10)
        vs.compute_M(image)
        print("WH is None")

    while True:
        #读取图片
        # image = cv2.imread("./datas/9.jpg")
        # while vs.WH is None:
        #     vs.compute_M(image)
        #     print("WH is None")

        image = capture_image()
        if image is None or image.size == 0:
            print("Image is empty!")
            continue
        cv2.imshow("image", image)
        cv2.waitKey(10)

        # 寻找最大轮廓并透视变换
        warped = vs.warp_image(image)
        if warped is None or warped.size == 0:
            print("warped is empty!")
            continue
        cv2.imshow("warped", warped)
        cv2.waitKey(10)

        #更新矩形框四个顶点位置
        lacations = vs.find_rec(warped)
        # 更新红绿色点位置
        vs.find_redpoint(warped)
        vs.find_greenpoint(warped)
        vs.find_center()
        if vs.result is not None and vs.result.size > 0:
            cv2.imshow("result", vs.result)
        else:
            print("vs.result is an empty image.")

        print("rectangle: ")
        for i in range(0, len(vs.rec_loc), 8):
            print(vs.rec_loc[i:i+8])
        print(f"\nredpoint: \n{vs.float2int(vs.select_point(vs.redpoint_loc))} \n\ngreenpoint: \n{vs.float2int(vs.select_point(vs.greenpoint_loc))} \n\ncenter: {vs.float2int(vs.center_loc)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 打开默认摄像头，通常索引为0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    run()
