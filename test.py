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

vs = VS(mode = 'test')#mode：test会产生效果图；run不会产生效果图中断程序

def on_compute():
    image = capture_image()
    vs.compute_M(image)

def run():
    # keyboard.add_hotkey('s', on_compute)
    # while vs.WH is None:#按下s进行M计算
    #     image = capture_image()
    #     if image is None or image.size == 0:
    #         print("Image is empty!")
    #         continue
    #     cv2.imshow("image", image)
    #     cv2.waitKey(10)


    while True:
        # 读取图片
        image = cv2.imread("./datas/2.png")
        vs.compute_M(image)

        # image = capture_image()
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
        print(f"\nredpoint: \n{vs.float2int(vs.redpoint_loc)} \n\ngreenpoint: \n{vs.float2int(vs.greenpoint_loc)} \n\ncenter: {vs.float2int(vs.center_loc)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 打开默认摄像头，通常索引为0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    run()
