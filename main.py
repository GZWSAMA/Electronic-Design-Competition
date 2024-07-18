#coding = utf-8
import cv2
import serial
from vision_detection.vision_detection import VisionDetection as VS
from axes_transfer.axes_transfer import AxesTransfer as AX

# 创建一个字典，用于映射命令到函数
# R——红点坐标
# G——绿点坐标
# T——a4纸四顶点坐标
# S——计算变换矩阵
# C——计算中心点
# 串口信息——指令+数据

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

def send_list_over_serial(command, data_list):
    try:
        if ser.isOpen():
            # 将列表转换为字符串，使用逗号作为分隔符
            # 使用str()函数确保所有元素都被转换为字符串
            data_str = ','.join(map(str, data_list))
            
            # 将字符串编码为字节串
            data_bytes = (command + data_str + '\r\n').encode('utf-8')
            
            # 发送数据
            ser.write(data_bytes)
            
            print("数据发送成功")
    
    except Exception as e:
        print(f"发生错误: {e}")

def run():
    mode = 'test'
    try:
        # 初始化VS类的一个实例，传入特定的参数
        vs = VS(
            mode='test',          # 第一个参数，模式参数。当设置为'test'时，系统将生成并展示中间处理结果或最终的效果图，
                                # 这对于调试和可视化处理流程非常有用。如果设置为'run'，则可能不会生成额外的输出，
                                # 以避免中断正常的程序流程或节省资源。

            blurred_para=5,      # 第二个参数，模糊参数。这可能用于控制图像模糊的程度，例如在进行边缘检测前，
                                # 使用高斯模糊或其他类型的模糊来减少图像噪声。数值越大，模糊效果越强。

            edge_para=120        # 第三个参数，边缘参数。这可能用于控制边缘检测的敏感度或阈值。在Canny边缘检测等算法中，
                                # 较高的值意味着只有非常明显的边缘才会被检测到，较低的值则会检测到更多的边缘细节。
        )
        ax = AX()
        while vs.WH is None:
            image = capture_image()
            vs.compute_M(image)
            print("M未计算")

        image = capture_image()
        warped = vs.warp_image(image)
        #更新矩形框四个顶点位置
        lacations = vs.find_rec(warped)
        vs.find_center()
        rec_int = vs.float2int(vs.rec_loc)
        center_int = vs.float2int(vs.center_loc)

        while True:
            image = capture_image()
            if image is None or image.size == 0:
                print("Image is empty!")
                continue
            cv2.imshow("Image", image)
            cv2.waitKey(10)
            warped = vs.warp_image(image)
            if warped is None or warped.size == 0:
                print("warped is empty!")
                continue
            if mode == 'test':
                cv2.imshow("warped", warped)
                cv2.waitKey(10)
            # 更新红绿色点位置
            vs.find_redpoint(warped)
            vs.find_greenpoint(warped)
            if mode == 'test':
                if vs.result is not None and vs.result.size > 0:
                    cv2.imshow("result", vs.result)
                else:
                    print("vs.result is an empty image.")
            if ser.in_waiting > 0:
                # 读取一行数据
                line = ser.readline().decode('utf-8').strip()
                # 分割数据，假设命令在第一个位置
                parts = line.split()
                command = parts[0]
                data =parts[1:]
                #传入数据为后续部分

                # # 读取图片
                # image = capture_image()
                # cv2.waitKey(10)
                # warped = vs.warp_image(image)
                
                # 根据命令调用相应的函数
                if command == "R":
                    send_list_over_serial(command, vs.float2int(vs.select_point(vs.redpoint_loc)))
                elif command == "G":
                    send_list_over_serial(command, vs.float2int(vs.select_point(vs.greenpoint_loc)))
                elif command == "T":
                    send_list_over_serial(command, rec_int)
                elif command == "S":
                    ax.calculate_transformation_matrix(command, data)
                elif command == "C":
                    send_list_over_serial(command, center_int)
                else:
                    print(f"Invalid command{command}")
    except KeyboardInterrupt:
        ser.close()

        # 释放摄像头资源
        cap.release()

if __name__ == "__main__":
    # 初始化串口
    ser = serial.Serial('/dev/ttyTHS0', 9600) 
    # 打开默认摄像头，通常索引为0
    cap = cv2.VideoCapture(0)
    run()