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
    try:
        vs = VS(mode = 'run')#mode：test会产生效果图；run不会产生效果图
        ax = AX()
        while vs.WH is None:
            image = capture_image()
            vs.compute_M(image)
        while True:
            if ser.in_waiting > 0:
                # 读取一行数据
                line = ser.readline().decode('utf-8').strip()
                # 分割数据，假设命令在第一个位置
                parts = line.split()
                command = parts[0]
                data =parts[1:]
                #传入数据为后续部分

                # 读取图片
                image = capture_image()
                cv2.waitKey(10)
                warped = vs.warp_image(image)
                
                # 根据命令调用相应的函数
                if command == "R":
                    vs.find_redpoint(warped)
                    cv2.waitKey(10)
                    send_list_over_serial(command, vs.float2int(vs.redpoint_loc))
                elif command == "G":
                    vs.find_greenpoint(warped)
                    cv2.waitKey(10)
                    send_list_over_serial(command, vs.float2int(vs.greenpoint_loc))
                elif command == "T":
                    vs.find_rec(warped)
                    cv2.waitKey(10)
                    send_list_over_serial(command, vs.float2int(vs.rec_loc))
                elif command == "S":
                    ax.calculate_transformation_matrix(command, data)
                elif command == "C":
                    vs.find_center(warped)
                    cv2.waitKey(10)
                    send_list_over_serial(command, vs.float2int(vs.center_loc))
                else:
                    print(f"Invalid command{command}")
    except KeyboardInterrupt:
        ser.close()

        # 释放摄像头资源
        cap.release()

if __name__ == "__main__":
    # 初始化串口
    ser = serial.Serial('./dev/ttyTHS0', 9600) 
    # 打开默认摄像头，通常索引为0
    cap = cv2.VideoCapture(0)
    run()