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
# 串口信息——指令+数据

# 初始化串口
ser = serial.Serial('COM3', 115200)  # 更改'COM3'为你的实际串口号

import serial

def send_list_over_serial(data_list):
    try:
        if ser.isOpen():
            # 将列表转换为字符串，使用逗号作为分隔符
            # 使用str()函数确保所有元素都被转换为字符串
            data_str = ','.join(map(str, data_list))
            
            # 将字符串编码为字节串
            data_bytes = data_str.encode('utf-8')
            
            # 发送数据
            ser.write(data_bytes)
            
            # 关闭串口
            ser.close()
            
            print("数据发送成功")
    
    except Exception as e:
        print(f"发生错误: {e}")

try:
    vs = VS(mode = 'run')#mode：test会产生效果图；run不会产生效果图
    ax = AX()
    while True:
        if ser.in_waiting > 0:
            sent_datas = []
            # 读取一行数据
            line = ser.readline().decode('utf-8').strip()
            # 分割数据，假设命令在第一个位置
            parts = line.split()
            command = parts[0]
            data = ' '.join(parts[1:])
            #传入数据为后续部分

            # 读取图片
            image = cv2.imread('./datas/1.png')
            warped = vs.warp_image(image)
            
            # 根据命令调用相应的函数
            if command == "R":
                vs.find_redpoint(warped)
                send_list_over_serial(vs.redpoint_loc)
            elif command == "G":
                vs.find_greenpoint(warped)
                send_list_over_serial(vs.greenpoint_loc)
            elif command == "T":
                vs.find_rec(warped)
                send_list_over_serial(vs.rec_loc)
            elif command == "S":
                ax.calculate_transformation_matrix(data)
            else:
                print(f"Invalid command{command}")
except KeyboardInterrupt:
    ser.close()
