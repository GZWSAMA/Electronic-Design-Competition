#coding = utf-8
import cv2
import serial
from vision_detection.vision_detection import VisionDetection as VS
from axes_transfer.axes_transfer import calculate_transformation_matrix

def run():
    vs = VS(mode = 'test')#mode：test会产生效果图；run不会产生效果图
    # 读取图片
    image = cv2.imread('./datas/1.png')

    # 寻找最大轮廓并透视变换
    warped = vs.warp_image(image)
    
    #更新矩形框四个顶点位置
    lacations = vs.find_rec(warped)
    
    # 更新红绿色点位置
    vs.find_redpoint(warped)
    vs.find_greenpoint(warped)
    print(f"rectangle: {vs.rec_loc} \nredpoint: {vs.redpoint_loc} \ngreenpoint: {vs.greenpoint_loc}")


# 创建一个字典，用于映射命令到函数
# R——红点坐标
# G——绿点坐标
# T——a4纸四顶点坐标
# S——计算变换矩阵
# 串口信息——指令+数据
command_functions = {
    "R": VS.find_redpoint,
    "G": VS.find_greenpoint,
    "T": VS.find_rec,
    "S": calculate_transformation_matrix
}

# 初始化串口
ser = serial.Serial('COM3', 115200)  # 更改'COM3'为你的实际串口号

try:
    vs = VS(mode = 'test')#mode：test会产生效果图；run不会产生效果图
    while True:
        if ser.in_waiting > 0:
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
            if command in command_functions:
                if command == "S":
                    calculate_transformation_matrix(data)
                else:
                    command_functions[command](warped)
            else:
                print(f"Unknown command: {command}")
except KeyboardInterrupt:
    ser.close()


if __name__ == '__main__':
    run()