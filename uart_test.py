import serial

# 初始化串口
ser = serial.Serial('COM3', 115200)  # 更改'COM3'为你的实际串口号

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
            
            print("数据发送成功")
    
    except Exception as e:
        print(f"发生错误: {e}")

def run():
    try:
        while True:
            if ser.in_waiting > 0:
                # 读取一行数据
                line = ser.readline().decode('utf-8').strip()
                # 分割数据，假设命令在第一个位置
                parts = line.split()
                command = parts[0]
                data = parts[1:]
                #传入数据为后续部分
                print(f"command: {command}, data: {data}")

                send_list_over_serial(command)

    except KeyboardInterrupt:
        ser.close()

    
if __name__ == "__main__":
    run()