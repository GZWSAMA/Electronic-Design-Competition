#coding = utf-8
import cv2
from vision_detection.vision_detection import VisionDetection as VS 


def run():
    vs = VS(mode = 'run')#mode：test会产生效果图；run不会产生效果图
    # 读取图片
    image = cv2.imread('./datas/1.png')

    # 寻找最大轮廓并透视变换
    warped = vs.warp_image(image)
    
    #更新矩形框四个顶点位置
    lacations = vs.find_rec(warped)
    
    # 更新红绿色点位置
    vs.find_redpoint(warped)
    vs.find_greenpoint(warped)
    vs.find_center()
    print("rectangle: ")
    for i in range(0, len(vs.rec_loc), 8):
        print(vs.rec_loc[i:i+8])
    print(f"\nredpoint: \n{vs.redpoint_loc} \n\ngreenpoint: \n{vs.greenpoint_loc} \n\ncenter: {vs.center_loc}")

if __name__ == '__main__':
    run()
