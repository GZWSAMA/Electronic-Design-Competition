#coding = utf-8
import numpy as np
import math
from vision_detection.vision_detection import VisionDetection as vs

class AxesTransfer:
    """
    AxesTransfer class
    """
    def __init__(self):
        self.transformation_matrix = []
    def calculate_transformation_matrix(self,datas):
        VS_cal = vs()
        PWM1_1 = math.tan(datas[0]/10)
        PWM1_2 = math.tan(datas[1]/10)/math.cos(datas[0]/10)
        PWM2_1 = datas[2]/10
        PWM2_2 = datas[3]/10
        point1 = (datas[4],datas[5])
        point2 = (datas[6],datas[7])
        src_points = [(PWM1_1, PWM1_2),(PWM2_1, PWM2_2)]
        dst_points = [point1,point2]
        T, residuals, rank, s = np.linalg.lstsq(src_points, dst_points, rcond=None)
        self.transformation_matrix = T


# Example usage:
# src_points = np.array([[100, 200],      [300, 400]])
# dst_points = np.array([[0.5, 0.6],      [0.7, 0.8]])

# Haeundae2A4_matrix = calculate_transformation_matrix(src_points, dst_points)
# print(f"Haeundae2A4_Matrix:\n{Haeundae2A4_matrix}\nA42Haeundae_Matrix:\n{np.linalg.inv(Haeundae2A4_matrix)}")
