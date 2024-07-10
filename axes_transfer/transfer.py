#coding = utf-8
import numpy as np

def calculate_transformation_matrix(src_points, dst_points):
    T, residuals, rank, s = np.linalg.lstsq(src_points, dst_points, rcond=None)
    return T

# Example usage:
src_points = np.array([[1, 2], [3, 4]])
dst_points = np.array([[0.5, 0.6], [0.7, 0.8]])

Haeundae2A4_matrix = calculate_transformation_matrix(src_points, dst_points)
print(f"Haeundae2A4_Matrix:\n{Haeundae2A4_matrix}\nA42Haeundae_Matrix:\n{np.linalg.inv(Haeundae2A4_matrix)}")
