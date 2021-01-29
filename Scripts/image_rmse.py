import cv2
import sys
import numpy as np


def rmse(img_0,img_1):
    return np.sqrt(np.mean((img_0-img_1)**2))

if __name__ == "__main__":
    img_file_0 = sys.argv[1]
    img_file_1 = sys.argv[2]

    img_0 = cv2.imread(img_file_0)
    img_1 = cv2.imread(img_file_1)

    error = rmse(img_0,img_1)
    
    print(f"Root Mean Squared Error: {error}")