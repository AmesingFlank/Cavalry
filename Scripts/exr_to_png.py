import cv2
import sys
import numpy as np

def gamma_correct(value):
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * pow(value, (1.0 / 2.4)) - 0.055


def ACESToneMapping(value):
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14

    value *= 2**1.8
    return (value * (A * value + B)) / (value * (C * value + D) + E)

def FilmicToneMapping(value):
    def F(x):
        A = 0.22
        B = 0.30
        C = 0.10
        D = 0.20
        E = 0.01
        F = 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    WHITE = 11.2
    return F(1.6 * value) / F(WHITE)

def luminance(pixel):
    return pixel[0]*0.212671 + pixel[1] * 0.715160 + pixel[2]* 0.072169

def tone_map(img):
    #img = np.vectorize(ACESToneMapping)(img)
    #img = np.vectorize(FilmicToneMapping)(img)
    tonemap = cv2.createTonemapDrago(1)
    img = tonemap.process(img)

    return img


def convert(exr_file,png_file):
    img = cv2.imread(exr_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = tone_map(img)
    img = np.vectorize(gamma_correct)(img)
    img *= 255
    print(np.max(img),np.min(img),np.median(img),np.mean(img))
    print(img.shape)

    #img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
    cv2.imwrite(png_file,img)


if __name__ == "__main__":
    convert(sys.argv[1],sys.argv[2])