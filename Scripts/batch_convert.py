
import exr_to_png

from os import listdir
from os.path import isfile, join
import sys

def batch_convert(dir):
    files = [f for f in listdir(dir)]
    files = [f for f in files if f.split('.')[-1]=='exr']
    files = [join(dir,f) for f in files]
    for exr_file in files:
        parts = exr_file.split('.')
        parts[-1] = 'png'
        png_file = '.'.join(parts)
        exr_to_png.convert(exr_file,png_file)


if __name__ == "__main__":
    dir = sys.argv[1]
    batch_convert(dir)
