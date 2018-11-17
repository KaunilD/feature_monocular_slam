import numpy as np
import cv2
import glob

W = 1920
H = 1080


def stitch(frames, size):
    extension = '.mp4'
    file_name = 'video' + '_' + str(size[0]) + '_' + str(size[1]) + extension

    video = cv2.VideoWriter(file_name,-1,1,size)
    for path in frames:
        img = cv2.imread(path)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    files = glob.glob('2011_09_26/2011_09_26_drive_0011_sync/image_00/data/*.png')
    stitch(files, (W//2, H//2))
