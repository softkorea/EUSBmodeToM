"""
opencv 3.4.1
python 3.5.3
numpy 1.4
if you want support, write an issue at https://github.com/softkorea/EUSBmodeToM/issues
All rights reserved by Juhee Kim
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from LineIterator import *  # if you want lineiterator class get it from  https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator.

base_x = 445
base_y = 243
radius = 150

#direction = [(-1,0),(1,0),(0,-1),(0,1),(-0.7071,-0.7071),(0.7071,-0.7071),(-0.7071,0.7071),(0.7071,0.7071)]
#direction = [(-1,0)]

ffangle = np.sqrt(2)/2
direction = [(1,0),(ffangle,ffangle),(0,1),(-ffangle,ffangle),(-1,0),
             (-ffangle,-ffangle),(0,-1),(ffangle,-ffangle) ]


def BGRtoRGB(frame):
    b, g, r = cv2.split(frame)
    return cv2.merge([r, g, b])

def getLineImage(frame,base_x,base_y,direction,raduius):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tbuffer = []
    for dx,dy in direction:
        t = createLineIterator(
            np.asarray([base_x,base_y]),
            np.asarray([int(base_x+dx*radius),int(base_y+dy*radius)]),
            radius,
            grayframe
        )
        tbuffer.append(t)
    return np.asarray(tbuffer)

def showSetting(filename):
    eusvid = cv2.VideoCapture(filename)
    ret, frame = eusvid.read()
    fig, ax = plt.subplots()

    ax.imshow(BGRtoRGB(frame))
    ax.scatter([base_x],[base_y],s=[2])
    for dx,dy in direction:
        ax.plot(
            [base_x,base_x+dx*radius],
            [base_y,base_y+dy*radius],
            color='red'
        )
    plt.show()
    eusvid.release()


def GetPseudoMmode(filename):
    eusvid = cv2.VideoCapture(filename)
    output = []

    while(eusvid.isOpened()):
        ret, frame = eusvid.read()
        if ret:
            l = getLineImage(frame, base_x, base_y, direction,radius)
            output.append(l)
        else:
            break

    eusvid.release()

    return np.asarray(output).transpose([1,0,2])

def showEUSoutput(rawoutput):
    ld = len(rawoutput)
    fig = plt.figure()
    count = 0
    for mmode in rawoutput:
        x = np.flip(np.asarray(mmode.tolist()).transpose(), axis=0)

        count += 1
        ax = fig.add_subplot(ld, 1, count)
        ax.imshow(x, cmap="gray")
    plt.show()

if __name__ == '__main__':
    avi = 'test.avi'
    showSetting(avi)
    rawoutput = GetPseudoMmode(avi)
    showEUSoutput(rawoutput)

