import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.animation as animate
from numpy import pi

FPS = 60
Bitrate = 10**4
Speed = 1/2

im = cv.imread('website.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = 0
for i in range (len(contours)) :
    if len(contours[i]) > len(contours[cnt]) :
        cnt = i
# Gets contour into x and y components
xlist = np.array(contours[cnt][:, :, 0]).flatten()
ylist = -np.array(contours[cnt][:, :, 1]).flatten()
# Centers image at (0, 0)
xlist = xlist - (np.max(xlist) + np.min(xlist))/2
ylist = ylist - (np.max(ylist) + np.min(ylist))/2

N = len(xlist)

fig, ax = plt.subplots()
ax.set_xlim(min(xlist) - 200, max(xlist) + 200)
ax.set_ylim(min(ylist) - 200, max(ylist) + 200)
ax.set_aspect(1)
ax.plot(xlist, ylist, color=(0, 0, 0, 0.2))
plt.show()