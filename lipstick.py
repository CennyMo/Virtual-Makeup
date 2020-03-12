import sys
import os
import dlib
import glob
import cv2
from skimage import io
from skimage import color
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from pylab import *

r, g, b = (159,19,20.)
up_left_end = 4
up_right_end = 7
in_left_end = 3
in_right_end = 7

lower_left_end = 5
upper_left_end = 11
lower_right_end = 16
upper_right_end = 22

inten = 0.8


def inter(lx, ly, k1='quadratic'):
    unew = np.arange(lx[0], lx[-1] + 1, 1)
    f2 = interp1d(lx, ly, kind=k1)
    return f2, unew

def getpoint(img):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    points = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    x = []
    y = []
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        i = 0
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            j = str(i)
            # cv2.putText(img, j, pt_pos,  cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0), 1)
            i = i + 1
            x.append(pt.x)
            y.append(pt.y)
        for i in range(0, len(x) - 1):
            if i >= 48 and i <= 59:
                pos = (x[i], y[i])
                points.append(pos)
        pos = (x[48], y[48])
        points.append(pos)
        for i in range(0, len(x) - 1):
            if i >= 60 and i <= 67:
                pos = (x[i], y[i])
                points.append(pos)
                if i == 64:
                    pos = (x[54], y[54])
                    points.append(pos)
        pos = (x[67], y[67])
        points.append(pos)
    return points


img = np.array(imread("su.jpg"))
points=getpoint(img)
points=np.array(points)
point_out_x = np.array((points[:12][:, 0]))
point_out_y = np.array(points[:12][:, 1])
point_in_x = np.array(points[12:][:, 0])
point_in_y = np.array(points[12:][:, 1])

im = img.copy()
im2 = img.copy()


o_l = inter([point_out_x[0]] + point_out_x[up_right_end - 1:][::-1].tolist(),
            [point_out_y[0]] + point_out_y[up_right_end - 1:][::-1].tolist(), 'quadratic')
o_u = inter( point_out_x[:up_right_end][::-1].tolist(),
             point_out_y[:up_right_end][::-1].tolist(), 'quadratic')

i_u = inter( point_in_x[:in_right_end][::-1].tolist(),
             point_in_y[:in_right_end][::-1].tolist(), 'quadratic')
i_l = inter([point_in_x[0]] + point_in_x[in_right_end - 1:][::-1].tolist(),
            [point_in_y[0]] + point_in_y[in_right_end - 1:][::-1].tolist(), 'quadratic')
x = []  # will contain the x coordinates of points on lips
y = []  # will contain the y coordinates of points on lips

for i in range(int(point_in_x[0]),int(point_in_x[6])):
 #   k = k + 1
    for j in range(int(o_u[0](i)),int(i_u[0](i))):
        x.append(j)
        y.append(i)
    for j in range(int(i_l[0](i)), int(o_l[0](i))):
        x.append(j)
        y.append(i)

for i in range(int(point_out_x[0]),int(point_in_x[0])):
    for j in range(int(o_u[0](i)),int(o_l[0](i))):
        x.append(j)
        y.append(i)

for i in range(int(point_in_x[6]),int(point_out_x[6])):
    for j in range(int(o_u[0](i)),int(o_l[0](i))):
        x.append(j)
        y.append(i)

#Change RGB color
val = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
ll, aa, bb = L1 - L, A1 - A, B1 - B
val[:, 0] +=ll*inten
val[:, 1] +=aa*inten
val[:, 2] += bb*inten

#Change HSV, making it more natural,optional
im[x, y] = color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255
hsv_val = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
hsv_val2=hsv_val[x,y]
L3,A3,B3=mean(hsv_val2[:, 0]), mean(hsv_val2[:, 1]),mean(hsv_val2[:, 2])
hsv_val2[:,0]-=0
hsv_val2[:,1]-=0
hsv_val2[:,2]+=0

#水润：
#hsl实验：

hsv_val[x,y]=hsv_val2
im=cv2.cvtColor(hsv_val,cv2.COLOR_HSV2BGR)
for i in range(0,len(x)-1):
       k=x[i]
       f=y[i]
       if(im[k,f][0]==225):
        im[k,f]=(220,220,220.)


#guassian blur
height,width = im.shape[:2]
filter = np.zeros((height,width))
cv2.fillConvexPoly(filter,np.array(c_[y, x],dtype = 'int32'),1)

filter = cv2.GaussianBlur(filter,(31,31),0)

# Erosion to reduce blur size
kernel = np.ones((10,10),np.uint8)
filter = cv2.erode(filter,kernel,iterations = 1)
alpha=np.zeros([height,width,3],dtype='float64')
alpha[:,:,0]=filter
alpha[:,:,1]=filter
alpha[:,:,2]=filter

imsave('gaosiblur3131.jpg',(alpha*im+(1-alpha)*im2).astype('uint8'))