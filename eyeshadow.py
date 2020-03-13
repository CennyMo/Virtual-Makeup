
import dlib
from pylab import *
from scipy.interpolate import interp1d
from skimage import color
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance

#Source colour
R,G,B = (57.,30.,29.)

inten =0.5

lower_left_end = 5
upper_left_end = 11
lower_right_end = 16
upper_right_end = 22

def inter(lx=[],ly=[],k1='quadratic'):
	unew = np.arange(lx[0], lx[-1]+1, 1)
	f2 = interp1d(lx, ly, kind=k1)
	return (f2,unew)

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
            i = i + 1
            x.append(pt.x)
            y.append(pt.y)

        pos = (x[17], y[17])
        points.append(pos)
        for i in range(35,40):
            if i>=36 and i<=39:
                pos = (x[i], y[i])
                points.append(pos)
        pos = (x[17], y[17])
        points.append(pos)
        for i in range(17,40):
            if i>=18 and i<=21:
                pos = (x[i], y[i])
                points.append(pos)
        pos = (x[39], y[39])
        points.append(pos)

        for i in range(41, 46):
            if i >= 42 and i <= 45:
                pos = (x[i], y[i])
                points.append(pos)
        pos = (x[26], y[26])
        points.append(pos)
        pos = (x[42], y[42])
        points.append(pos)
        for i in range(21, 27):
            if i >= 22 and i <= 26:
                pos = (x[i], y[i])
                points.append(pos)
    return points

img = np.array(imread("Input.jpg"))
points=getpoint(img)
points=np.array(points)
im = img.copy();
im2 = img.copy();
point_down_x = np.array((points[:lower_left_end][:,0]))
point_down_y = np.array(points[:lower_left_end][:,1])
point_up_x = np.array(points[lower_left_end:upper_left_end][:,0])
point_up_y = np.array(points[lower_left_end:upper_left_end][:,1])
point_down_x_right = np.array((points[upper_left_end:lower_right_end][:,0]))
point_down_y_right = np.array(points[upper_left_end:lower_right_end][:,1])
point_up_x_right = np.array((points[lower_right_end:upper_right_end][:,0]))
point_up_y_right = np.array(points[lower_right_end:upper_right_end][:,1])

l_l = inter(point_down_x[:],point_down_y[:],'quadratic')
u_l = inter(point_up_x[:],point_up_y[:],'quadratic')
l_r = inter(point_down_x_right[:],point_down_y_right[:],'quadratic')
u_r = inter(point_up_x_right[:],point_up_y_right[:],'quadratic')
imshow(im)

point_down_y_max = max(point_down_y)
point_up_y_min = min(point_up_y)
offset_left = point_down_y_max - point_up_y_min
point_up_y[0] += offset_left*0.6
point_up_y[1] += offset_left*0.3
point_up_y[2] += offset_left*0.3
point_up_y[3] += offset_left*0.3
point_up_y[4] += offset_left*0.3
point_down_y[0] += offset_left*0.6

point_down_y_right_max = max(point_down_y_right)
point_up_y_right_min = min(point_up_y_right)
offset_right = point_down_y_right_max - point_up_y_right_min
point_up_y_right[-1] += offset_right*0.6
point_up_y_right[1] += offset_right*0.3
point_up_y_right[2] += offset_right*0.3
point_up_y_right[3] += offset_right*0.3
point_up_y_right[4] += offset_right*0.3
point_down_y_right[-1] += offset_right*0.6

figure()

height,width = im.shape[:2]

L,A,bB = 0,0,0

x = []
y = []


for i in l_l[1]:
    for j in range(int(u_l[0](i)),int(l_l[0](i))):
            x.append(int(j))
            y.append(int(i))

for i in l_r[1]:
    for j in range(int(u_r[0](i)),int(l_r[0](i))):
            x.append(int(j))
            y.append(int(i))

val = color.rgb2lab((im[x,y]/255.).reshape(len(x),1,3)).reshape(len(x),3)
L = mean(val[:,0])
A = mean(val[:,1])
bB = mean(val[:,2])

rgbmean = (im[x,y])
rmean = mean(rgbmean[:,0])
gmean = mean(rgbmean[:,1])
bmean = mean(rgbmean[:,2])
# print rmean, gmean, bmean

L,A,bB = color.rgb2lab(np.array((rmean/255.,gmean/255.,bmean/255.)).reshape(1,1,3)).reshape(3,)
L1,A1,B1 = color.rgb2lab(np.array((R/255.,G/255.,B/255.)).reshape(1,1,3)).reshape(3,)
val[:,0] += (L1-L)*inten
val[:,1] += (A1-A)*inten
val[:,2] += (B1-bB)*inten

image_blank = img.copy();
image_blank *= 0
image_blank[x,y] = color.lab2rgb(val.reshape(len(x),1,3)).reshape(len(x),3)*255

original = color.rgb2lab((im[x,y]*0/255.).reshape(len(x),1,3)).reshape(len(x),3)

tobeadded = color.rgb2lab((image_blank[x,y]/255.).reshape(len(x),1,3)).reshape(len(x),3)
original += tobeadded
im[x,y] = color.lab2rgb(original.reshape(len(x),1,3)).reshape(len(x),3)*255

# Blur Filter
filter = np.zeros((height,width))
cv2.fillConvexPoly(filter,np.array(c_[y, x],dtype = 'int32'),1)
#cv2.fillConvexPoly(filter,np.array(c_[yright, xright],dtype = 'int32'),1)

filter = cv2.GaussianBlur(filter,(81,81),0)

# Erosion to reduce blur size
kernel = np.ones((10,10),np.uint8)
filter = cv2.erode(filter,kernel,iterations = 1)

alpha=np.zeros([height,width,3],dtype='float64')
alpha[:,:,0]=filter
alpha[:,:,1]=filter
alpha[:,:,2]=filter


#imshow((alpha*im+(1-alpha)*im2).astype('uint8'))
imsave('ou4.jpg',(alpha*im+(1-alpha)*im2).astype('uint8'))
#show()