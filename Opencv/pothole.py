import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread("/Users/asmitabasak/Downloads/index4.jpeg")

def rescale(frame,scale=3):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimen=(width,height)
    return cv.resize(frame,dimen,interpolation=cv.INTER_AREA)
img1=rescale(img,3)
roi=img1[120:800,50:850]
cv.imshow('roi',roi)
img1=cv.cvtColor(roi,cv.COLOR_BGR2GRAY)

blur=cv.GaussianBlur(img1,(23,23),11)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(blur,kernel,iterations = 1)
img2=cv.blur(erosion,(9,9))

cv.imshow("BLUR",img2)
ret,thresh=cv.threshold(img2,127,255,cv.THRESH_BINARY)
cv.imshow("Thresh",thresh)
contours, hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
# img2=img.copy()
print(f'{len(contours)} contours found')
blank=np.zeros(img1.shape,dtype='uint8')
cv.drawContours(blank,contours,-1,(255,255,255),1)
cv.imshow('Contours',blank)
cv.waitKey(0)
