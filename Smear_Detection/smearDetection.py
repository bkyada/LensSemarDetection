import cv2
from matplotlib import pyplot as plt
import numpy
import glob
import imutils
from skimage.filters import threshold_adaptive
from imutils import contours
from skimage import measure

#importing all the images in a folder
cv_img=[];
avg_img=numpy.zeros((500,500,3),numpy.float);
for img in glob.glob("393408652.jpg"):
    a=cv2.imread(img);
    b=imutils.resize(a,width=500,height=500);
    print("Image "+img+" imported and resized");
                    
    cv_img.append(b);
    #cv2.imshow('image',b);
    #cv2.waitKey(0);
               
for img in cv_img:
    img1=cv2.GaussianBlur(img,(3,3),0);
    imarr=numpy.array(img1,dtype=numpy.float);
    #avg_img = avg_img+imarr/len(cv_img);
    avg_img = avg_img+imarr;

avg_img = numpy.array(numpy.round(avg_img),dtype=numpy.uint8);
avg = 1;
cv2.imwrite("Average.jpg",avg_img);
#displaying the average image           
if avg==1:
    cv2.imshow("Average",avg_img);
    cv2.waitKey(0);
    
avg_img_grey = cv2.cvtColor(avg_img,cv2.COLOR_BGR2GRAY);
thresh_img = cv2.threshold(avg_img_grey, 100, 255, cv2.THRESH_BINARY)[1];
thresh_img = cv2.adaptiveThreshold(avg_img_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2);     
thresh_img = cv2.threshold(avg_img_grey, 100, 255, cv2.THRESH_BINARY_INV)[1];                         
                                  
if avg==1:
     cv2.imshow("Threshold Gaussian",thresh_img);
     cv2.imwrite("ThresholdImg.jpg",thresh_img);          
     cv2.waitKey(0);
     
edge1 = cv2.Canny(thresh_img,75,100);               
edge = edge1;
if avg==1:
   cv2.imshow("Edge Detection",edge);
   cv2.imwrite("EdgesDetected.jpg",edge);  
   cv2.waitKey(0); 

cnts,hier = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE);
cv2.drawContours(avg_img,cnts,-1,(0,255,255),2);
if avg==1:
   cv2.imshow("Cont",avg_img);
   cv2.imwrite("AllContours.jpg",avg_img);          
   cv2.waitKey(0); 
                            
#cv2.drawContours(avg_img,cnts,-1,(0,255,0),3);
list1=[];
mask_img=numpy.zeros((500,500,1),numpy.float);
                    
oimg=cv2.imread("393408652.jpg");
oimg=imutils.resize(oimg,width=500,height=500);               
for c in cnts:
    p = cv2.arcLength(c,True);
    approx = cv2.approxPolyDP(c,0.02*p,True);
    
    (x,y),radius=cv2.minEnclosingCircle(c);
    radius = int(radius);
    if abs(cv2.contourArea(c)-(3.14*(radius**2))) <200 and cv2.contourArea(c)>200:
        cv2.drawContours(oimg,[approx],-1,(255,255,0),2);
        cv2.drawContours(mask_img,[approx],-1,(255,255,255),-1);                
        list1.append(c);   
if avg==1:
    cv2.imshow("FinalResult",oimg);
    cv2.waitKey(0);          
    cv2.imshow("Mask",mask_img);          
           
    cv2.waitKey(0); 
    cv2.imwrite("FinalResult.jpg",oimg);
    cv2.imwrite("Mask.jpg",mask_img);           
cv2.destroyAllWindows();                

                     
