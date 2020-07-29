# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:06:13 2020

@author: himad
"""

import cv2
import numpy as np
from transform import four_point_transform
from Sorting import sort_contours


ANSWER_KEY = {0: 1, 
              1: 4, 
              2: 0, 
              3: 3, 
              4: 1}

image = cv2.imread("pic1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged= cv2.Canny(blurred, 50, 200)

#cv2.imshow("edged", edged)

cnts, hierarchy= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) >0:
    
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        
        if len(approx)==4:
            doCnt=approx
            break
        

paper = four_point_transform(image, doCnt.reshape(4,2))
warped = four_point_transform(gray, doCnt.reshape(4,2))
 
_, thresh  = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)

#cv2.imshow("warped", warped)
#cv2.imshow("thresh", thresh)
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

questionCnts = []


for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    
    if w>=20 and h>=20 and ar>=.9 and ar<=1.1:
        questionCnts.append(c)
        

#cv2.drawContours(paper, questionCnts, -1, (0,0,255), 3)
#cv2.imshow("COntour Ensure", paper)


questionCnts= sort_contours(questionCnts, method="top-to-bottom")[0]
correct=0

for (q,i) in enumerate(np.arange(0, len(questionCnts), 5)):
    
    cnts = sort_contours(questionCnts[i: i+5], method= "left-to-right")[0]
    bubbled=None
    
    #cv2.drawContours(paper, cnts, -1, cv, 3)
    #cv2.imshow("check", paper)
    ans = False
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    for (j,c) in enumerate(cnts):
        
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask = mask)
        total = cv2.countNonZero(mask)
       # if q==2 and j==1:
         #   cv2.imshow("check", mask)
        if bubbled is None or total>bubbled[0]:
            bubbled = (total, j)
        
        if  bubbled[1]==k and ans==False:
            ans=True
            color= (0, 255, 0)
            correct +=1
            
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)


score = (correct) * 20
print("[INFO] score: {:.2f}%".format(score))

cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imshow("Original", image)
cv2.imshow("Exam", paper)


cv2.waitKey(0)
cv2.destroyAllWindows()