# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:09:15 2020

@author: himad
"""

import numpy as np
import cv2

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i=0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method== "bottom-to-top":
        i=1
        
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), 
                                key = lambda b:b[1][i], reverse= reverse))
    
    return (cnts, boundingBoxes)

def draw_contour(image, c,i):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
    
    return image

cv2.waitKey(0)
cv2.destroyAllWindows()













