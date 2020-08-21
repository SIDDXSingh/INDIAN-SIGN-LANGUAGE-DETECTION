#TEAM MEMBERS:
#Prathmesh Ringe
#Himanshu Mahanjan
#Siddharth Singh
#Mokshada


#DETECTION OF INDIAN SIGN LANGUAGE

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

cv2.namedWindow("a",cv2.WINDOW_NORMAL)
cv2.namedWindow("b",cv2.WINDOW_NORMAL)

while(1):

    # Take each frame
    _, img = cap.read()
    img = cv2.flip(img, 1)

    #finding region of interest
    roi=img[100:300,100:300]
    cv2.rectangle(img,(100,100),(300,300),(0,255,0),0)

    #Finding HSV values for image
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


    # define range of skin color in HSV
    lower_red = np.array([0,0,29])
    upper_red = np.array([25,249,254])



    # Threshold the HSV image to get only color of skin
    ths = cv2.inRange(hsv, lower_red, upper_red)

    #Applying Morphological Transformations to improve mask
    k=np.ones((3,3),dtype='uint8')
    ths=cv2.dilate(ths,k ,iterations=5)
    ths=cv2.morphologyEx(ths, cv2.MORPH_CLOSE, k)

    #Finding The contours
    contours, hierarchy = cv2.findContours(ths,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=0:

        #Finding Contour for hand
        cnt=max(contours,key=cv2.contourArea)

        #Now we find its convex hull
        hull = cv2.convexHull(cnt)

        #Finding area covered by the convex hull and the contour around the hand
        hull_area =cv2.contourArea(hull)
        cntarea=cv2.contourArea(cnt)


        #Drawing the cinvex Hull
        roi=cv2.drawContours(roi,[hull],-1,(0,0,255),3)

        #Finding defects.
        #hull1 is same as hull but with returnPoints argument defined as false
        hull1 = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull1)

        # 'm' is a variable to count number of defects in the hand
        m=0

        #Difference of contour area and hull area and then finding the ratio of difference and hull area
        ratio=((-cntarea+hull_area)/cntarea)*100  if cntarea!=0 else np.inf


        #Finding Straight Bounding Rectangle and then the aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h


        #Extent is the ratio of hull area and area of Straight Bounding Rectangle
        extent = float(hull_area) / rect_area*100
        aspect_ratio = float(w) / h

        #A is the variable to store the angle between the fingers which is useful in finding number of defects
        A=0


        if(str(type(defects))!="<class 'NoneType'>"):
            for i in range(defects.shape[0]):


                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                #defining a triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                #Finding the angle using cosine rule
                A=math.acos(((b**2 + c**2) - (a**2))/(2*b*c))*57

                #Obtuse angles do not count as defects therefore we reject them
                if(A<90):
                    m=m+1
                    cv2.circle(roi,far,3,(0,0,255),-1)



        #Using various ratios found above to differentiate between the numbers
        if (m==0):

            #Note the difference of extent between 1 and 6 is not much so there maybe some error but it works most of the time
            if(ratio >12):
                if (extent<72):
                    cv2.putText(img,'1',(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3,cv2.LINE_AA)
                else:
                    cv2.putText(img,'6',(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3,cv2.LINE_AA)


            else:
                if(aspect_ratio*100<70):
                    cv2.putText(img,'9',(0,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3,cv2.LINE_AA)
                elif(aspect_ratio*100>80):
                    cv2.putText(img, '0', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)


        elif (m==1):
            if (aspect_ratio*100<66):
                cv2.putText(img, '2', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(img, '7', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)


        elif(m==2):
            if (extent<77):
                cv2.putText(img, '8', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(img, '3', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)


        elif(m==3):
            cv2.putText(img, '4', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)


        elif(m==4):
            cv2.putText(img, '5', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)




    cv2.imshow('a',img)
    cv2.imshow('b',ths)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
