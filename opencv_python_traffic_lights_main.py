#!/usr/bin/env python
# coding: utf-8




import cv2
import numpy as np
from matplotlib import pyplot as plt

#I'm using a jupyter notebook to create this so displaying the images requires a little function trickery

#grayscale plot or binary image plot
def gplot(img):
    plt.figure(figsize=(12,12))
    return plt.imshow(img,cmap="gray", vmin=0, vmax=255)

#convert BGR image from BGR to RGB then plot it
def iplot(img):
    plt.figure(figsize=(12,12))
    imgrgb= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return plt.imshow(imgrgb)





#Read in the target image and plot it:
target = cv2.imread('/Users/code/Downloads/CamVidLights/CamVidLights10.png')
iplot(target)





#Final approach works with: 10,1,2





#segmenting into a left and right image so I can more easily process the two traffic lights
#my process will most likely only work when one traffic light is in the image area being looked at
#although I could extend it by having it run the process again and looking for the second place confidence matches
#no longer cropping height wise however the variable names have remained to maintain working operation

(h, w) = target.shape[:2]
croppedleft=target[:int(h),:int(w/2)]
croppedright=target[:int(h),int(w/2):]
iplot(croppedleft)
iplot(croppedright)





#Here I apply Mean Shift to the images, however, I left the parameters what they were in the documentation tutorial
meanshiftedleft=cv2.pyrMeanShiftFiltering(croppedleft,20,40,2)
meanshiftedright=cv2.pyrMeanShiftFiltering(croppedright,20,40,2)
iplot(meanshiftedleft)
iplot(meanshiftedright)





#Now I convert the image to grayscale to use in canny edge detection, although it now occurs to me I don't have to do this
#as i believe the canny operator contains a "to-grayscale conversion"





meanshiftedleftgray=cv2.cvtColor(meanshiftedleft,cv2.COLOR_BGR2GRAY)





meanshiftedrightgray=cv2.cvtColor(meanshiftedright,cv2.COLOR_BGR2GRAY)





gplot(meanshiftedleftgray)
gplot(meanshiftedrightgray)





#using canny:





meanshiftedleftedge=cv2.Canny(meanshiftedleftgray,50,200)
meanshiftedrightedge=cv2.Canny(meanshiftedrightgray,50,200)





gplot(meanshiftedleftedge)
gplot(meanshiftedrightedge)





#Now I find contours in the image. I found that findContours and my solution worked best on edge detection instead of grayscale mean-shifted version
#hence the commented our findContours on the grayscale version





imagecontours, contours, hierarchy = cv2.findContours(meanshiftedleftedge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)





#imagecontours, contours, hierarchy = cv2.findContours(meanshiftedleftgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)





#Here I am intializing lists:
#That i am using to collect coordinates of circular and rectangular contours:
#The meanlist is used to determine the average intensity within a circular region which is used to help verify the circlular region is an illuminated light
#circleinside list is used to capture the circles found within rectangular regions which is helpful to determine if the rectangular region contains a traffic light





circlist=[]
rectlist=[]
meanlist=[]
complist=[]
circleinsidelist=[]





#Here I am looking for circlular regions within the image:
for cnt in contours:
    #I tried using hough circle transformation but got poor results as
    #the parameter numbers given to the hough circle function seemed liked magic and were very particluar to each photo
    #Thus i decided to use minimum encolosing circle contour selection
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    radius = int(radius)
    center = (int(x),int(y))
    area=cv2.contourArea(cnt)
    
    #Here I am matching the minimum enclosing circle to a perfect circle using matchShapes. 
    #Everything in this section done before  matchShapes is creating the contour for the perfect circle
    mask=np.zeros(meanshiftedleftgray.shape,np.uint8)
    cv2.circle(mask,center,radius,255,-1,cv2.LINE_AA)
    #pretty sure this will generate error if it finds multiple signals- like the red and yellow picture
    tempcontour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    #Also I am calculating the mean intensity value inside the circular region as mentioned previously
    mean_val=cv2.mean(meanshiftedleftgray,mask)[0]
    compare_val=cv2.matchShapes(cnt,tempcontour,1,0.0)

    #Now I run my conditional statement that checks to see if the circular region has 
    #a reasonably expected radial size, mean intensity value, and a matchShape value.
    
    #I have left all of these commented out condtional statements to show the various tweaking I had to do
    #in order to get it to work as best as possible on the images
    #I defintely see this necessary tweaking as a weak point of my solution and it limits the usability of my approach on a different set of images
    
    
    #if radius<20 and radius>10 and area>100:
    #if radius<20 and radius>5 and area>50:
    #if radius<15 and radius>5 and area>100 and mean_val>160 and compare_val<1:#i think this worked last
    #if radius<15 and radius>5 and area>100 and mean_val>100 and compare_val<1:#i think this worked last
    #if radius<15 and radius>5 and area>100 and mean_val>100 and compare_val<1: #not working on 3
    #if radius<20 and radius>3 and mean_val>100 and compare_val<1:  not working on 4
    if radius<20 and radius>3 and mean_val>40 and compare_val<1:  
        #Here I draw the circle on the image:
        cv2.circle(imagecontours,center,radius,(255,255,255),2)
        #Here i collect the coordinate information, intensity information, and match value info
        #of the circle to later be used 
        circlist.append([center,radius,compare_val])
        meanlist.append([mean_val])
        complist.append([compare_val])

    





gplot(imagecontours)





#Now I am looking for rectangular regions that also contain a circlualr region that was found previously

for cnt in contours: 
    #Finding rectangular regions
    x,y,w,h=cv2.boundingRect(cnt)
    centerrect=(int(x),int(y))
    dimrect=(int(w),int(h))
    arearect=w*h
    
    #Comparing rectangular regions to perfect rectangle
    
    mask=np.zeros(meanshiftedleftgray.shape,np.uint8)
    cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1,cv2.LINE_AA)
    tempcontour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    compare_val=cv2.matchShapes(cnt,tempcontour,1,0.0)

    #Checking to see if a circular region found previously exists in this rectangle
    # I think I'm doing something wrong with this piece of code
    #As sometimes my circular regions found preivously get ignored
    circlecheck=0
    tempcircle=[]
    
    for cent in circlist:
        result=cv2.pointPolygonTest(cnt,cent[0],False)

        if result==0 or result==1:
            circlecheck=1
            tempcircle=cent

    #Here I have my conditional statement to determine if the rectangular region is worth recording and draw:
    
    #Again I've left the various tweaking I had to do: 
    #if arearect>5000 and arearect<20000 and circlecheck==1:
    #for image 1 the rectangle is about 1800; 5000 was for image 10
    #if arearect>1500 and arearect<20000 and circlecheck==1 and compare_val<1.5:
    if arearect>500 and arearect<20000 and circlecheck==1 and compare_val<1.5:
        cv2.rectangle(imagecontours,(x,y),(x+w,y+h),(255,255,255),2)
        rectlist.append([centerrect,dimrect,compare_val])
        circleinsidelist.append([tempcircle])

        





#circleinsidelist





#tempcircle





gplot(imagecontours)





#circlist





#when it detects two close to each other it must be 





#circlist[0][0]





#rectlist





#len(rectlist)





#If multiple rectangles with circular were found I select the rectangle that had the best match with a perfect rectangular shape
#But in theory I should also include the match with the circlar
#And maybe even if the rectangular region best matches with a shape of a traffic light, so basically template matching on these 
#regions
rectlistfinal=[]
tempmatchglobal=999999
if len(rectlist)>1:
    for rect in rectlist:
        tempmatch=rect[2]
        if tempmatch<tempmatchglobal:
            reclistfinal=rect
            tempmatchglobal=tempmatch
else:
    reclistfinal=rectlist[0]





#circleinsidelist





#Here I have the final coordinates of my rectangle
reclistfinal





#determine the top left and bottom right of box
topleft=reclistfinal[0]
topleft





bottomright=(reclistfinal[0][0]+reclistfinal[1][0],reclistfinal[0][1]+reclistfinal[1][1])





bottomright





#circleinsidelist





#If multiple circular existed in the selected rectangular region
#I now find the best matched circule in this region

circlelistfinal=[]
tempmatchglobal=999999
if len(circleinsidelist)>1:
    for circle in circleinsidelist:
        tempmatch=circle[0][2]
        if tempmatch<tempmatchglobal:
            circlelistfinal=circle
            tempmatchglobal=tempmatch
else:
    circlelistfinal=circleinsidelist[0]





circlelistfinal





#Pulling out the center of circle:
circlelistfinalcenter=circlelistfinal[0][0]
circlelistfinalcenter





#Pulling out radius
circlelistfinalradius=circlelistfinal[0][1]
circlelistfinalradius





#Now I determine the color of the light:
#First I am going to create a mask from the selected circlular region
#It will be overlayed on the intial read-in image





mask=np.zeros(meanshiftedleftgray.shape,np.uint8)
cv2.circle(mask,circlelistfinalcenter,circlelistfinalradius,255,-1,cv2.LINE_AA)
#mean_val=cv2.mean(meanshiftedleftgray,mask)[0]





#mean_val





gplot(mask)





#Now I use this mask on the original color image:
maskcolor=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
tempimage=cv2.bitwise_and(meanshiftedleft,maskcolor)
iplot(tempimage)





#Now I backproject and count number of pixels for each color backprojected
#actually I decided to just avg the probabilities found





#Here I read in the different training sets for each color:

#WARNING YOU HAVE TO ENTER IN THE NEW PATHS:
#I did not get to dynamically coding this so it sleects the training images in the current folder
yellowtrain=cv2.imread('/Users/code/Downloads/CamVidLights/trainforyellow2.png')

greentrain = cv2.imread('/Users/code/Downloads/CamVidLights/trainfor12610.png')

redtrain=cv2.imread('/Users/code/Downloads/CamVidLights/trainforred.png')





#viewing the training set:
trainingset=[yellowtrain,greentrain,redtrain]
for img in trainingset:
    iplot(img)





#now I create a backprojection function
#will have to apply this to both left and right

def backprojection(img,train):
    
    hsvimg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    hsvtrain = cv2.cvtColor(train,cv2.COLOR_BGR2HSV)
    hsvtrainhist = cv2.calcHist([hsvtrain],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    cv2.normalize(hsvtrain,hsvtrain,0,255,cv2.NORM_MINMAX)
    
    probimg = cv2.calcBackProject([hsvimg],[0,1],hsvtrainhist,[0,180,0,256],1)
    avgprob=cv2.mean(probimg,mask)[0] 
    # if mask is not perfect i might get black pixels which will throw off avg prob
    return avgprob





#determing the color based on my color list
def findcolorindex(img,trainlist):
    temp=0
    index=0
    for i,t in enumerate(trainlist):
        tempavg=backprojection(img,t)
        print(tempavg)
        if tempavg>temp:
            index=i
            temp=tempavg

    return index





#printing out color based on index value
def findcolor(img,trainlist):
    
    cindex=findcolorindex(img,trainlist)
    
    if cindex==0:
        color='yellow'
    elif cindex==1:
        color='green'
    elif cindex==2:
        color='red'
    
    return color





findcolorindex(meanshiftedleft,trainingset)





findcolor(meanshiftedleft,trainingset)





#Finally I draw the rectangle on the original image
meanshiftedleftlocated=cv2.rectangle(target,topleft,bottomright,(0,0,255),2)





#Of course I would need to do all of this to the right hand side as well





iplot(meanshiftedleftlocated)

