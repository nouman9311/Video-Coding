import numpy as np
import cv2
#import cPickle as pickle  #for python2
import pickle   #for python3
import time as time
import utilityFunctions as utFuncs

VERSION = 5.0
print("File Version: ", VERSION)

#Duration in seconds
DURATTION = 10.0
#Program to capture video from a camera and store it in an recording file, in Python txt format, using cPickle
#This is a framework for a simple video encoder to build.
#It writes into file 'videorecord.txt'
#Gerald Schuller, April 2015

cv2.namedWindow('Original')
cv2.namedWindow('Y Component')
cv2.namedWindow('Cb Component')
cv2.namedWindow('Cr Component')

cap = cv2.VideoCapture(0)

f1=open('videorecord.txt', 'wb')
f2=open('videorecordfloat.txt', 'wb')

end_time = time.time() + DURATTION;

frames=0;
rec_video = False

#Some more changes for testing
#:) :) :)

#Process 25 frames:
while time.time() <= end_time:
    ret, frame = cap.read()
    framerec=np.zeros(frame.shape)
    #to count frames per second
    frames=frames+1
    if ret==True:
        [Y,Cb,Cr] = utFuncs.getYCC(frame)

        cv2.imshow('Original',frame)
        cv2.imshow('Y Component',Y)
        cv2.imshow('Cb Component',Cb)
        cv2.imshow('Cr Component',Cr)


        #Here goes the processing to reduce data... 
        #reducedYCC = frameYCC.copy()
        #reduced=np.array(reduced,dtype='uint8')# for the Cb and Cr components use the int8 type
        #"Serialize" the captured video frame (convert it to a string)

        #With out scaling up
        # YrReduced= np.array(np.uint(Y),dtype='uint8')
        # CbReduced=np.array(np.int8(Cb),dtype='int8')
        # CrReduced=np.array(np.int8(Cr),dtype='int8')

        #With scaling up  cb= -1 to 1
        YrReduced= np.array(np.uint8(Y*255),dtype='uint8') 
        CbReduced=np.array(np.int8(Cb*127),dtype='int8') # -127 to +127    1*127= 255 
        CrReduced=np.array(np.int8(Cr*127),dtype='int8')

        #using pickle, and write/append it to file f:
        pickle.dump(Y,f2,-1)
        pickle.dump(Cb,f2,-1)
        pickle.dump(Cr,f2,-1)

        #using pickle, and write/append it to file f:
        pickle.dump(YrReduced,f1,-1)
        pickle.dump(CbReduced,f1,-1)
        pickle.dump(CrReduced,f1,-1)
    
        #displaying Components
        # framerec = utFuncs.convertFrameToRGB(frame,framerec)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# Release everything if job is finished


# After Finishing the task Get Sixe of file by using Utility Functions
# filesize=os.path.getsize("c:/Users/rfaheem/Desktop/VideoCodingSharedSeminars/vc2020/Seminar 1/videorecord.txt")
# filesize=utFuncs.getSize("c:/Users/rfaheem/Desktop/VideoCodingSharedSeminars/vc2020/Seminar 1/videorecord.txt")
filesize1=utFuncs.getSize("e:/Summer Semester 2020/Video Coding/Seminar stuff/Seminar 1/videorecord.txt")
print("File sixe of videorecord.txt is (MegaBytes)", filesize1/(1024*1024))

filesize2=utFuncs.getSize("e:/Summer Semester 2020/Video Coding/Seminar stuff/Seminar 1/videorecordfloat.txt")
print("File size of videorecordfloat.txt is (MegaBytes)", filesize2/(1024*1024))

print("Frames per sec: ", frames//DURATTION)
cap.release()
f1.close()
f1.close()
cv2.destroyAllWindows()
