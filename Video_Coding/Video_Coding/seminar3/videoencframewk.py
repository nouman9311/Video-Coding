import numpy as np
import cv2
# import cPickle as pickle  #for python2
import pickle  # for python3
import utilityFunctions as utFuncs
from utilityFunctions import view_as_block
import scipy.signal
from decodeAndDisplay import decodeAndDisplay

VERSION = 5.0
print("File Version: ", VERSION)
# sampling fator
N = 2
# Program to capture video from a camera and store it in an recording file, in Python txt format, using cPickle
# This is a framework for a simple video encoder to build.
# It writes into file 'videorecord.txt'

cv2.namedWindow('Original')
# cv2.namedWindow('Y Component')
# cv2.namedWindow('Cb Component')
# cv2.namedWindow('Cr Component')

# Low Pass Kernel:
# rectangular filter kernel:
filt1 = np.ones((2, 2))/2
# Triangular filter kernel:
pyramid_filt = scipy.signal.convolve2d(filt1, filt1)/2
filt = pyramid_filt

filteron = True

dctOn = True  # to turn dct on and off


cap = cv2.VideoCapture(0)

f1 = open('videorecord.txt', 'wb')
f2 = open('videorecord_DS.txt', 'wb')
f3 = open('videorecord_DS_compressed.txt', 'wb')
# f4=open('videorecord_DS_compressed_dct.txt','wb')


frames = 0
rec_video = True

# Some more changes for testing
#:) :) :)

# Process 25 frames:
while frames < 500:
    ret, frame = cap.read()
    # to count frames per second
    frames = frames+1  # for counting frames and runnig the loop
    if ret == True:
        framerec = np.zeros(frame.shape)
        [Y, Cb, Cr] = utFuncs.getNewYCC(frame)

        # cv2.imshow('Original',frame)
        # cv2.imshow('Y Component',Y)
        # cv2.imshow('Cb Component',Cb)
        # cv2.imshow('Cr Component',Cr)

        # Here goes the processing to reduce data...
        # reducedYCC = frameYCC.copy()
        # reduced=np.array(reduced,dtype='uint8')# for the Cb and Cr components use the int8 type
        # "Serialize" the captured video frame (convert it to a string)

        # With out scaling up
        # YrReduced= np.array(np.uint(Y),dtype='uint8')
        # CbReduced=np.array(np.int8(Cb),dtype='int8')
        # CrReduced=np.array(np.int8(Cr),dtype='int8')

        if filteron == True:
            Cb = scipy.signal.convolve2d(Cb, filt, mode='same')
            Cr = scipy.signal.convolve2d(Cr, filt, mode='same')

        # returns downsampled with zeros
        [Y, Cbds, Crds] = utFuncs.chroma_subsampling_4_2_0(Y, Cb, Cr, N)
        # returns downsampled without zeros
        [Y, Cbds_comp, Crds_comp] = utFuncs.chroma_subsampling_4_2_0_comp(
            Y, Cb, Cr, N)

        
        p=3   #int he for p/q
        q=4  #for removing p/q frequencies e.g removing 3/4 = onöy 25% lower frequencies p=3,q=4,,,, for 1/2=50% p=1, q=2 
        
        if dctOn:
            l=0
            m=0
            newY=np.zeros((Y.shape[0]//q,Y.shape[1]//q))   #size of new y would be 1/4
            for i, j, b in view_as_block(Y, (8, 8)):
                l=i//q
                m=j//q
                # Y[i:i+8, j:j+8] = utFuncs.dct_2d(b)
                newY[l:l+8//q,m:m+8//q]=utFuncs.dct_2d(b,p,q)
     
            l=0
            m=0
            newCb=np.zeros((Cbds_comp.shape[0]//q,Cbds_comp.shape[1]//q))
            for i, j, b in view_as_block(Cbds_comp, (8, 8)):
                l=i//q
                m=j//q
                # Cbds_comp[i:i+8, j:j+8] = utFuncs.dct_2d(b)
                newCb[l:l+8//q,m:m+8//q]=utFuncs.dct_2d(b,p,q)

            newCr=np.zeros((Crds_comp.shape[0]//q,Crds_comp.shape[1]//q))
            for i, j, b in view_as_block(Crds_comp, (8, 8)):
                l=i//q
                m=j//q
                # Crds_comp[i:i+8, j:j+8] = utFuncs.dct_2d(b)
                newCr[l:l+8//q,m:m+8//q]=utFuncs.dct_2d(b,p,q)

        
        # To take dct of full frame without blocks
        # [Y,Cbds_comp,Crds_comp] = utFuncs.dct_filtering(Y,Cbds_comp,Crds_comp)

        #cv2.imshow('Cb Component',Y)
        # cv2.imshow('Cb Component',Cbds)
        # cv2.imshow('Cr Component',Crds)
        # With scaling up  cb= -1 to 1
        YrReduced1 = np.array((Y))
        CbReduced1 = np.array((Cb))  # -127 to +127    1*127= 127
        CrReduced1 = np.array((Cr))

        # With scaling up  cb= -1 to 1
        YrReduced2 = np.array((Y))
        CbReduced2 = np.array((Cbds))  # -127 to +127    1*127= 127
        CrReduced2 = np.array((Crds))

        # With scaling up  cb= -1 to 1
        YrReduced3 = np.array((Y))
        CbReduced3 = np.array((Cbds_comp))  # -127 to +127    1*127= 127
        CrReduced3 = np.array((Crds_comp))

        # using pickle,dumping original YCbCr
        pickle.dump(YrReduced1, f1, -1)
        pickle.dump(CbReduced1, f1, -1)
        pickle.dump(CrReduced1, f1, -1)

        # using pickle,dumping Downsampled YCbCr
        pickle.dump(YrReduced2, f2, -1)
        pickle.dump(CbReduced2, f2, -1)
        pickle.dump(CrReduced2, f2, -1)

        # using pickle,dumping downsampled and compressed

        pickle.dump(newY, f3, -1)
        pickle.dump(newCb, f3, -1)
        pickle.dump(newCr, f3, -1)

        # displaying Components
        # framerec = utFuncs.convertFrameToRGB(frame,framerec)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Camera problem")
        break


# Release everything if job is finished


# After Finishing the task Get Sixe of file by using Utility Functions
f1size = utFuncs.getSize("videorecord.txt")
f2size = utFuncs.getSize("videorecord_DS.txt")
f3size = utFuncs.getSize("videorecord_DS_compressed.txt")
# filesize1=utFuncs.getSize("e:/Summer Semester 2020/Video Coding/Seminar stuff/Seminar 1/videorecord.txt")

print("File size of videorecord.txt is (MegaBytes)", f1size/(1024*1024))
print("File size of videorecord_DS.txt is (MegaBytes)", f2size/(1024*1024))
print("File size of videorecord_DS_compressed.txt is (MegaBytes)", f3size/(1024*1024))

try:
    print("Ratio = original: Downsampled: ", f1size/f2size)
    print("Ratio = original: Downsampled_Compressed: ", f1size/f3size)
except ZeroDivisionError:
    print("Printing Error")
    print(ZeroDivisionError)

# print("Total frames: ", frames)
cap.release()
f1.close()
f2.close()
f3.close()
cv2.destroyAllWindows()
print("I ran till the end")
