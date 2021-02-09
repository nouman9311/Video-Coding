import numpy as np
import cv2
# import cPickle as pickle  #for python2
import pickle  # for python3
import utilityFunctions as utFuncs
from utilityFunctions import view_as_block
import scipy.signal
# from decodeAndDisplay import decodeAndDisplay

VERSION = 3.0
print("File Version: ", VERSION)
# sampling fator
N = 2
# Program to capture video from a camera and store it in an recording file, in Python txt format, using cPickle
# This is a framework for a simple video encoder to build.
# It writes into file 'videorecord.txt'

# cv2.namedWindow('Original')
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


cap = cv2.VideoCapture(0)

# Get size of frame:
[retval, frame] = cap.read()
[rows, columns, d] = frame.shape
print(rows, columns)


# Prevous Y frame:
Yprev = np.zeros((rows, columns))
# Vectors for current frame as graphic:
framevectors = np.zeros((rows, columns, 3))
# motion vectors, for each block a 2-d vector:
mv = np.zeros((int(rows/8), int(columns/8), 2))


f1 = open('videorecord.txt', 'wb')
f2 = open('videorecord_DS.txt', 'wb')


frames = 0
rec_video = True
bestCorrArgValue = 70.0
# Process 25 frames:
while frames < 500:
    ret, frame = cap.read()
    [rows, columns, c] = frame.shape
    center=((frame.shape[0])//2,(frame.shape[1]//2) ) 
    # to count frames per second
    frames = frames+1  # for counting frames and runnig the loop
    if ret == True:
        framerec = np.zeros(frame.shape)
        [Y, Cb, Cr] = utFuncs.getNewYCC(frame)

        cv2.imshow('Original', frame/255+framevectors)

        if filteron == True:
            Cb = scipy.signal.convolve2d(Cb, filt, mode='same')
            Cr = scipy.signal.convolve2d(Cr, filt, mode='same')

        # returns downsampled
        [Y, Cbds, Crds] = utFuncs.chroma_subsampling_4_2_0(Y, Cb, Cr, N)

# Motion estimation, correlate current Y block with previous 16x16 block which contains current 8x8 block:
        # Start pixel for block wise motion estimation:
        block = np.array([8, 8])
# Block of  8X8 Values
        framevectors = np.zeros((rows, columns, 3))
# Initialize Frame Vector
        # for loops for the blocks:
        # print("for loops for the motion vectors:")
# Motion Estimation is staring from here
        for yblock in range(10):
            # print("yblock=",yblock)
            # here 200 and Bellow 300 Speciffies Middle Area of our Video
            block[0] = yblock*8+center[0]
            for xblock in range(10):
                block[1] = xblock*8+center[1]

                Yc = Y[block[0]:block[0]+8, block[1]:block[1]+8]
                # print("Yc= ",Yc)
                # previous block of 16*16
                Yp = Yprev[block[0]-8: block[0]+8, block[1]-8: block[1]+8]

                # Or i Could Start from same postion like bellow
                # Yp = Yprev[block[0]-4: block[0]+12, block[1]-4: block[1]+12]
                # print("Yp= ",Yp)
                # correlate both to find motion vector
                # print("Yp=",Yp)
                # print(Yc.shape)
                # Some high value for MAE for initialization:
#                

                Ycorr = scipy.signal.fftconvolve(
                    Yp, Yc[::-1, ::-1], mode='valid')
                # print Ycorr
                # print(np.argmax(Ycorr)) 
                Motion_Vector_Block = np.unravel_index(
                    np.argmax(Ycorr), (Ycorr.shape))


 # Motion Vector will have the pixel index where the block has maximum correlation
#Maximum correlation value will lead to the position where there is maximum similarity #
# between the current and the previous block hence motion estimation #
                if np.argmax(Ycorr) >= bestCorrArgValue:
                    mv = np.add(block, Motion_Vector_Block)
                    cv2.line(framevectors, tuple(block)[
                             ::-1], tuple(mv)[::-1], (1.0, 1.0, 255.0), 1)
                    
                    # print('here')
                    #secarg = np.zeros((np.shape(block)))
                    #cv2.line(framevectors, tuple(block)[::-1], tuple(block)[::-1],(1.0,1.0,1.0),3)
                #cv2.line(framevectors, (block[1], block[0]),(block[1] + mv[yblock, yblock, 1].astype(int), block[0] + mv[yblock, yblock, 0].astype(int)),(1.0, 1.0, 1.0));

        Yprev = Y.copy()
        # converting  images back to integer:
        frame = np.array(frame, dtype='uint8')

        # Ycorr=scipy.signal.correlate2d(Yp, Yc,mode='valid')
        # print("Ycorr= ", Ycorr)
        # motion vector:
        # 1-d Index of maximum
        # index1d=np.argmax(Ycorr)
        # print("inedx1d=",index1d)
        # convert it to 2d index:
        # index2d=np.unravel_index(index1d,(9,9))
        # print("arg max of correlation: ", index2d)
        # 2-d index minus center coordinates (4,4) is the motion vector
        # print("mv=",mv[0,0,:])
        # print(np.subtract(index2d,(4,4)))
        # mv[0,0,:]=np.subtract(index2d,(4,4))
        # print("mv[0,0,:]=",mv[0,0,:])
        # print(tuple(np.add(block,mv[0,0,:]).astype(int)))
        # cv2.line(framevectors, block, (block[0]+mv[0],block[1]+mv[1]),(1.0,1.0,1.0))
        #         cv2.line(framevectors, (block[1], block[0]), (block[1]+mv[yblock, yblock, 1].astype(
        #             int), block[0]+mv[yblock, yblock, 0].astype(int)), (1.0, 1.0, 1.0))
        # Yprev = Y.copy()

        # cv2.imshow('Y Component', Y)
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

        # using pickle,dumping original YCbCr
        pickle.dump(YrReduced1, f1, -1)
        pickle.dump(CbReduced1, f1, -1)
        pickle.dump(CrReduced1, f1, -1)

        # using pickle,dumping Downsampled YCbCr
        pickle.dump(YrReduced2, f2, -1)
        pickle.dump(CbReduced2, f2, -1)
        pickle.dump(CrReduced2, f2, -1)

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
# f3size = utFuncs.getSize("videorecord_DS_compressed.txt")
# filesize1=utFuncs.getSize("e:/Summer Semester 2020/Video Coding/Seminar stuff/Seminar 1/videorecord.txt")

print("File size of videorecord.txt is (MegaBytes)", f1size/(1024*1024))
print("File size of videorecord_DS.txt is (MegaBytes)", f2size/(1024*1024))

try:
    print("Ratio = original: Downsampled: ", f1size/f2size)
    # print("Ratio = original: Downsampled_Compressed: ", f1size/f3size)
except ZeroDivisionError:
    print("Printing Error")
    print(ZeroDivisionError)

# print("Total frames: ", frames)
cap.release()
f1.close()
f2.close()

cv2.destroyAllWindows()
print("I ran till the end")
