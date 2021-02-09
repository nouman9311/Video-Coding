import numpy as np
import os
import scipy.fftpack as sft
from scipy.fftpack import dct, idct

###seminar 3
def view_as_block(img, blocksize=(8, 8)):
    for i in range(0, img.shape[0], blocksize[0]):  # TBD
        for j in range(0, img.shape[1], blocksize[1]):  # TBD
            block = img[i:i+blocksize[0], j:j+blocksize[1]]
            yield i, j, block

###Seminar 3
def dct_2d(block,pfactor,qfactor):
    """ perform a 2d DCT for the given block """
    # use dct from scipy.fftpack
    q=qfactor
    p=pfactor
    r, c = block.shape  

    # For rows:
    Mr = np.ones((r, 1))
    Mr[int(r/q):r, 0] = np.zeros(int(p/q*r))
    # For columns:
    Mc = np.ones((1, c))
    Mc[0, int(c/q):c] = np.zeros(int(p/q*c))
    # Together:
    M = np.dot(Mr, Mc)

    FirstDCT = dct(block, axis=1, norm='ortho')
    SecondDCT = (dct(FirstDCT, axis=0, norm='ortho'))
    reduced= SecondDCT*M 
    reduced=reduced[0:8//q,0:8//q] 
    return reduced# TBD

###seminar 3
def idct_2d(block):
    """ perform a 2d inverse DCT for a given block"""
    # use idct from scipy.fftpack
    idctFirst = idct(block, axis=0, norm='ortho')
    idctSecond = idct(idctFirst, axis=1, norm='ortho')
    return idctSecond  # TBD in a similar way than dct_2d

##seminar3
def getNewYCC(frame):
    # Here goes conversion
    Y = (0.114*frame[:, :, 0]+0.587*frame[:, :, 1]+0.299*frame[:, :, 2])

    Cb = (0.4997*frame[:, :, 0]-0.33107*frame[:, :, 1]-0.16864*frame[:, :, 2])

    Cr = (-0.081282*frame[:, :, 0]-0.418531 *
          frame[:, :, 1]+0.499813*frame[:, :, 2])
    return Y, Cb, Cr

# used


def getYCC(frame):
    """Converts an RGB image frame into Y Cb and Cr Components

    Parameters
    ----------
    frame : 3 Dimentional Numpy Array
    Returns
    -------
    Y
        1-D Numpy array of type uint8
    Cb
        1-D Numpy array of type int8
    Cr
        1-D Numpy array of type int8
    """
    # Y=np.array(np.zeros(frame[:,:,0].shape),dtype='uint8')
    Y = (np.uint8(0.114*frame[:, :, 0])+np.uint8(0.587 *
                                                 frame[:, :, 1])+np.uint8(0.299*frame[:, :, 2]))/255

    #Cb = 0.564(B-Y)
    Cb = np.array(np.zeros(frame[:, :, 0].shape), dtype='int8')
    Cb = (0.564*(frame[:, :, 0]/255-Y))

    #Cr = (0.713(R-Y))
    Cr = np.array(np.zeros(frame[:, :, 2].shape), dtype='int8')
    Cr = (0.713*(frame[:, :, 2]/255-Y))
    return Y, Cb, Cr


# used
def convertToRGBFrame(Y, Cb, Cr):
    """Converts Y, Cb and Cr components into an RGB frame

    Parameters
    ----------
    Y
        1-D Numpy array of type uint8
    Cb
        1-D Numpy array of type int8
    Cr
        1-D Numpy array of type int8
    Returns
    -------
    frameRGB
        3-D Numpy array of type
    """
    frameRGB = np.zeros((Y.shape[0], Y.shape[1], 3))
    # #B = (Cb/0.564 + Y)
    B = (Y+(1.772*((Cb))))
    # #R = (Cr/0.713 + Y)
    R = (Y + (1.402*(Cr)))
    # #G = (Y - 0.114B - 0.299R)/0.587
    G = (Y-(0.34414*(Cb))-(0.71414*(Cr)))

    # R=np.int8(Y + 1.4025*Cr)
    # G=np.int8(Y -0.34434*Cb-0.7144*Cr)
    # B=np.int8(Y + 1.7731*Cb)

    frameRGB[:, :, 0] = B
    frameRGB[:, :, 1] = G
    frameRGB[:, :, 2] = R
    return frameRGB
# used


def getSize(filename):
    st = os.stat(filename)
    return st.st_size

# put everything together in one method


def chroma_subsampling_4_2_0(Y, C_b, C_r, N):
    """ returns each component with applied 4:2:0 sampling including zeros"""
    # sub sample Cb, Cr components, according to 4:2:0 method:
    C_b_s = np.zeros((Y.shape[0], Y.shape[1]))  # TBD
    C_b_s[::N, ::N] = C_b[::N, ::N]
    C_r_s = np.zeros((Y.shape[0], Y.shape[1]))  # TBD
    C_r_s[::N, ::N] = C_r[::N, ::N]

    assert((np.array(C_b_s.shape[0:2]) == Y.shape[0:2]).all())
    assert((np.array(C_r_s.shape[0:2]) == Y.shape[0:2]).all())
    return np.array([Y, C_b_s, C_r_s])

# put everything together in one method


def chroma_subsampling_4_2_0_comp(Y, C_b, C_r, N):
    """ returns each component with applied 4:2:0 sampling without zeros"""
    # sub sample Cb, Cr components, according to 4:2:0 method:
    C_b_s = np.zeros((Y.shape[0]//N, Y.shape[1]//N))  # TBD
    C_b_s[::, ::] = C_b[::N, ::N]
    C_r_s = np.zeros((Y.shape[0]//N, Y.shape[1]//N))  # TBD
    C_r_s[::, ::] = C_r[::N, ::N]

    assert((np.array(C_b_s.shape[0:2]) * N == Y.shape[0:2]).all())
    assert((np.array(C_r_s.shape[0:2]) * N == Y.shape[0:2]).all())
    return np.array([Y, C_b_s, C_r_s])


def chroma_upsample_4_2_0(Y, C_b, C_r, N):
    """ returns each component with applied 4:2:0 sampling"""

    # sub sample Cb, Cr components, according to 4:2:0 method:
    C_b_s = np.zeros((Y.shape[0], Y.shape[1]))  # TBD
    C_b_s[::N, ::N] = C_b[::, ::]
    C_r_s = np.zeros((Y.shape[0], Y.shape[1]))  # TBD
    C_r_s[::N, ::N] = C_r[::, ::]

    assert((np.array(C_b_s.shape[0:2]) == Y.shape[0:2]).all())
    assert((np.array(C_r_s.shape[0:2]) == Y.shape[0:2]).all())
    return np.array([Y, C_b_s, C_r_s])


def dct_filtering(Y, C_b, C_r):

    r, c = Y.shape
    # For rows:
    Mr = np.ones((r, 1))
    Mr[int(r/4.0):r, 0] = np.zeros(int(3.0/4.0*r))
    # For columns:
    Mc = np.ones((1, c))
    Mc[0, int(c/4.0):c] = np.zeros(int(3.0/4.0*c))
    # Together:
    M = np.dot(Mr, Mc)

    r1, c1 = C_b.shape
    # For rows:
    Mr1 = np.ones((r1, 1))
    Mr1[int(r1/4.0):r1, 0] = np.zeros(int(3.0/4.0*r1))
    # For columns:
    Mc1 = np.ones((1, c1))
    Mc1[0, int(c1/4.0):c1] = np.zeros(int(3.0/4.0*c1))
    # Together:
    M1 = np.dot(Mr1, Mc1)

    # while i < c:
    #     Y8=Y[i:i+8, i:i+8]  #8 by 8 block    Y[0:8,0:8] for i= 0    Y[8:16,8:16] for i= 1, Y[16:32,16:32] for i= 2,
    #     X=sft.dct(Y8,axis=1,norm='ortho')
    #     X=sft.dct(X,axis=0,norm='ortho')
    #     Y[i:i+8, i:i+8]= X
    #     i+=8

    YD = sft.dct(Y, axis=1, norm='ortho')
    YDD = sft.dct(YD, axis=0, norm='ortho')

    CbD = sft.dct(C_b, axis=1, norm='ortho')
    CbDD = sft.dct(CbD, axis=0, norm='ortho')

    CrD = sft.dct(C_r, axis=1, norm='ortho')
    CrDD = sft.dct(CrD, axis=0, norm='ortho')

    YDD = YDD*M  # keeping only low freq content
    CbDD = CbDD*M1  # keeping only low freq content
    CrDD = CrDD*M1  # keeping only low freq content

    return YDD, CbDD, CrDD


def inverseDCT(Y, Cb, Cr):

    YD = sft.idct(Y, axis=1, norm='ortho')
    YDD = sft.idct(YD, axis=0, norm='ortho')

    CbD = sft.idct(Cb, axis=1, norm='ortho')
    CbDD = sft.idct(CbD, axis=0, norm='ortho')

    CrD = sft.idct(Cr, axis=1, norm='ortho')
    CrDD = sft.idct(CrD, axis=0, norm='ortho')

    return YDD, CbDD, CrDD

