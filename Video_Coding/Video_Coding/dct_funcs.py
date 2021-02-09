import numpy as np

def test_function_removeZeros(x, factor):

    r, c = x.shape

    for k in range(factor, c + 1, factor):
        # x = np.delete(x, [k, k+1, k+2, k+3, k+4, k+5],  axis=1)
        x = np.delete(x, np.arange(k, k + (8-factor)), axis=1)

    # print x

    for l in range(factor, r + 1, factor):
        # x = np.delete(x, [[l], [l+1], [l+2], [l+3], [l+4], [l+5]],  axis=0)
        x = np.delete(x, np.arange(l, l + (8-factor)).T, axis=0)

    # print x

    return x


def DCTRemoveZeros(frame, f):
    r,c=frame.shape
    rframe = np.zeros(((r//8)*f, (c//8)*f, frame.shape[2]))
    #print rframe.shape

    rframe = test_function_removeZeros(frame, f)

    return rframe#.astype('int8')