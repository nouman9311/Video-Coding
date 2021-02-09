import cv2
import sys

photo = cv2.imread(str(sys.argv[0]))

cv2.imshow('Photo', photo)

while True:
    print('press q to quit')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
