import cv2
import matplotlib as mtp
cap = cv2.VideoCapture(0)

[ret, frame] = cap.read()

if ret:
    cv2.imshow('photo', frame)

    cv2.imwrite('pycolorphoto.jpg', frame)

    while True:
        print('press q to quit')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("frame not captured")
