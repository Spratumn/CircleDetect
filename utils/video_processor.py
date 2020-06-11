import cv2

cap = cv2.VideoCapture('C:/Users/QIU/Desktop/Code/CircleDetect/data/video/moulde1.mp4')
frame_count = 217
while True:
    ret, frame = cap.read()
    if ret is not True:
        print("That's all!")
        exit()
    else:
        frame_name = 'C:/Users/QIU/Desktop/Code/CircleDetect/data/dataset' + '/' + str(frame_count) + '.jpg'
        cv2.imwrite(frame_name, frame)
        frame_count += 1
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
