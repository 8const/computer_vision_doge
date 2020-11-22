import cv2 
import imutils 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   

def frame_processing(frame):
    """ frame_processing(frame) -> doge frame;
    frame is a numpy.ndarray returned by cv2.imread() or cv2.VideoCapture.read();
    doge frame is a slightly mofified frame; such doge!
    """
    bread = cv2.imread('bread.jpg') 
    doge  = cv2.imread('doge.png') 

    frame = imutils.resize(frame, width=min(2256, frame.shape[1])) 

    #detect regions with faces
    faces = face_cascade.detectMultiScale(frame, 
                                          scaleFactor=1.3, 
                                          minSize=(200,200), 
                                          maxSize=(1500,1500)
                                          )
    #detect regions with walkers
    (walkers, _) = hog.detectMultiScale(frame,  
                                        winStride=(6, 6), 
                                        padding=(64, 64), 
                                        scale=1.30,
                                        ) 
    #replace faces with doge face 
    for (x, y, w, h) in faces:
        frame[y : y + h, x : x + w] = cv2.resize(doge, (w, h))

    #replace pedestrians with doge bread
    for (x, y, w, h) in walkers:
        frame[y : y + h, x : x + w] = cv2.resize(bread, (w, h))

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture('dogs.mp4')
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output.avi',
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          30,
                          (frame_width, frame_height)
                          )

    #loop through frames
    while(cap.isOpened()):
	ret, frame = cap.read()
	if (ret==True):
	    frame = frame_processing(frame)
	    out.write(frame)
	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	else:
	    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
