import cv2
import datetime

# open default camera
cam = cv2.VideoCapture(1)

# default cam width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# define codec and create writer object
fourcc = cv2.VideoWriter_fourcc(*'H264')
filename = f"data/raw/cap_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mkv"
out = cv2.VideoWriter(filename, fourcc, 20.0, 
                      (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    
    # write framr to output file
    out.write(frame)
    
    # display frame
    cv2.imshow('Camera', frame)
    
    # press q to exit
    if cv2.waitKey(1) == ord('q'):
        break

# release capture and writer object
cam.release()
out.release()
cv2.destroyAllWindows()