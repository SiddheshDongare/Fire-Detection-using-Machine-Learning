import math
import cv2
import cvzone
from ultralytics import YOLO

# Set input_type to 'video' or 'image'
input_type = 'video'  # Change this to 'video' for video input
video_file = 'demo2.mp4'
image_file = 'demo4.jpg'

model = YOLO('best.pt')
classnames = ['fire']

if input_type == 'video':
    # Using live feed
    cap = cv2.VideoCapture(0)

    # Using static video file
    # cap = cv2.VideoCapture(video_file)
else:
    frame = cv2.imread(image_file)
    if frame is not None:
        frame = cv2.resize(frame, (640, 480))
    else:
        print("Error: Image not read correctly. Check the image file path.")
        exit()

while True:
    if input_type == 'video':
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

    result = model(frame, stream=True)

    # Getting box, confidence, and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if 0 <= Class < len(classnames):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

    cv2.imshow('frame', frame)

    if input_type == 'image':
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if input_type == 'video':
    cap.release()
cv2.destroyAllWindows()
