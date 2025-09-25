import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # 'yolov5n' for Nano model

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform detection
    results = model(frame)

    # Render results
    results.render()  # updates results.ims with boxes and labels

    # Display the resulting frame
    cv2.imshow('Webcam', results.ims[0])  # Use 'ims' instead of 'imgs'

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()