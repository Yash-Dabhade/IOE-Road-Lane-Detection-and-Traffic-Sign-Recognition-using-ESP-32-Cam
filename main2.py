import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Replace 'IP_ADDRESS' with the IP address of your ESP32
url = 'http://<ESP32_IP_ADDRESS>/stream'

def detect_lane(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Define region of interest (ROI)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width // 2, height // 2), (width, height)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

def detect_traffic_signal(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Create a mask for red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours around detected traffic signals
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

def detect_pedestrians(frame):
    # Load pedestrian detection model
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # Detect pedestrians in the frame
    pedestrians, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # Draw rectangles around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def detect_traffic_signs(frame):
    # Load pre-trained MobileNet SSD model for traffic sign detection
    net = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_traffic_sign.pb', 'ssd_mobilenet_v2_coco.pbtxt')
    # Define classes for traffic signs
    classes = ["background", "speed limit", "yield", "stop", "no parking", "no entry"]
    # Convert frame to blob format
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
    # Set input to the network
    net.setInput(blob)
    # Perform forward pass and get output
    detections = net.forward()
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Threshold for confidence
            class_id = int(detections[0, 0, i, 1])
            if class_id in [1, 2, 3, 4, 5]:  # Traffic sign classes
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                # Draw bounding box and class label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = "{}: {:.2f}%".format(classes[class_id], confidence * 100)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# IMPORT THE TRANNIED MODEL
model=load_model("model.h5")  ## rb = READ BYTE
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'
 


# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture("./lanes_clip.mp4");
                

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Perform lane detection
    detect_lane(frame)

    # Perform traffic signal detection
    # detect_traffic_signal(frame)

    # Perform pedestrian detection
    # detect_pedestrians(frame)

    # Perform traffic sign recognition
    # detect_traffic_signs(frame)

    #Perform Traffic Sign Detection with custom model
    predictions = model.predict(frame)
    classIndex =np.argmax(predictions)
    preds = getClassName(classIndex)

    # Display the processed frame
    cv2.imshow('ESP32 Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
