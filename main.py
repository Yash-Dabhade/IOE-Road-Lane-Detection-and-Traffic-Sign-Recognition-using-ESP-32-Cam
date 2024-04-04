import cv2
import numpy as np

# Replace 'IP_ADDRESS' with the IP address of your ESP32
url = 'http://192.168.0.104/cam-hi.jpg'

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

# cap = cv2.VideoCapture("./lanes_clip.mp4");
cap = cv2.VideoCapture(url); #Replace with url for ESP32 Cam Transmission
            

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    # Perform lane detection
    # detect_lane(frame)

    # Perform traffic signal detection
    # detect_traffic_signal(frame)

    # Display the processed frame
    cv2.imshow('ESP32 Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
