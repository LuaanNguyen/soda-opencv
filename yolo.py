from ultralytics import YOLO
import cv2 


model = YOLO("yolov8s.pt")

img = cv2.imread("test.jpg")

results = model(img)[0]

results.save(filename="output.jpg")

plotted_img = results.plot()
cv2.imwrite("output.jpg", plotted_img)

cv2.destroyAllWindows()