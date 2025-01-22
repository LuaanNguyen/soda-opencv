from ultralytics import YOLO
import cv2


#loading the model
model = YOLO('yolov8s.pt')

img = cv2.imread('test.jpg')
results = model(img)[0]

plotted_img = results.plot()
cv2.imwrite('output.jpg', plotted_img)

for result in results.boxes.data.tolist():
    x1,y1,x2,y2,score,class_id = result
    class_name = results.names[int(class_id)]

    cv2.rectangle(img, (int(x1), (int(y1))) , (int(x2), (int(y2))), (0,0,0), 4)
    cv2.putText(img, f'{class_name} {score: .2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0), 3)
    pass

cv2.imshow('image', img)
cv2.waitKey(0)

if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()