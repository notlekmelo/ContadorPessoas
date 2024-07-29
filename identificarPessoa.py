import cv2
from ultralytics import YOLO
from tracker import Tracker

webcam = cv2.VideoCapture(0)
modelo = YOLO('yolov8n.pt')
contador = 0
tracked_ids = set()
tracker = Tracker()


if webcam.isOpened():
    validacao, frame = webcam.read()
    while validacao:
        validacao, frame = webcam.read()
        results = modelo(source=frame, classes=0, verbose=False,conf=0.65)

        detections = []
        x,y,w,h = 0,0,0,0
        for result in results:
            for box, cls, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x,y,w,h = box
                x,y,w,h = int(x),int(y),int(w),int(h)
                detections.append([x, y, w, h, score])

        tracker.update(frame, detections)

        for output in tracker.tracks:
            bbox = output.bbox
            x, y, w, h = bbox
            track_id = output.track_id
            if track_id not in tracked_ids:
                tracked_ids.add(track_id)
                contador += 1

            x,y,w,h = int(x),int(y),int(w),int(h)       
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 4)
              
        cv2.imshow("Video da WebCam", frame)
        key = cv2.waitKey(5)
        if key == 27:
            break
        if cv2.getWindowProperty("Video da WebCam", cv2.WND_PROP_VISIBLE) < 1:
            break

webcam.release()
cv2.destroyAllWindows()
print(contador)