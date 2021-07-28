import cv2
import numpy as np
import time
import math

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h, idd = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
           
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 30:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            if(idd == 1):
                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

tracker = EuclideanDistTracker()
net = cv2.dnn.readNet("yolov3.weights", "yolov3_testing.cfg")
classes = ["car"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4")
frame_id = 0
top_count = 0
writer = cv2.VideoWriter('Detected1_result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))

success = cap.read() 
while success:
    # Get frame
    _, frame = cap.read()   
    frame_id += 1
    if frame_id % 1 != 0:
        continue
    
    height, width, channels = frame.shape
    # print(height)
    # print(width)

    # Detecting objects 00392
    blob = cv2.dnn.blobFromImage(frame, 0.00261, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(output_layers)
    class_ids = []
    result = []
    confidences = []
    boxes = []
    liste = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                if frame_id < 20:
                    x1=240
                else:
                    x1=350
                y1=600
                
                if x1 < center_y < y1 and w > 20:
                    idd=1
                else:
                    idd=0
                    
                boxes.append([x, y, w, h, idd])
        
    boxes_ids = tracker.update(boxes)
    # print(boxes_ids)
    boxes_ids_list = []
    new_id_list =[]
   
    for i in range(len(boxes_ids)):
        if boxes_ids[i][4] not in boxes_ids_list:
            boxes_ids_list.append(boxes_ids[i][4])
            new_id_list.append(boxes_ids[i])
    for box_id in new_id_list:
            x, y, w, h, id = box_id
            cv2.putText(frame, str(id), (x+3 , y + h-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 3)
            print(new_id_list)
            # cv2.imshow("FRAME",frame)

    try:
        _,_,_,_,val = max(new_id_list, key=lambda item: item[4])
        if val > top_count:
            top_count = val
    except:
        pass
    cv2.putText(frame, "Abdul Mannan", (905, 90), font, 3, (0, 0, 255), 2)
    cv2.putText(frame, str(top_count), (1800, 90), font, 3, (0, 0, 255), 2)

    writer.write(frame)  
    cv2.imshow("Car Counting", frame)
    
    k = cv2.waitKey(1) & 0xff 
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
