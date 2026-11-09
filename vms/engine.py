import cv2
from ultralytics import YOLO
from .counting import LineCounter

class PeopleCounter:
    def __init__(self, weights="yolov8n.pt", confidence=0.35, line_position=0.65):
        self.model=YOLO(weights)
        self.confidence=confidence
        self.counter=LineCounter(line_position)

    def process(self, frame):
        result=self.model.track(frame, persist=True, classes=[0],
                                conf=self.confidence, tracker="bytetrack.yaml",
                                verbose=False)[0]
        detections=[]
        if result.boxes is not None and result.boxes.id is not None:
            boxes=result.boxes
            for box,tid,score in zip(boxes.xyxy.cpu().numpy(),
                                     boxes.id.int().cpu().tolist(),
                                     boxes.conf.cpu().tolist()):
                x1,y1,x2,y2=map(int,box)
                cx=(x1+x2)//2; cy=(y1+y2)//2
                event=self.counter.update(tid,cy,frame.shape[0])
                detections.append({"track_id":int(tid),"bbox":(x1,y1,x2,y2),
                                   "confidence":float(score),"event":event})
        return detections

def draw_overlay(frame, detections, counter, line_position=0.65):
    h,w=frame.shape[:2]; y=int(h*line_position)
    cv2.line(frame,(0,y),(w,y),(0,220,255),2)
    cv2.putText(frame,"COUNTING LINE",(10,max(22,y-10)),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),2)
    for d in detections:
        x1,y1,x2,y2=d["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(40,220,80),2)
        label=f'Person ID {d["track_id"]}'
        if d["event"]: label+=f' | {d["event"]}'
        cv2.putText(frame,label,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,
                    .5,(40,220,80),2)
    text=f"IN: {counter.in_count}   OUT: {counter.out_count}   NET: {counter.total}"
    cv2.rectangle(frame,(0,0),(w,38),(20,20,20),-1)
    cv2.putText(frame,text,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2)
    return frame
