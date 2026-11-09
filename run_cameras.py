import argparse, json, cv2
from vms.engine import PeopleCounter, draw_overlay

def main():
    ap=argparse.ArgumentParser(description="Multi-camera AI VMS people counter")
    ap.add_argument("--config",default="cameras.json")
    ap.add_argument("--weights",default="yolov8n.pt")
    ap.add_argument("--confidence",type=float,default=.35)
    args=ap.parse_args()
    with open(args.config,encoding="utf-8") as f: config=json.load(f)
    if not config: raise SystemExit("No cameras configured.")
    cameras=[]
    for name,item in config.items():
        source=item["source"]
        if isinstance(source,str) and source.isdigit(): source=int(source)
        cap=cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[WARN] {name}: cannot open {source}"); continue
        line=float(item.get("line_position",.65))
        engine=PeopleCounter(args.weights,args.confidence,line)
        cameras.append((name,cap,engine,line))
    if not cameras: raise SystemExit("No camera streams could be opened.")
    try:
        while True:
            tiles=[]
            for name,cap,engine,line in cameras:
                ok,frame=cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    ok,frame=cap.read()
                if not ok: continue
                detections=engine.process(frame)
                view=draw_overlay(frame,detections,engine.counter,line)
                cv2.putText(view,name,(10,view.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,
                            .7,(255,255,0),2)
                tiles.append((name,view))
            if not tiles: break
            # Resize to common tile size and make a 2-column monitoring wall.
            tile_w,tile_h=640,360
            resized=[]
            for name,img in tiles:
                img=cv2.resize(img,(tile_w,tile_h))
                resized.append(img)
            blank=__import__("numpy").zeros_like(resized[0])
            if len(resized)%2: resized.append(blank)
            rows=[cv2.hconcat(resized[i:i+2]) for i in range(0,len(resized),2)]
            wall=cv2.vconcat(rows)
            cv2.imshow("AI VMS - Multi Camera People Counting",wall)
            if cv2.waitKey(1)&0xFF==ord("q"): break
    finally:
        for _,cap,_,_ in cameras: cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__": main()
