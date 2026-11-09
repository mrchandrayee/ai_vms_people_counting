import streamlit as st
import cv2, tempfile, os
import pandas as pd
from .engine import PeopleCounter, draw_overlay

def app():
    st.title("AI VMS — People Counting")
    st.caption("Person detection • ByteTrack IDs • virtual line IN/OUT counting")
    with st.sidebar:
        st.header("Model settings")
        weights=st.text_input("YOLO weights", "yolov8n.pt")
        conf=st.slider("Confidence",0.10,0.90,0.35,0.05)
        line=st.slider("Counting line position",0.10,0.90,0.65,0.05)
        stride=st.number_input("Process every Nth frame",1,10,1)
    uploaded=st.file_uploader("Upload CCTV recording",type=["mp4","avi","mov","mkv"])
    if uploaded is None:
        st.info("Upload a CCTV video to analyze. For live multi-camera RTSP streams, use run_cameras.py.")
        return
    if st.button("Analyze video",type="primary"):
        suffix=os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer()); video_path=tmp.name
        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Unable to open uploaded video."); os.unlink(video_path); return
        try:
            engine=PeopleCounter(weights,conf,line)
            preview=st.empty(); progress=st.progress(0)
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_idx=0; rows=[]; last_frame=None
            while True:
                ok,frame=cap.read()
                if not ok: break
                frame_idx+=1
                if frame_idx % int(stride): continue
                detections=engine.process(frame)
                for d in detections:
                    if d["event"]:
                        rows.append({"frame":frame_idx,"track_id":d["track_id"],
                                     "direction":d["event"],"confidence":round(d["confidence"],3)})
                annotated=draw_overlay(frame.copy(),detections,engine.counter,line)
                preview.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),channels="RGB",
                              use_container_width=True)
                last_frame=annotated
                progress.progress(min(frame_idx/total,1.0))
            cap.release()
            c1,c2,c3=st.columns(3)
            c1.metric("IN",engine.counter.in_count)
            c2.metric("OUT",engine.counter.out_count)
            c3.metric("Net occupancy",engine.counter.total)
            df=pd.DataFrame(rows,columns=["frame","track_id","direction","confidence"])
            st.subheader("Crossing events")
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button("Download events CSV",df.to_csv(index=False),
                               "people_count_events.csv","text/csv")
            if last_frame is not None:
                st.image(cv2.cvtColor(last_frame,cv2.COLOR_BGR2RGB),
                         caption="Last processed frame",use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
            st.exception(exc)
        finally:
            cap.release()
            if os.path.exists(video_path): os.unlink(video_path)
