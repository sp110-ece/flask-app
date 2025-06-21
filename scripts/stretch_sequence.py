import cv2
from flask import Flask, Response, render_template
from standing_quad import webcam_stream
from sitting_hams import ham_stream
import numpy as np
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "scripts", "bg.jpg")
def stretch_stream():

    delay_frame = quad_transition_frame()
    duration = 5
    fps = 10
    for _ in range(duration * fps):
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + delay_frame + b'\r\n') 
        time.sleep(1 / fps)

    for frame in webcam_stream():
        yield frame
    
    ham_frame = ham_transition_frame()
    duration = 5
    fps = 10
    for _ in range(duration * fps):
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + ham_frame + b'\r\n') 
        time.sleep(1 / fps)
    
    for frame in ham_stream():
        yield frame
    
    end_frame = end_transition_frame()
    
    yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + end_frame + b'\r\n') 



def quad_transition_frame():
    frame = cv2.imread("./scripts/bg.jpg")
    if frame is None:
        raise FileNotFoundError("not found")
    frame = cv2.resize(frame, (1000, 1000))
    overlay = 255 * np.ones_like(frame, dtype = np.uint8)

    alpha = 0.9

    faded = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    cv2.putText(faded, "First: Standing Quad Stretch", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.putText(faded, "1. Find you balance on one leg", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "2. Bend the knee of your non-standing leg as far", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "   as possible", (50, 385), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "3. Pull your leg in by foot or ankle", (50, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "4. Hold for 30 seconds", (50, 485), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    success, buffer = cv2.imencode('.jpg', faded)
    if not success:
        raise ValueError("failure")
    return buffer.tobytes()

def ham_transition_frame():
    frame = cv2.imread("./scripts/bg.jpg")
    if frame is None:
        raise FileNotFoundError("not found")
    frame = cv2.resize(frame, (1000, 1000))
    overlay = 255 * np.ones_like(frame, dtype = np.uint8)

    alpha = 0.9

    faded = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    cv2.putText(faded, "Next: Sitting Hamstring Stretch", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 0), 2)
    cv2.putText(faded, "1. Sit down with your right leg extended in front of you", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "2. Keep your back straight and knee extended", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "3. Lean forward and touch your toes", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(faded, "4. Hold for 30 seconds", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    success, buffer = cv2.imencode('.jpg', faded)
    if not success:
        raise ValueError("failure")
    return buffer.tobytes()

def end_transition_frame():
    frame = cv2.imread("./scripts/bg.jpg")
    if frame is None:
        raise FileNotFoundError("not found")
    frame = cv2.resize(frame, (1000, 1000))
    overlay = 255 * np.ones_like(frame, dtype = np.uint8)

    alpha = 0.9

    faded = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    cv2.putText(faded, "All done", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    success, buffer = cv2.imencode('.jpg', faded)
    if not success:
        raise ValueError("failure")
    return buffer.tobytes()