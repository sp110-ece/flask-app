from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import math
from model import load_model
import torchvision.transforms as T
import numpy as np
import time
from app_instance import app
import sitting_hams
import torch
from PIL import Image
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def draw_feedback_bar(frame, title, message, bg_color, text_color, accent_color, text_size, small_size):
    height, width, _ = frame.shape

    bar_width = int(width * 0.8)
    bar_height = 80
    bar_x = (width - bar_width) // 2
    bar_y = 20
    radius = bar_height // 2

    overlay = frame.copy()

    rect_top_left = (bar_x + radius, bar_y)
    rect_bottom_right = (bar_x + bar_width - radius, bar_y + bar_height)
    cv2.rectangle(overlay, rect_top_left, rect_bottom_right, bg_color, -1)

    left_center = (bar_x + radius, bar_y + radius)
    right_center = (bar_x + bar_width - radius, bar_y + radius)
    cv2.circle(overlay, left_center, radius, bg_color, -1)
    cv2.circle(overlay, right_center, radius, bg_color, -1)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, title, (bar_x + 30, bar_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, text_size, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, message, (bar_x + 30, bar_y + 65),
                cv2.FONT_HERSHEY_DUPLEX, small_size, accent_color, 1, cv2.LINE_AA)



transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
def is_side_profile(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    body_scale = math.dist(
        [left_shoulder.x, left_shoulder.y],
        [right_hip.x, right_hip.y]
    )
    z_diff = abs(left_hip.z - right_hip.z) / body_scale

    # Visibility check for frontal view landmarks
    visible = (
        (left_shoulder.visibility > 0.9 and
        left_hip.visibility > 0.9 and
        left_foot.visibility > 0.9) or
        (right_shoulder.visibility > 0.9 and
        right_hip.visibility > 0.9 and
        right_foot.visibility > 0.9)
    )

    if not visible:
        return "Stand with your entire body in frame, with left side towards camera"
    elif right_foot.visibility < 0.9 and left_foot.visibility < 0.9:
        return "Move backwards"
    elif right_foot.z > left_foot.z:
        return "Turn to the right"
    elif z_diff < 1.00:
        return "Turn to the right"
    elif 1.0 <= z_diff < 1.6:
        return "Good"
    elif z_diff >= 1.6:
        return "Turn to the left"
    else:
        print(z_diff)
        return "Stand with your entire body in frame, with left side towards camera"
        

def is_left_profile(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    body_scale = math.dist(
        [right_shoulder.x, right_shoulder.y],
        [left_hip.x, left_hip.y]
    )
    z_diff = abs(right_hip.z - left_hip.z) / body_scale

    # Visibility check for frontal view landmarks
    visible = (
        (left_shoulder.visibility > 0.9 and
        left_hip.visibility > 0.9) or
        (right_shoulder.visibility > 0.9 and
        right_hip.visibility > 0.9)
    )

    if not visible:
        return "Stand with your entire body in frame, with right side towards camera"
    elif right_foot.visibility < 0.9 and left_foot.visibility < 0.9:
        return "Move backwards"
    elif right_foot.z < left_foot.z:
        return "Turn to the left"
    elif z_diff < 1.00:
        return "Turn to the left"
    elif 1.0 <= z_diff < 1.6:
        return "Good"
    elif z_diff >= 1.6:
        return "Turn to the right"
    else:
        print(z_diff)
        return "Stand with your entire body in frame, with right side towards camera"


def cnn_calculate(input_frame, model, frame):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w, _ = frame.shape
    image = input_frame.to(device)
    outputs = model(image)
    outputs = outputs.detach().cpu().numpy().reshape(-1, 2) * 224
    outputs = outputs * [w / 224, h / 224]
    # for (x, y) in outputs:
    #     cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), 1)
    # output = outputs[4]
    # cv2.circle(frame, (int(output[0]), int(output[1])), 3, (255, 0, 0), 1)

    return outputs


def calculate_angle(a, b, c):
    a_arr = np.array([a.x, a.y, a.z])
    b_arr= np.array([b.x, b.y, b.z])
    c_arr = np.array([c.x, c.y, c.z])

    x = a_arr - b_arr 
    y = c_arr - b_arr

    num = np.dot(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y)

    return np.degrees(np.arccos(num/den))


def stretch_check(landmarks, model_output):
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    standing_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    active_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    

    knee_angle = calculate_angle(hip, knee, active_ankle)
    hip_angle = calculate_angle(knee, hip, shoulder)
    ankle_angle = calculate_angle(foot, standing_ankle, knee)


    if (shoulder.visibility < 0.7 and hip.visibility < 0.7 and standing_ankle.visibility < 0.7 and active_ankle.visibility < 0.7 and knee.visibility < 0.7 and foot.visibility < 0.7):
        active_ankle = model_output[0]
        hip = model_output[1]
        knee = model_output[2]
        shoulder = model_output[3]
        standing_ankle = model_output[4]

        knee_angle = calculate_angle(hip, knee, active_ankle)
        hip_angle = calculate_angle(knee, hip, shoulder)
        ankle_angle = 50


    
    if (knee_angle <= 80 and hip_angle >=125 and ankle_angle > 40 and ankle_angle < 60):
        return "Good! Hold that position"
    elif (hip_angle < 125):
        # print(hip_angle)
        return "Try not to bend your hip"
    elif (knee_angle > 80):
        return "Bring in your ankle more, to feel a better stretch"
    elif hip.x < knee.x:
        return "Keep your knee in line with your hip"
    elif (ankle_angle <= 40 or ankle_angle >= 60):
        # print(ankle_angle)
        return "Try not to sway on your foot"
    else:
        return "out of frame"
    


def left_stretch_check(landmarks):
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    standing_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    active_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]


    knee_angle = calculate_angle(hip, knee, active_ankle)
    hip_angle = calculate_angle(knee, hip, shoulder)
    ankle_angle = calculate_angle(foot, standing_ankle, knee)
    if (knee_angle <= 80 and hip_angle >=125 and ankle_angle > 40 and ankle_angle < 60):
        return "Good! Hold that position"
    elif (hip_angle < 125):
        # print(hip_angle)
        return "Try not to bend your hip"
    elif (knee_angle > 80):
        return "Bring in your ankle more, to feel a better stretch"
    elif hip.x < knee.x:
        return "Keep your knee in line with your hip"
    elif (ankle_angle <= 40 or ankle_angle >= 60):
        # print(ankle_angle)
        return "Try not to sway on your foot"
    else:
        return "out of frame"

    
    
done = False
def webcam_stream(): 
    global done
    cap = cv2.VideoCapture(0)
    pose = mp.solutions.pose.Pose(static_image_mode=False)
    last_detected_label = None
    last_detected_comment = None
    stable_comment_time = None
    last_stable_time = time.time()
    good_start_time = None
    timer_start = None
    message = "Stand with entire body in frame"
    buffer_time = 0.5
    doing_side_profile_check = True
    doing_left_profile_check = False
    timer = 30
    loop_timer = None
    left_stretch = True
    bg_color = (40, 40, 40)             # Dark gray background
    text_primary = (255, 255, 255)      # White text
    accent_color = (251, 219, 72)       # Sky blue
    wait_state = False
    broadast = None
    right_stretch =False
    model = load_model()
    while True: 
        done = True
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
        results = pose.process(rgb_frame)
        debug_frame = frame.copy()
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # mp_drawing.draw_landmarks(
            #     debug_frame, 
            #     results.pose_landmarks, 
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(180, 200, 190), thickness=1, circle_radius=1),
            #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 140, 120), thickness=2)
            # )

            if (left_stretch == True):

                if doing_side_profile_check:
                    detected_label = is_side_profile(landmarks)    
                    if detected_label != last_detected_label:
                        last_detected_label = detected_label
                        last_stable_time = time.time()
                    else:
                        if time.time() - last_stable_time >= buffer_time:
                            message = detected_label
                    

                    draw_feedback_bar(debug_frame, "Tracking...", message, bg_color, text_primary, accent_color, 1, 0.7)
                    if message == "Good":
                        if good_start_time is None:
                            good_start_time = time.time()
                        elif (time.time() - good_start_time > 2):
                            # cv2.putText(debug_frame, "Let's start a simple left quad stretch", (40, 150),
                            #     cv2.FONT_HERSHEY_SIMPLEX, 1.0, accent_color, 2, cv2.LINE_AA)
                            doing_side_profile_check = False
                    else:
                        good_start_time = None
                    
                    if not results.pose_landmarks:
                        draw_feedback_bar(debug_frame, "Tracking...", "Stand with entire body in frame", bg_color, text_primary, accent_color, 1, 0.7)
                else: 
                    
                    img = Image.fromarray(rgb_frame)
                    tensor = transform(img)
                    tensor = tensor.unsqueeze(0)
                    cnn_output = cnn_calculate(tensor, model, debug_frame)
                    # print(cnn_output)
                    detected_comment = stretch_check(landmarks, cnn_output)
                    
                    if detected_comment != last_detected_comment:
                        last_detected_comment = detected_comment
                        stable_comment_time = time.time()
                    else:
                        if time.time() - stable_comment_time >= 1:
                            message = detected_comment
                    draw_feedback_bar(debug_frame, "Stretch Feedback", message, bg_color, text_primary, accent_color, 1, 0.7)
                    
                    if message == "Good! Hold that position":
                        if timer_start is None:
                            timer_start = time.time()
                            loop_timer = time.time()
                        elif (time.time() - timer_start > 1):
                           
                            draw_feedback_bar(debug_frame, "Let's start our left stretch", None, bg_color, text_primary, accent_color, 1, 0.7)
                            if loop_timer is not None and time.time() - loop_timer >= 1:
                                timer -= 1
                                loop_timer = time.time()
                            if timer > 0:
                                draw_feedback_bar(debug_frame, "", str(timer), bg_color, text_primary, accent_color, 1, 0.7)
                            else:
                                last_detected_label = None
                                last_detected_comment = None
                                stable_comment_time = None
                                last_stable_time = time.time()
                                good_start_time = None
                                timer_start = None
                                buffer_time = 0.5
                                doing_side_profile_check = True
                                timer = 30
                                loop_timer = None
                                doing_left_profile_check=True
                                message = "flip"
                                left_stretch=False
                                wait_state = True
                                cv2.putText(debug_frame, "Left stretch complete, now let's stretch the right", (40, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, accent_color, 2, cv2.LINE_AA)

                    else:
                        timer_start = None
            if wait_state == True:
                draw_feedback_bar(debug_frame, "Now we're going to repeat on the other side", "Flip around", bg_color, text_primary, accent_color, 1, 0.7)
                if (broadast == None):
                    broadast = time.time()
                elif (time.time() - broadast > 3):
                    wait_state = False
                    right_stretch = True
            if (right_stretch == True):
                if doing_left_profile_check:
                    detected_label = is_left_profile(landmarks)    
                    
                    
                    if detected_label != last_detected_label:
                        last_detected_label = detected_label
                        last_stable_time = time.time()
                    else:
                        if time.time() - last_stable_time >= buffer_time:
                            message = detected_label
                    # Add elegant text
                    draw_feedback_bar(debug_frame, "Tracking...", message, bg_color, text_primary, accent_color, 1, 0.7)
                    if message == "Good":
                        if good_start_time is None:
                            good_start_time = time.time()
                        elif (time.time() - good_start_time > 2):
                            doing_left_profile_check = False
                            # draw_feedback_bar(debug_frame, "Let's start our right stretch", "", bg_color, text_primary, accent_color)
                            
                    else:
                        good_start_time = None
                    
                    if not results.pose_landmarks:
                        draw_feedback_bar(debug_frame, "Tracking...", "Stand with entire body in frame", bg_color, text_primary, accent_color, 1, 0.7)
                else: 
                    detected_comment = left_stretch_check(landmarks)
                    
                    if detected_comment != last_detected_comment:
                        last_detected_comment = detected_comment
                        stable_comment_time = time.time()
                    else:
                        if time.time() - stable_comment_time >= 1:
                            message = detected_comment
                    draw_feedback_bar(debug_frame, "Stretch Feedback", message, bg_color, text_primary, accent_color, 1, 0.7)
                    
                    if message == "Good! Hold that position":
                        if timer_start is None:
                            timer_start = time.time()
                            loop_timer = time.time()
                        elif (time.time() - timer_start > 1):
                            draw_feedback_bar(debug_frame, "Let's start our right stretch", "", bg_color, text_primary, accent_color, 1, 0.7)
                            
                            if loop_timer is not None and time.time() - loop_timer >= 1:
                                timer -= 1
                                loop_timer = time.time()
                            if timer > 0:
                                draw_feedback_bar(debug_frame, "", str(timer), bg_color, text_primary, accent_color, 1, 0.7)
                            else:
                                draw_feedback_bar(debug_frame, "Stretch Complete", "Now let's do a sitting hamstring stretch", bg_color, text_primary, accent_color, 1, 0.7)
                                return
                                
                    
                    
                    else:
                        timer_start = None

            
                

        
        
                
        
        ret, buffer = cv2.imencode('.jpeg', debug_frame)
        frame_arr = buffer.tobytes()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_arr + b'\r\n') 



if (done == True):
    sitting_hams.ham_stream()

# @app.route('/')
# def index():
#     return render_template_string("""
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <title>Posture Feedback</title>
#         <style>
#             body {
#                 background-color: #f5f0e6;
#                 display: flex;
#                 justify-content: center;
#                 align-items: center;
#                 height: 100vh;
#                 margin: 0;
#                 font-family: 'Helvetica Neue', sans-serif;
#             }
#             .camera-frame {
#                 border-radius: 20px;
#                 overflow: hidden;
#                 box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
#                 border: 2px solid #ddd;
#             }
#         </style>
#     </head>
#     <body>
#         <div class="camera-frame">
#             <img src="{{('/video_feed') }}" width="800" />
#         </div>
#     </body>
#     </html>
#     """)
# @app.route('/video_feed')
# def video_feed():
#     return Response(webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    


if __name__ == '__main__':
    app.run(debug=True)