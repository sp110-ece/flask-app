import cv2
from flask import Flask, Response
from app_instance import app
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
import math
import numpy as np
import time


def calculate_angle(a, b, c):
    a_arr = np.array([a.x, a.y, a.z])
    b_arr= np.array([b.x, b.y, b.z])
    c_arr = np.array([c.x, c.y, c.z])

    x = a_arr - b_arr 
    y = c_arr - b_arr

    num = np.dot(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y)

    return np.degrees(np.arccos(num/den))





def draw_feedback_bar(frame, title, message, bg_color, text_color, accent_color):
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
                cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, message, (bar_x + 30, bar_y + 65),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, accent_color, 1, cv2.LINE_AA)
    


def is_left(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    
    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

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
        return "sit so that your entire body is in camera view"
    elif hip_angle > 120:
        return "You need to be seated"
    elif (right_foot.z < left_foot.z):
        return "Good"
    else:
        return "flip around"
    




def is_right(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

    hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

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
        return "sit so that your entire body is in camera view"
    elif (hip_angle > 120):
        return "You need to be seated"
    elif (right_foot.z > left_foot.z):
        return "Good"
    else:
        return "flip around"



def left_stretch_check(landmarks):
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    active_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    toes = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    

    knee_angle = calculate_angle(hip, knee, active_ankle)
    hip_angle = calculate_angle(shoulder,  hip, knee)
    foot_angle = calculate_angle(active_ankle, heel, toes)
    # print (foot_angle)
    if (knee_angle > 120 and hip_angle < 65):
        return "good"
    elif (knee_angle <= 115):
        return "try and straigthen your leg"
    elif hip_angle >= 65:
        return "bend towards your leg more"
    return "test"


def right_stretch_check(landmarks):
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    active_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    toes = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    

    knee_angle = calculate_angle(hip, knee, active_ankle)
    hip_angle = calculate_angle(shoulder,  hip, knee)
    foot_angle = calculate_angle(active_ankle, heel, toes)
    # print (foot_angle)
    if (knee_angle > 125 and hip_angle < 70):
        return "good"
    elif (knee_angle <= 125):
        return "try and straigthen your leg"
    elif hip_angle >= 70:
        return "bend towards your leg more"
    return "test"





def ham_stream(): 
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
    while True: 
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
                    detected_label = is_left(landmarks)    
                    if detected_label != last_detected_label:
                        last_detected_label = detected_label
                        last_stable_time = time.time()
                    else:
                        if time.time() - last_stable_time >= buffer_time:
                            message = detected_label
                    

                    draw_feedback_bar(debug_frame, "Hamstring: Tracking...", message, bg_color, text_primary, accent_color)
                    if message == "Good":
                        if good_start_time is None:
                            good_start_time = time.time()
                        elif (time.time() - good_start_time > 2):
                            doing_side_profile_check = False
                    else:
                        good_start_time = None
                    
                    if not results.pose_landmarks:
                        draw_feedback_bar(debug_frame, "Tracking...", "Stand with entire body in frame", bg_color, text_primary, accent_color)
                else: 
                    detected_comment = left_stretch_check(landmarks)
                    
                    if detected_comment != last_detected_comment:
                        last_detected_comment = detected_comment
                        stable_comment_time = time.time()
                    else:
                        if time.time() - stable_comment_time >= 1:
                            message = detected_comment
                    draw_feedback_bar(debug_frame, "Stretch Feedback", message, bg_color, text_primary, accent_color)
                    
                    if message == "good":
                        if timer_start is None:
                            timer_start = time.time()
                            loop_timer = time.time()
                        elif (time.time() - timer_start > 1):
                           
                            draw_feedback_bar(debug_frame, "Let's start our left stretch", None, bg_color, text_primary, accent_color)
                            if loop_timer is not None and time.time() - loop_timer >= 1:
                                timer -= 1
                                loop_timer = time.time()
                            if timer > 0:
                                draw_feedback_bar(debug_frame, "", str(timer), bg_color, text_primary, accent_color)
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
                draw_feedback_bar(debug_frame, "Now we're going to repeat on the other side", "Flip around", bg_color, text_primary, accent_color)
                if (broadast == None):
                    broadast = time.time()
                elif (time.time() - broadast > 3):
                    wait_state = False
                    right_stretch = True
            if (right_stretch == True):
                if doing_left_profile_check:
                    detected_label = is_right(landmarks)    
                    
                    
                    if detected_label != last_detected_label:
                        last_detected_label = detected_label
                        last_stable_time = time.time()
                    else:
                        if time.time() - last_stable_time >= buffer_time:
                            message = detected_label
                    # Add elegant text
                    draw_feedback_bar(debug_frame, "Tracking...", message, bg_color, text_primary, accent_color)
                    if message == "Good":
                        if good_start_time is None:
                            good_start_time = time.time()
                        elif (time.time() - good_start_time > 2):
                            doing_left_profile_check = False
                            # draw_feedback_bar(debug_frame, "Let's start our right stretch", "", bg_color, text_primary, accent_color)
                            
                    else:
                        good_start_time = None
                    
                    if not results.pose_landmarks:
                        draw_feedback_bar(debug_frame, "Tracking...", "Stand with entire body in frame", bg_color, text_primary, accent_color)
                else: 
                    detected_comment = right_stretch_check(landmarks)
                    
                    if detected_comment != last_detected_comment:
                        last_detected_comment = detected_comment
                        stable_comment_time = time.time()
                    else:
                        if time.time() - stable_comment_time >= 1:
                            message = detected_comment
                    draw_feedback_bar(debug_frame, "Stretch Feedback", message, bg_color, text_primary, accent_color)
                    
                    if message == "good":
                        if timer_start is None:
                            timer_start = time.time()
                            loop_timer = time.time()
                        elif (time.time() - timer_start > 1):
                            draw_feedback_bar(debug_frame, "Let's start our right stretch", "", bg_color, text_primary, accent_color)
                            
                            if loop_timer is not None and time.time() - loop_timer >= 1:
                                timer -= 1
                                loop_timer = time.time()
                            if timer > 0:
                                draw_feedback_bar(debug_frame, "", str(timer), bg_color, text_primary, accent_color)
                            else:
                                return
                                
                    
                    
                    else:
                        timer_start = None

            
                

        
        
                
        
        ret, buffer = cv2.imencode('.jpeg', debug_frame)
        frame_arr = buffer.tobytes()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_arr + b'\r\n') 


@app.route('/sitting_hams')
def sitting_hams():
    return Response(ham_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    


if __name__ == '__main__':
    app.run(debug=True)