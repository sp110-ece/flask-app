from flask import Flask, Response, render_template_string, render_template, jsonify
import cv2
import mediapipe as mp
import math
from model import load_model
import torchvision.transforms as T
import torch
import numpy as np
import time
import standing_quad 
from stretch_sequence import stretch_stream

from app_instance import app
import sitting_hams

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(stretch_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stretch1')
def stretch1():
    return render_template("stretch1.html")

def handler(environ, start_response):
    return app(environ, start_response)


if __name__ == '__main__':
    app.run(debug=True)
