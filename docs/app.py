from flask import Flask, render_template, request
import cv2
import math
import numpy as np

app = Flask(__name__)


MIN_WIDTH_RECT = 80
MIN_HEIGHT_RECT = 80
COUNT_LINE_POSITION = 550
OFFSET = 6 


video_paths = [
    'vid1.mp4',  # Road 1
    'vid2.mp4',  # Road 2
    'vid3.mp4',  # Road 3
    'vid4.mp4',  # Road 4
    'vid5.mp4',  # Road 5
    'vid6.mp4',  # Road 6
]

DURATION_PER_VEHICLE = 1.5  

def process_video(selected_roads):
    vehicle_counts = [0] * len(video_paths)
    traffic_times = [0] * len(video_paths)
    current_cycles = ['RED'] * len(video_paths)

    
    algo = cv2.createBackgroundSubtractorMOG2()

    for i in selected_roads:
        cap = cv2.VideoCapture(video_paths[i])
        if not cap.isOpened():
            continue

        while True:
            ret, frame1 = cap.read()
            if not ret:
                break

            grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey, (5, 5), 0)
            img_sub = algo.apply(blur)
            dilat = cv2.dilate(img_sub, np.ones((5, 5)), iterations=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
            countershape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detect = []
            for c in countershape:
                (x, y, w, h) = cv2.boundingRect(c)
                if (w >= MIN_WIDTH_RECT) and (h >= MIN_HEIGHT_RECT):
                    center = (x + w // 2, y + h // 2)
                    detect.append(center)

            detected_vehicles = 0
            for (x, y) in detect:
                if COUNT_LINE_POSITION - OFFSET < y < COUNT_LINE_POSITION + OFFSET:
                    detected_vehicles += 1

            
            vehicle_counts[i] += detected_vehicles

        cap.release()

    
    max_count_index = vehicle_counts.index(max(vehicle_counts))

    
    for i in range(len(vehicle_counts)):
        current_cycles[i] = 'GREEN' if i == max_count_index else 'RED'
        
        traffic_times[i] = math.ceil(vehicle_counts[i] * DURATION_PER_VEHICLE)

    return vehicle_counts, current_cycles, traffic_times

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_roads = request.form.getlist('roads')
        selected_roads = [int(i) for i in selected_roads]
        vehicle_counts, current_cycles, traffic_times = process_video(selected_roads)
        return render_template('results.html', 
                               vehicle_counts=vehicle_counts, 
                               selected_roads=selected_roads, 
                               current_cycles=current_cycles, 
                               traffic_times=traffic_times)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)