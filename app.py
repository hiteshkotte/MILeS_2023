from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import threading
import time
import torch
from torchvision import transforms
import cv2
import numpy as np
import shutil
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from moviepy.editor import ImageSequenceClip
from bicep import run_bicep
from lunges import run_lunges
from pushup import run_pushup
from shoulder_lateral_raise import run_shoulder_lateral_raise
from squats import run_squats

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static\\uploads'
app.config['OUTPUT_FOLDER'] = 'static'
app.config['VIDEO_FOLDER'] = 'static'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        print("GPU:", torch.cuda.get_device_name(device))
        model.half().to(device)
    else:
        print("No GPU available")
    return model

model = load_model()

def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0]  # shape: (567, 960, 3)

    # Apply transforms
    image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    else:
        image = image.to(device)
    # Turn image into batch
    image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image

def draw_keypoints(output, image):
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence Threshold
                                     0.65,  # IoU Threshold
                                     nc=model.yaml['nc'],  # Number of Classes
                                     nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg


def process_video(video_file, exercise, webcam, draw_skeleton, recommendation):
    global processing_complete
    # Define the output path for the processed video
    if video_file is not None:
        fname, fext = os.path.splitext(os.path.basename(video_file))
    else:
        fname, fext = f"{exercise}", ".mp4"
    output_filename = f"output_{fname}_conv{fext}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    print("def pose estimation", video_file)
    
    if exercise == 'bicep':
        if webcam == 'true':
            run_bicep(source='0', drawskeleton=draw_skeleton, recommendation = recommendation)
        else:
            run_bicep(source= video_file, drawskeleton=draw_skeleton, recommendation = recommendation) 
    elif exercise == 'lunges':
        if webcam == 'true':
            run_lunges(source='0', drawskeleton=draw_skeleton, recommendation = recommendation)
        else:
            run_lunges(source= video_file, drawskeleton=draw_skeleton, recommendation = recommendation)
    elif exercise == 'pushup':
        if webcam == 'true':
            run_pushup(source='0', drawskeleton=draw_skeleton, recommendation = recommendation)
        else:
            run_pushup(source= video_file, drawskeleton=draw_skeleton, recommendation = recommendation)
    elif exercise == 'shoulder_lateral_raise':
        if webcam == 'true':
            run_shoulder_lateral_raise(source='0', drawskeleton=draw_skeleton, recommendation = recommendation)
        else:
            run_shoulder_lateral_raise(source= video_file, drawskeleton=draw_skeleton, recommendation = recommendation)
    elif exercise == 'squats':
        if webcam == 'true':
            run_squats(source='0', drawskeleton=draw_skeleton, recommendation = recommendation)
        else:
            run_squats(source= video_file, drawskeleton=draw_skeleton, recommendation = recommendation)
        
    
    processing_complete = True
    # Return the output path of the processed video
    return output_path



def run_flask_server():
    app.run()

# Set the initial value of the processing_complete flag
processing_complete = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    #print(request.form())
    exercise = request.form['exercise']
    
    
    webcam = request.form['webcamstream']
    if request.form['drawSkeleton'] == 'yes':
        draw_skeleton = True
    else:
        draw_skeleton = False

    if request.form['recommend'] == 'yes':
        recommendation = True
    else:
        recommendation = False
    
    if webcam == 'false':
        file = request.files['video']

        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            output_path = process_video(video_path, exercise, webcam, draw_skeleton, recommendation)

            print(output_path)
            output_filename = os.path.basename(output_path)
            output_url = f"/static/uploads/{output_filename}"

            return jsonify({'processed': True, 'video_url': output_url})
    else: 
        output_path = process_video(None, exercise, webcam, draw_skeleton, recommendation)
        output_filename = os.path.basename(output_path)
        output_url = f"/static/uploads/{output_filename}"

        return jsonify({'processed': True, 'video_url': output_url})

    return jsonify({'processed': False})

@app.route('/check_processing_status')
def check_processing_status():
    global processing_complete
    global output_filename

    if processing_complete:
        video_url = f"/static/uploads/{output_filename}"
        return jsonify({'status': 'complete', 'video_url': video_url})

    return jsonify({'status': 'incomplete'})

@app.route('/static/uploads/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(video_path)
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    # Run the Flask server on a separate thread
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.start()
    #app.run()
