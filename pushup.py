import subprocess
import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
from trainer import findAngle
from PIL import ImageFont, ImageDraw, Image


@torch.no_grad()
def run_pushup(poseweights='yolov7-w6-pose.pt', source='static/uploads/bicep.mp4', device='cpu', curltracker=True, drawskeleton=True, recommendation = False):

    path = source
    if path.isnumeric():
        ext = path     
    else:
        ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print("GPU not available so running on CPU")
        device = select_device(device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)
        webcam = False

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        fw, fh = int(cap.get(3)), int(cap.get(4))
        if ext.isnumeric():
            webcam = True
            fw, fh = 1280, 768
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "D:\\yolov7\\static\\uploads\\output_pushup" if path.isnumeric(
        ) else "D:\\yolov7\\static\\uploads\\output_pushup"
        out = cv2.VideoWriter(f"{out_video_name}.mp4", cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (fw, fh))

        frame_count, total_fps = 0, 0
        bcount = 0
        direction = 0
        max_percentage = 0
        feedback = ""

        fontpath = "./sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)

        font1 = ImageFont.truetype(fontpath, 170)
        font2 = ImageFont.truetype(fontpath, 50)

        print("check check", out_video_name)
        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                # preprocess image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                if webcam:
                    image = cv2.resize(
                        image, (fw, fh), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (fw),
                                  stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(
                    output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                print("before curltracker")
                
                if curltracker:
                    for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        # Right arm =(5,7,9), left arm = (6,8,10)
                        # set draw=True to draw the arm keypoints.
                        angle = findAngle(img, kpts, 5, 7, 9, draw=drawskeleton)
                        percentage = np.interp(angle, (210, 280), (0, 100))
                        bar = np.interp(angle, (220, 280), (fh-100, 100)) # hypers: bar = np.interp(angle, (20, 150), (200, fh-100))

                        color = (254, 118, 136)
                        
                        max_percentage = max(percentage, max_percentage)

                        if percentage >= 60: # hypers: if percentage == 100: 
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                                
                        if percentage == 0:
                            if direction == 1:
                                bcount += 0.5 
                                direction = 0
                                if max_percentage <= 75:
                                    feedback = "Go down, Engage your Lats" if recommendation else ""
                                elif max_percentage <= 90:
                                    feedback = "Almost There, Lock your lats" if recommendation else ""
                                else:
                                    feedback = "Great work! Keep going" if recommendation else ""

                                max_percentage = 0

                        # draw Bar and counter
                        cv2.line(img, (100, 200), (100, fh-100),
                                 (255, 255, 255), 30)
                        cv2.line(img, (100, int(bar)),
                                 (100, fh-100), color, 30)

                        if (int(percentage) < 10):
                            cv2.line(img, (155, int(bar)),
                                     (190, int(bar)), (254, 118, 136), 40)
                        elif (int(percentage) >= 10 and (int(percentage) < 100)):
                            cv2.line(img, (155, int(bar)),
                                     (200, int(bar)), (254, 118, 136), 40)
                        else:
                            cv2.line(img, (155, int(bar)),
                                     (210, int(bar)), (254, 118, 136), 40)

                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        draw.rounded_rectangle((fw-280, (fh//2)-230, fw-80, (fh//2)-30), fill=color,
                                               radius=40)

                        draw.text(
                            (145, int(bar)-17), f"{int(percentage)}%", font=font, fill=(255, 255, 255))
                        draw.text(
                            (fw-228, (fh//2)-229), f"{int(bcount)}", font=font1, fill=(255, 255, 255))
                        draw.text(
                            (fw-300, (fh//2)+200), f"{int(10-bcount)}", font=font1, fill=(255, 0, 0))
                        draw.text(
                            (fw-280, (fh//2)+400), f"More to Go!", font=font, fill=(0, 0, 255))
                        draw.text(
                            (fw-1800, (fh//2)-450), feedback, font=font2, fill=(0, 0, 0))
                        img = np.array(im)

                if drawskeleton:
                    for idx in range(output.shape[0]):
                        plot_skeleton_kpts(img, output[idx, 7:].T, 3)

                if webcam:
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection", img_)
                    cv2.waitKey(1)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)

                if frame_count == 95:
                    break
                
            else:
                break
        
        cap.release()
        out.release()
        #cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")

        command = "ffmpeg -y -i {}.mp4 {}.mp4".format(out_video_name, out_video_name + "_conv")

        subprocess.run(command, shell=True)
