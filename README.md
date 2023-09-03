Real-Time Posture Correction in Gym Exercises: A Computer Vision-Based Approach for Performance Analysis, Error Classification and Feedback.

Hitesh Kotte, Milos Kravcik and Nghia Duong-Trung


https://github.com/hiteshkotte/MILeS_2023/assets/35593884/7835485c-a494-40b1-826d-0bdde969e541


Create Virtual environment and activate the Virtual environment.

1. Create a virtual environment (Install this preferrably on Anaconda Prompt. 
This will create a empty virtual environment.

		conda create -n yolov7_custom

3. To activate the virtual environment:

		conda activate yolov7_custom

---------------------------------------------------------------------------------------------------------------------------------------------
Scripts :

app.py - Python flask application that needs to run on a web server to handle the website communication with the YOLO model.

index.html - Html file to manage the website webpage and handle UI. (This can be found inside the "Templates" folder)

The other scripts for each exercises is named with their following exercise names.

----------------------------------------------------------------------------------------------------------------------------------------------
ffmpeg :

This is used to convert the processed video to mp4 format.
This is already installed from the requirements.txt.

 To check if it is installed succesfully: 
 
 		ffmpeg -version



----------------------------------------------------------------------------------------------------------------------------------------------
How to run:

We are using publicly available yolov7 model from github repo: https://github.com/WongKinYiu/yolov7.git

1. Install the necessary dependencies
   
		pip install -r requirements.txt
	
3. Click on [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) to get the pre trained model weights pytorch file and place it the yolov7 directory.

4. Run the python app.py to run the website on local server.
   
   		python app.py


Note: If you wish to run this on GPU, then:

You need to install CUDA that is compatible with the GPU available on your machine. Refer official CUDA installation and add them to the enviromnment variables for GPU acceleration.

-----------------------------------------------------------------------------------------------------------------------------------------------
For Input videos of exercises, Please download from the below google drive link.

	https://drive.google.com/drive/folders/10dz-wZCnio7Sub48rYiIDp3Gb_mW_Fq-?usp=sharing





