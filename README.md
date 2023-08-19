Create Virtual Environment and activate the virtual environment.

1. Create a virtual environment (Install this on Anaconda Prompt)
	conda create -n yolov7_custom 
2. To activate the virtual environment
	conda activate yolov7_custom

---------------------------------------------------------------------------------------------------------------------------------------------
Scripts -
	app.py - Python flask application that needs to run on a web server to handle the website communication with the YOLO model.
	index.html - Html file to manage the website webpage and handle UI.
	The other scripts for each exercises is named with their following exercise names.

----------------------------------------------------------------------------------------------------------------------------------------------
ffmpeg -
	This is used to convert the processed video to mp4 format.
	Install the executable file from the official webiste and add it to the system'S environment variables.
	
	To check if it is installed succesfuly run: ffmpeg -version

----------------------------------------------------------------------------------------------------------------------------------------------
How to run:

We are using publicly available yolov7 model from github repo: https://github.com/WongKinYiu/yolov7.git
1. Install the necessary dependencies
	pip install -r requirements.txt
	FFMPEG needs to be installed. 
	
2. Click on [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) to get the pre trained model weights pytorch file and place it the yolov7 directory.

3. Run python app.py to run the website on local server.	


Note: If you wish to run this on GPU, then:
You need to install CUDA that is compatible with the GPU available on your machine. Refer official cuda installation and add them to the enviromnment variables for GPU acceleration.

-----------------------------------------------------------------------------------------------------------------------------------------------




