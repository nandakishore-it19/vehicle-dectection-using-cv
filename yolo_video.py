import numpy as np
import argparse
import imutils
import time
import cv2
import os
from scipy import spatial
import pytesseract
import datetime 
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

import pandas as pd

time_now=datetime.datetime.now()
filename=time_now.strftime("%H-%M-%S-%p__%d-%m-%Y")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())



classname = []
list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
FRAMES_BEFORE_CURRENT = 10 

bicycle__temp=0
car__temp=0
motorbike__temp=0
bus__temp=0
truck__temp=0

# bicycle__temp,car__temp,motorbike__temp,bus__temp,truck__temp

def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, #Image
		'Detected Vehicles: ' + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0xFF, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def get_vehicle_count(boxes, class_names):
	total_vehicle_count = 0 
	dict_vehicle_count = {} # dictionary with count of each distinct vehicles detected
	for i in range(len(boxes)):
		class_name = class_names[i]
		if(class_name in list_of_vehicles):
			total_vehicle_count += 1
			dict_vehicle_count[class_name] = dict_vehicle_count.get(class_name,0) + 1

	return total_vehicle_count, dict_vehicle_count


def insertData(da):
	licePlate=[]
	licePlate.append(da)
	df=pd.DataFrame(licePlate)
	df.to_csv(filename+'.csv',mode='a', index=False)




def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame,bicycle__temp,car__temp,motorbike__temp,bus__temp,truck__temp):
	current_detections = {}
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			centerX = x + (w//2)
			centerY = y+ (h//2)



			# When the detection is in the list of vehicles, AND
			# it crosses the line AND
			# the ID of the detection is not present in the vehicles
			if (LABELS[classIDs[i]] in list_of_vehicles):
				# if(LABELS[classIDs[i]]=='car'):
				# 	car__temp=car__temp+1
				# 	print("Car......................",car__temp)
				current_detections[(centerX, centerY)] = vehicle_count 
				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
					vehicle_count += 1
					if(LABELS[classIDs[i]]=='bicycle'):
						bicycle__temp=bicycle__temp+1
					if(LABELS[classIDs[i]]=='car'):
						car__temp=car__temp+1
					if(LABELS[classIDs[i]]=='motorbike'):
						motorbike__temp=motorbike__temp+1
					if(LABELS[classIDs[i]]=='bus'):
						bus__temp=bus__temp+1
					if(LABELS[classIDs[i]]=='truck'):
						truck__temp=truck__temp+1
					# vehicle_crossed_line_flag += True
				# else: #ID assigning
					#Add the current detection mid-point of box to the list of detected items
				# Get the ID corresponding to the current detection

				ID = current_detections.get((centerX, centerY))
				# If there are two detections having the same ID due to being too close, 
				# then assign a new ID to current detection.
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1
					if(LABELS[classIDs[i]]=='bicycle'):
						bicycle__temp=bicycle__temp+1
					if(LABELS[classIDs[i]]=='car'):
						car__temp=car__temp+1
					if(LABELS[classIDs[i]]=='motorbike'):
						motorbike__temp=motorbike__temp+1
					if(LABELS[classIDs[i]]=='bus'):
						bus__temp=bus__temp+1
					if(LABELS[classIDs[i]]=='truck'):
						truck__temp=truck__temp+1

				#Display the ID at the center of the box
				cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

	return vehicle_count, current_detections,bicycle__temp,car__temp,motorbike__temp,bus__temp,truck__temp




labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions

plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

vs = cv2.VideoCapture(args["input"])

writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = 200
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream

list_of_vehicles = ["car","bus","motorbike","truck","bicycle"]
num_frames, vehicle_count = 0, 0
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]

car__Count=0
bus__Count=0
motorBike__Count=0
truck__Count=0
bicycle__Count=0

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	
	
	# ************************************Licence Plate
	# gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# plate = plat_detector.detectMultiScale(gray_video,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))
	
	# for (x,y,w,h) in plate:
	# 	cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
	# 	frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h,x:x+w],ksize=(10,10))
	# 	cv2.putText(frame,text='License Plate',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11,17,17)
	edge = cv2.Canny(gray, 170,200)
	cnts, new = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	image1 = frame.copy()
	cv2.drawContours(image1,cnts,-1,(0,225,0),3)
	cnts =sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
	NumberPlateCount = None
	image2 = frame.copy()
	cv2.drawContours(image2,cnts,-1,(0,255,0),3)
	count = 0
	name = 1

	for i in cnts:
		perimeter = cv2.arcLength(i, True)
		approx = cv2.approxPolyDP(i,0.02*perimeter,True)
		if(len(approx)==4):
			NumberPlateCount = approx
			mask = np.zeros(gray.shape,np.uint8)
			new_image = cv2.drawContours(mask,[NumberPlateCount],0,255,-1,)
			new_image = cv2.bitwise_and(frame,frame,mask=mask)
			(x, y) = np.where(mask == 255)
			(topx, topy) = (np.min(x), np.min(y))
			(bottomx, bottomy) = (np.max(x), np.max(y))
			crp_img = frame[topx:bottomx+1, topy:bottomy+1]


            # x,y,w,h = cv2.boundingRect(i)
            # print("--->",x,y,w,h)
            # crp_img = image[y:y+h, x:x+w]
			text = pytesseract.image_to_string(crp_img, config='--psm 11')
			# print(text,"--->",len(text))
			# cv2.putText(frame, text, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,250,250), 2)
			# cv2.imwrite(str(name)+ '.png', crp_img)
			name +=1
			if(len(text)>=8):
				cv2.drawContours(frame,[NumberPlateCount], -1,(0,255,0),3)
				cv2.putText(frame, text, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,250,250), 2)
				if(text != 0):
					# insertData(text)

						




					print(text)
			break
    


	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				classname.append(LABELS[classID])

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	


	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])

			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)
	
	vehicle_count, current_detections,bicycle__temp__,car__temp__,motorbike__temp__,bus__temp__,truck__temp__ = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame,bicycle__temp,car__temp,motorbike__temp,bus__temp,truck__temp)

	bicycle__Count=bicycle__Count+bicycle__temp__
	car__Count=car__Count+car__temp__
	motorBike__Count=motorBike__Count+motorbike__temp__
	bus__Count=bus__Count+bus__temp__
	truck__Count=truck__Count+truck__temp__



	displayVehicleCount(frame, vehicle_count)
	cv2.putText(
		frame, 'Bicycle Count: ' + str(bicycle__Count),(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.8,
		(0, 0xFF, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
	cv2.putText(
		frame, 'Car Count: ' + str(car__Count),(20, 100),cv2.FONT_HERSHEY_SIMPLEX,0.8,
		(0, 0xFF, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
	cv2.putText(
		frame, 'MotorBike Count: ' + str(motorBike__Count),(20, 140),cv2.FONT_HERSHEY_SIMPLEX,0.8,
		(0, 0xFF, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
	cv2.putText(
		frame, 'Bus Count: ' + str(bus__Count),(20, 180),cv2.FONT_HERSHEY_SIMPLEX,0.8,
		(0, 0xFF, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
	cv2.putText(
		frame, 'Truck Count: ' + str(truck__Count),(20,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,
		(0, 0xFF, 0),2,cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

	

	# check if the video writer is None

	total_vehicles, each_vehicle = get_vehicle_count(boxes, classname)
	# print("Total vehicles in image", total_vehicles)


	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	

		# Updating with the current frame detections
	previous_frame_detections.pop(0) #Removing the first frame from the list
	# previous_frame_detections.append(spatial.KDTree(current_detections))
	previous_frame_detections.append(current_detections)

# release the file pointers
print("[INFO] cleaning up...")
# writer.release()
vs.release()


# python yolo_video.py --input videos/highway.mp4 --output output/highway_output.avi --yolo yolo-coco
# python yolo_video.py --input videos/video1.mp4 --output output/video1_output.avi --yolo yolo-coco
# python yolo_video.py --input videos/video2.mp4 --output output/video2_output.avi --yolo yolo-coco