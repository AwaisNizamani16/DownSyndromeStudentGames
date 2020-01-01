import numpy as np
import argparse as arg
import time
import cv2
import os

param = arg.ArgumentParser()
param.add_argument("-i", "--pic", required=True,
	help="Pictures Path")
param.add_argument("-y", "--TFlearn", required=True,
	help="Path to YOLO Weights for applying transfer Learning")
param.add_argument("-c", "--confidence", type=float, default=0.5,
	help="Best Bounding Box Selection")
param.add_argument("-t", "--threshold", type=float, default=0.3,
	help="Making Edges and Bounding Boxes Better")
args = vars(param.parse_args())

# Applying Transfer Learning due to which getting data set values
NAMES = os.path.sep.join([args["TFlearn"], "coco.names"])
OUTPUT = open(NAMES).read().strip().split("\n")

# Initializing Colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(OUTPUT), 3),
	dtype="uint8")

# Weights and Yolo model so that we don't need to train for each class
# Files are taken from darknet trained Weights
W = os.path.sep.join([args["TFlearn"], "yolov3.weights"])
C = os.path.sep.join([args["TFlearn"], "yolov3.cfg"])

# Algorithm can detect 80 distinct classes as per the Darknet analysis
# Darknet model is accessible through CV library
print("Applying Transfer Learning ON YOLO")
YOLO = cv2.dnn.readNetFromDarknet(C, W)

# Loading the image
PIC = cv2.imread(args["pic"])
(H, W) = PIC.shape[:2]

# getting output layers all labeled information
ln = YOLO.getLayerNames()
# extracting each output layer label that were detected
ln = [ln[i[0] - 1] for i in YOLO.getUnconnectedOutLayers()]

# applying LOG BLOB to extract interest point of the Picture
# afterthat applying YOLO to get the Image Output with their probabilities
# interest points --> IP are provided as an input to out algo
IP = cv2.dnn.blobFromImage(PIC, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
YOLO.setInput(IP)
Begin = time.time()
OutputLayer = YOLO.forward(ln)
Finish = time.time()

# Time Taken By Algo
print("Time Taken: {:.4f} Seconds".format(Begin - Finish))

# As there are multiple bounding boxes for each detection confidence score is being checked
# INPUTName of the classes, respectively
BOX = []
CONFs = []
INPUTNames = []

# getting output layers detections
for outcomes in OutputLayer:
	# extracting each outcome detected as an object
	for Objects in outcomes:
		# extracting it's Name and object confidence scores
		# Extracting each detection by excluding it's starting 4 entries
		Prob = Objects[5:]
		INPUTName = np.argmax(Prob)
		CONF = Prob[INPUTName]

		# filtering weaker bounding boxes
		# Checking with the minimum probability Set
		if CONF > args["confidence"]:
			# Apply Boudning Box
			# Getting Image
			# Returing Center of the Image is that X Y
			# Assigning box on their height and width
			boundingbox = Objects[0:4] * np.array([W, H, W, H])
			(Xpoint, Ypoint, width, height) = boundingbox.astype("int")

			# Finding Corners X for Left
			# Y for Right
			x = int(Xpoint - (width / 2))
			y = int(Ypoint - (height / 2))

			# list updation of BOX CONF and INPUTName
			# and class IDs
			BOX.append([x, y, int(width), int(height)])
			print(BOX)
			CONFs.append(float(CONF))
			INPUTNames.append(INPUTName)
            
            # Applying Non Maxima Supression on
# boxes
coord = cv2.dnn.NMSBoxes(BOX, CONFs, args["confidence"],
	args["threshold"])

# Check for detection to get coord
if len(coord) > 0:
	# get all the indexes of the coordinates
	for idxs in coord.flatten():
		# extract the bounding box coordinates
		(x, y) = (BOX[idxs][0], BOX[idxs][1])
		(w, h) = (BOX[idxs][2], BOX[idxs][3])

		# creating a box for the image
		color = [int(ID) for ID in COLORS[INPUTNames[idxs]]]
		cv2.rectangle(PIC, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.2f}".format(OUTPUT[INPUTNames[idxs]], CONFs[idxs])
		print(text)
		cv2.putText(PIC, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# Viewing the output image
cv2.imshow("pic", PIC)
cv2.waitKey(0)
