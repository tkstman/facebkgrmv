from imutils import face_utils
from PIL import Image
import os
import numpy as np
import argparse
import imutils
import dlib
import cv2



sourcefolder="C:/Users/tstone/Documents/Python/testimgs"
destinationfolder="C:/Users/tstone/Documents/Python/processed"

def signPxlVar(firstPxTuple, secPxTuple,percent):
	if len(firstPxTuple) == len(secPxTuple):
		for x,y in zip(firstPxTuple,secPxTuple):
			if(x!=y or (abs(x-y)/x)*100 <percent):
				return False
				 



for sourcepath,dirs,files in os.walk(sourcefolder):
	for file_ in files:
		if file_.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):			
			#print(os.path.join(sourcepath,file_))
			#ap = argparse.ArgumentParser()
			#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
			#ap.add_argument("-i","--image", required=True, help="path to input image")
			#args = vars(ap.parse_args())
			detector = dlib.get_frontal_face_detector()
			predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
			#predictor = dlib.shape_predictor(args["shape_predictor"])
			image = cv2.imread(os.path.join(sourcepath,file_))
			
			im = Image.open(os.path.join(sourcepath,file_))
			im = im.convert("RGBA")
			pix = im.load()
			data = np.array(im)

			#Get all the colors in a 10 x 10 pixel box at the top left of the image
			#These colors will be removed from the image
			listoCol = []
			allpixels =[]
			for x in range(70):
				for y in range(150):
					if(pix[x,y] not in listoCol):						
						listoCol.append(pix[x,y])
						#print(pix[x, y])
						
						#print("new pixel value: ",pix[x, y])
			#im = im.convert("P",palette=Image.ADAPTIVE,colors=256)
			for x in range(im.size[0]):
				for y in range(im.size[1]):
					allpixels.append(pix[x, y])
					if pix[x, y] in listoCol:
						pix[x, y] = (255, 255, 255, 255)
			#acsvfile = np.asarray(allpixels)
			#np.savetxt("img.csv",allpixels,fmt="%d", delimiter=",")
			im.show()
			#print(pix[x,y])
			#pix.show
			#print(data)
			#print(listoCol)
			#print(pix[x,y])
			#convertedimg = np.where(data==pix[x,y])
			#print(convertedimg)
			#convertedimg = Image.fromarray(convertedimg)
			#im.show()
			#im.close()
			
			#image = cv2.imread(args["image"])
			#image = imutils.resize(image, width=500)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			rects = detector(gray,1)

			# loop over the face detections
			for (i, rect) in enumerate(rects):
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				# convert dlib's rectangle to a OpenCV-style bounding box
				# [i.e., (x, y, w, h)], then draw the face bounding box
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				# show the face number
				cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				# loop over the (x, y)-coordinates for the facial landmarks
				# and draw them on the image
				for (x, y) in shape:
					cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
			# show the output image with the face detections + facial landmarks
			cv2.imshow(file_, image)
			cv2.waitKey(0)
			break
		break