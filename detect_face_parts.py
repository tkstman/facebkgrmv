from imutils import face_utils
from PIL import Image
import os
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt



sourcefolder="C:/Users/tstone/Documents/Python/testimgs"
destinationfolder="C:/Users/tstone/Documents/Python/processed"

def signPxlVar(firstPxTuple, secPxTuple,percent):
	if len(firstPxTuple) == len(secPxTuple):
		for x,y in zip(firstPxTuple,secPxTuple):
			if((abs(x-y)/x)*100 >percent):
				return True
	return False			 



for sourcepath,dirs,files in os.walk(sourcefolder):
	for file_ in files:
		if file_.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):			
			#print(os.path.join(sourcepath,file_))
			#ap = argparse.ArgumentParser()
			#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
			#ap.add_argument("-i","--image", required=True, help="path to input image")
			#args = vars(ap.parse_args())
			# detector = dlib.get_frontal_face_detector()
			# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
			#predictor = dlib.shape_predictor(args["shape_predictor"])
			image = cv2.imread(os.path.join(sourcepath,file_))
			
			imagegray = cv2.cvtColor(image,cv2.COLOR_BGR2YUV) #change to grayscale COLOR_BGR2RGBA COLOR_RGB2BGR
			imageenhanced = cv2.cvtColor(imagegray, cv2.COLOR_BGR2YUV)
			imageenhanced[:,:,0] = cv2.equalizeHist(imageenhanced[:,:,0])

			imageenhancedrgb = cv2.cvtColor(imageenhanced,cv2.COLOR_YUV2RGB)
			plt.imshow(imageenhancedrgb)#, plt.axis("off")
			plt.show()

			#cv2.imshow('image with gray',imagegray)
			ret, thresh1 = cv2.threshold(imageenhancedrgb,130,255,cv2.THRESH_BINARY)
			kernel = np.ones((5,5), np.uint8)
			erosion = cv2.erode(thresh1,kernel,iterations=0)

			opening= cv2.morphologyEx(erosion,cv2.MORPH_OPEN,kernel)
			closing= cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)

			plt.imshow(closing,'gray')
			plt.xticks([]), plt.yticks([])
			plt.show()

			contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

			#cv2.imshow('imagegray',closing)
			#cv2.drawContours(closing,contours,-1,(255,255,255),4)
			#cv2.waitKey(0)



			#im = Image.open(os.path.join(sourcepath,file_))
			#im = im.convert("RGBA")
			#pix = im.load()
			#data = np.array(im)

			#Get all the colors in a 10 x 10 pixel box at the top left of the image
			#These colors will be removed from the image
			""" listoCol = []
			allpixels =[]
			for x in range(30):
				for y in range(150):
					if(pix[x,y] not in listoCol):						
						listoCol.append(pix[x,y]) """
						#print(pix[x, y])
						
			#Get the pixels that are within the percentage provided	
			""" for pxFromSample in listoCol:
				for x in range(im.size[0]):
					for y in range(im.size[1]):					
						if not signPxlVar(pxFromSample,pix[x, y],20):
							pix[x, y] = (255, 255, 255, 255)
 """

						#print("new pixel value: ",pix[x, y])
			#im = im.convert("P",palette=Image.ADAPTIVE,colors=256)
			#for x in range(im.size[0]):
			#	for y in range(im.size[1]):
			#		allpixels.append(pix[x, y])
			#		if pix[x, y] in listoCol:
			#			pix[x, y] = (255, 255, 255, 255)
			#acsvfile = np.asarray(allpixels)
			#np.savetxt("img.csv",allpixels,fmt="%d", delimiter=",")
			# im.show()
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
			""" gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			rects = detector(gray,1) """

			# loop over the face detections
			""" for (i, rect) in enumerate(rects):
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
			break """
		break