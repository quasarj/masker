
#import cv2


#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread("D:/UAMS/Project_defacer/masker_test_output/volumerendering_b4_cpw.png")
#img = cv2.imread("D:/UAMS/Project_defacer/masker_test_output/price-vincent-03-g.jpg")
#img = cv2.imread("D:/UAMS/Project_defacer/masker_test_output/350px-Vincentprice.png")


#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#faces=face_cascade.detectMultiScale(image=img, scaleFactor=3, minNeighbors=None)
#for (x, y, w, h) in faces:
#   cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#cv2.imwrite("D:/UAMS/Project_defacer/masker_test_output/facedetected.png", img)


#print('Successfully saved')


########################

#import mtcnn
# face detection with mtcnn on a photograph
#from matplotlib import pyplot
#from mtcnn.mtcnn import MTCNN
# load image from file
#filename = "D:/UAMS/Project_defacer/masker_test_output/price-vincent-03-g.jpg"
#filename = "D:/UAMS/Project_defacer/masker_test_output/volumerendering_b4_cpw.png"
#pixels = pyplot.imread(filename)
# create the detector, using default weights
#detector = MTCNN()
# detect faces in the image
#detector.detect_faces(pixels)
#faces = detector.detect_faces(pixels)
#for face in faces:
#	print(face)

############################

from retinaface import RetinaFace
#filename = "D:/UAMS/Project_defacer/masker_test_output/price-vincent-03-g.jpg"
#filename = "D:/UAMS/Project_defacer/masker_test_output/volumerendering_b4_cpw.png"
filename = "D:/UAMS/Project_defacer/masker_test_output/volumerendering_b4_T2.png"
#filename = "D:/UAMS/Project_defacer/masker_test_output/volumerendering_b4_FLAIR.png"
faces=RetinaFace.detect_faces(filename,threshold=0.001)
len(faces)
faces["face_1"]


## Find largest face
for key in faces:
    width=faces[key]["facial_area"][2]-faces[key]["facial_area"][0]
    height=faces[key]["facial_area"][3]-faces[key]["facial_area"][1]
    print(str(width)+" "+str(height))



#faces = RetinaFace.extract_faces(filename, align = True)
#for face in faces:
  #plt.imshow(face)
  #plt.show()

import matplotlib.pyplot as plt
faces = RetinaFace.extract_faces(filename, threshold=0.001)
plt.imshow(faces[7])
plt.show()
##
int(np.round(160/3,0))

