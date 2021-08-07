import cv2 as cv 
import os 
path = 'imq'
images = []
classes = []
imglist = os.listdir(path)


orb = cv.ORB_create()
for cls in imglist:
    curimg = cv.imread(f'{path}/{cls}',0) # path is the image path and cl is img name 
    images.append(curimg)
    classes.append(os.path.splitext(cls)[0]) # cl is image name ,cl [0] is imge name and cl [1]is extention 
print(classes)
def descripters(images): # it returs descripter of all the image from imagelist ss
    desListOfOurImg = []
    for i in images:
        kp ,desOfOurImg = orb.detectAndCompute(i, None)
        desListOfOurImg.append(desOfOurImg)
    return desListOfOurImg

# ----------------------------------------- 
# find id from web cam img frame  

def findid(img , desListOfOurImg , thres = 15):
    kp2 , desOfWebCam = orb.detectAndCompute(img , None)
    bf = cv.BFMatcher()
    matchList = []
    outputValue = -1
    try:
        for des in desListOfOurImg:
            matches = bf.knnMatch(des,desOfWebCam,k=2) # match two descriters 
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
        print(matchList) 
    except:
        pass 

    if len(matchList)!=0:
        if max(matchList) > thres:
            outputValue = matchList.index(max(matchList))

    return outputValue      


desListOfOurImg = descripters(images)

webcam = cv.VideoCapture(0) # zero for using webcam 

while True :
    isHaveFrame , frame = webcam.read()
    originalframe = frame.copy() # coping img 
    frame  = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    valueindex =  findid(originalframe,desListOfOurImg)
    if valueindex != -1:
        cv.putText(originalframe , classes[valueindex] , (50,50),cv.Formatter_FMT_C,1,(0,0,255),2)


    cv.imshow('video' , originalframe)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break















