import cv2
import os
import shutil

# input video
dirname = '../YawDD dataset/Mirror/Male_mirror'
savedir = '../YawDD dataset/Mirror/Male_mirror_jpg'
files = os.listdir(dirname)
for filename in files:
    savedpath = savedir + '/' + filename.split('.')[0] + '/'
    filepath = dirname + '/' + filename
    isExists = os.path.exists(savedpath)
    if not isExists:
        os.makedirs(savedpath)
        print('path of %s is build' % (savedpath))
    else:
        shutil.rmtree(savedpath)
        os.makedirs(savedpath)
        print('path of %s already exist and rebuild' % (savedpath))
    # the gap of frame
    count = 5
    videoCapture = cv2.VideoCapture(filepath)
    i = 0
    j = 0
    while True:
        success, frame = videoCapture.read()
        if not success:
            print('video is all read')
            break
        x, y = frame.shape[0:2]
        frame = cv2.resize(frame, (int(y / 2), int(x / 2)))
        if (i % count == 0):
            j += 1
            savedname = filename.split('.')[0] + '_frame' + str(i) + '.jpg'
            cv2.imwrite(savedpath + savedname, frame)
        i += 1