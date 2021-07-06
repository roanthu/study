import cv2, sys, numpy, os
size = 1
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'
fn_out = 'train2'
haar_cascade = cv2.CascadeClassifier(fn_haar)
(im_width, im_height) = (19, 19)
for (subdirs,dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        subjectpath = os.path.join(fn_dir,subdir)
        subpath = os.path.join(fn_out,subdir)
        if not os.path.isdir(subpath):
            os.mkdir(subpath)
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(subpath)
        if n[0]!='.' ]+[0])[-1] + 1
        for filename in os.listdir(subjectpath):
            path = subjectpath + '\\' + filename
            # print(path)
            image = cv2.imread(path)
            height, width, channels = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
            faces = haar_cascade.detectMultiScale(mini,minSize=(10,10),scaleFactor=1.1,minNeighbors=0)
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]
                x = x -10
                y = y-10
                w = w+20
                h = h+20
                face = gray[y:y + h, x:x + w]
                # cv2.imshow("image",face)
                # face_resize = cv2.resize(face, (im_width, im_height))
                # cv2.waitKey(0)
                cv2.imwrite('%s/%s.png' % (subpath, pin), face)
                pin += 1