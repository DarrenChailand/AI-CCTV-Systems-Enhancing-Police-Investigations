from ultralytics import YOLO
from deepface import DeepFace
import face_recognition
import cv2
import numpy as np
import math
from sort import Sort
import datetime
import mysql.connector, base64, io
import os
encodeListKnown = 0

def FaceRecog1(img, encodeListKnown, className):
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces=face_cascade.detectMultiScale(imgS,1.3,5)
    breakorno = 0
    name = 'Unknown'
    ptime = 0
    fps1 = 0
    saveUnkown = 0
    bbox2=()
    bbox = []
    height, width, _ = imgS.shape
    x1 = int(0)
    y1 = int(0)
    x2 = int(width)
    y2 = int(height)
    bbox.append(y1)
    bbox.append(x2)
    bbox.append(y2)
    bbox.append(x1)
    bbox2 += (bbox,)
    encodesCurFrame = face_recognition.face_encodings(imgS, bbox2)
    for encodeFace, faceloc in zip(encodesCurFrame, bbox2) :
        if not encodeListKnown == []:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            if True in matches :
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
            name = 'Unknown_done'
        if not encodeListKnown == [] and True in matches and matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = className[matchIndex]
    print(name)
    return(name)
def BuildEncodings(images):
    encodelist = []
    test = 0
    for img in images :
     try :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
        test+=1
     except:
        pass

    return encodelist
def main():
    global images
    global className
    global path_unk
    global myList
    images = []
    className = []
    mycursor.execute("SELECT * FROM target_koneksi")
    myresult = mycursor.fetchall()
    for i in myresult:
        image_data = base64.b64decode((str(i[1])[2:-1])[(str(i[1])[2:-1]).index(',')+1:])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(image)
        className.append(i[0])

    global encodeListKnown
    encodeListKnown = BuildEncodings(images)

def Pendarian_Muka_Terbaik(g, face_cascade):
    list_pic = g
    best_image = None
    best_quality = 0
    best_angle = 10
    best_image_x = None
    best_quality_x = 0
    best_angle_x = 10
    number=0
    xMax = len(list_pic)
    print(xMax)
    best_image_size_bytes= 0
    best_image_size_bytes_x = 0
    bbox2=()
    bbox = []
    for x in list_pic:
        try:
            image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            image_size_bytes = image.nbytes
            encode = face_recognition.face_encodings(image)
            if not encode == [] :
                if image_size_bytes >=  best_image_size_bytes:
                        try:
                            result = DeepFace.analyze(image, actions = ['emotion'])
                        except:
                            result = "Not"
                        if result == "neutral":
                            best_image = x.copy()
                            best_image_size_bytes = image_size_bytes
                if best_image is None and image_size_bytes >= best_image_size_bytes_x:
                        best_image_x = x.copy()
                        best_image_size_bytes_x = image_size_bytes
        except:
            pass
    if best_image is not None:
        print("noexpresi")
        return best_image
    elif best_image_x is not None:
        print("berexpresi : " + str(best_image_size_bytes_x) + " "+ str(best_angle_x))
        return best_image_x
    else:
        return None
def Pencarian_Muka_ID(image, face_cascade):
    faces=face_cascade.detectMultiScale(image,1.3,5)
    for (p,q,r,s) in faces:
        image = image[q:q+s,p:p+r]
    if len(faces) == 0:
        print("No faces detected")
        return None
    return image

def main(dtp_id, filname, index1):
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    database="database_investigator"
    )

    mycursor = mydb.cursor()

    tracker = Sort(max_age=30, min_hits=0, iou_threshold=0.1)
    model = YOLO('AI_Model/Train_4.pt')
    cap = cv2.VideoCapture(filname)
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    with open("File/data_koordinat.txt", 'r') as file:
            file_contents = file.read()
    koordinat_data = eval("[" + file_contents + "]")

    Id_Muka_List = []
    appended = 0
    z = 0
    while True:
        appended = 0
        s, img = cap.read()
        if not s:
            break
        cv2.rectangle(img, (int(koordinat_data[z][0]), int(koordinat_data[z][1])),(int(koordinat_data[z][2]), int(koordinat_data[z][3])), (255, 0, 255), -1)
        z += 1
        result = model(img, stream=True)
        detections = np.empty((0, 5))
        for r in result:
            boxes = r.boxes
            for box in boxes:
                conf = math.ceil((box[0].conf * 100)) / 100
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    currentarray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentarray))
        resultstrack = tracker.update(detections)
        for results in resultstrack:
            x1, y1, x2, y2, Id = results
            image = img[int(y1):int(y2),int(x1):int(x2)]
            muka = Pencarian_Muka_ID(image, face_cascade)
            if muka is not None:
                for x in Id_Muka_List:
                    if x[0] == Id :
                        x.append(muka)
                        appended = 1
                if appended == 0 :
                    Id_Muka_List.append([Id, muka])
        cv2.imshow("vid", img)
        cv2.waitKey(1)

    file_name = None

    Muka_Koneksi = []
    for x in Id_Muka_List:
        print("Start Proses " + str(x[0]))
        Id_Best = x[0]
        x.remove(x[0])
        saved_face = Pendarian_Muka_Terbaik(x, face_cascade)
        if saved_face is not None:
            Muka_Koneksi.append(saved_face)

    main()

    Id_Koneksi_Target = []
    print(Muka_Koneksi)
    for Muka_Koneksi_Event in Muka_Koneksi:
        name = FaceRecog1(Muka_Koneksi_Event, encodeListKnown, className)
        if name == "Unknown_done":
            cv2.imshow("Muka Koneksi " + str(name), Muka_Koneksi_Event)
            cv2.waitKey(0)
            file_name = "File/" + str(datetime.datetime.now()) + ".jpg"
            cv2.imwrite(str(file_name), Muka_Koneksi_Event)
            with open(str(file_name), 'rb') as file:
                my_string = base64.b64encode(file.read())
            string = "data:image/jpeg;base64," + str(my_string)[2:-1]
            sql = "INSERT INTO target_koneksi (tk_image) VALUES (%s)"
            val = (string,)
            mycursor.execute(sql, val)
            mydb.commit()
            last_inserted_id = mycursor.lastrowid
            Id_Koneksi_Target.append(last_inserted_id)
            os.remove(file_name)

            mydb.commit()
        elif name == "Unknown":
            pass
        else :
            if not name in Id_Koneksi_Target:
                Id_Koneksi_Target.append(name)
                cv2.imshow("Muka Koneksi " + str(name), Muka_Koneksi_Event)
                cv2.waitKey(1)

    print("Id Face Jaringan : " + str(Id_Koneksi_Target))
    for x in Id_Koneksi_Target:
        sql = "INSERT INTO detail_target_koneksi (target_pribadi_tp_id, detail_target_pribadi_dtp_id, taregt_koneksi_tk_id) VALUES (%s,%s,%s)"
        val = (index1, dtp_id, x)
        mycursor.execute(sql, val)
        mydb.commit()

    