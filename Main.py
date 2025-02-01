from ultralytics import YOLO
import cv2, queue, threading, time, os, datetime, face_recognition, math, numpy as np, mysql.connector, base64, io
from sort import Sort
from PIL import Image
import tampilan_evidence

def tampilan_evidence_1(x,y,z,m):
    tampilan_evidence.save_cuplikan(x,y,z,m)

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  database="database_investigator"
)

mycursor = mydb.cursor()

dtp_long = 106.005913
dtp_lat = -6.035569
encodeListKnown = 0
model = YOLO('AI_Model/Train_4.pt')
tracker = Sort(max_age=30, min_hits=0, iou_threshold=0.2)
face_recognation_data = [] 
sus_movement_data = [] 
forbidden_object_data = [] 
dict_data_victim = [] 
fps = 0
name = 'Unkown'
list_name = []
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def FaceRecog1(img, encodeListKnown, className):
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces=face_cascade.detectMultiScale(imgS,1.3,5)

    breakorno = 0
    name = 'Unknown'
    index = None
    ptime = 0
    fps1 = 0
    saveUnkown = 0
    bbox2=()
    for (p,q,r,s) in faces:
        bbox = []
        x1 = int(p)
        y1 = int(q)
        x2 = int(p+r)
        y2 = int(q+s)
        bbox.append(y1)
        bbox.append(x2)
        bbox.append(y2)
        bbox.append(x1)
        bbox2 += (bbox,)
    encodesCurFrame = face_recognition.face_encodings(imgS, bbox2)
    for encodeFace, faceloc in zip(encodesCurFrame, bbox2) :
        if not encodeListKnown == []:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            print(matches)
            if True in matches :
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
            index = 'Unknown_done'
        if not encodeListKnown == [] and True in matches and matches[matchIndex] and faceDis[matchIndex] < 0.4:
            index = matchIndex
    print(index)
    return(index)
def BuildEncodings(images):
    encodelist = []
    for img in images :
     try :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
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
    mycursor.execute("SELECT * FROM target_pribadi")
    myresult = mycursor.fetchall()
    for x in myresult:
        image_data = base64.b64decode((str(x[2])[2:-1])[(str(x[2])[2:-1]).index(',')+1:])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(image)
        className.append([x[0], x[1]])
    global encodeListKnown
    encodeListKnown = BuildEncodings(images)
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
main()
IddalamPerekaman = -1
total_frame_koordinat = []
framegone = 0
list_unkown_terdetek = []
bool_checked_id = 0
cap = cv2.VideoCapture(2)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
# cap = VideoCapture("/Users/darrenchailand/Documents/VSCode/OPSI_2023/Dataset/Vid_test.mov")
index1 = 0
while True:
    s,img = cap.read()
    try:
        raw_img = img.copy()
    except:
        # cap = cv2.VideoCapture("test_vid.MOV")
        # s,img = cap.read()
        print("Fail")
        pass
    fps += 1
    result = model(img, stream=True)
    detections = np.empty((0, 5))
    coordinates_people = []
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
    currentList = []
    for results in resultstrack:
        name = "Unknown"
        index = None
        booltru = 0
        x1, y1, x2, y2, Id = results
        currentList.append(Id)
        if fps > 0 :
            if x1 < 0 : x1 = 0
            if y1 < 0 : y1 = 0
            if x2 < 0 : x2 = 0
            if y2 < 0 : y2 = 0
            prosesIMG = img[int(y1):int(y2), int(x1):int(x2)]
            if not len(list_name) == 0: 
                for x in list_name:
                    if x[0] == Id:
                        name = x[1]
                        if x[1] == "Tidak_Di_Kenal" : pass
            if name == "Unknown":
                index = FaceRecog1(prosesIMG, encodeListKnown, className)
                if index is None:
                    name = "Unknown"
                elif index == 'Unknown_done':
                    name = 'Unknown_done'
                else:
                    name = className[index][1].upper()
                if not name == "Unknown" and not name =="Unknown_done":
                    list_name.append([Id, name, dtp_long, dtp_long, datetime.datetime.now(), 0, 0]) 
                    if IddalamPerekaman == -1:
                        file_name = "File/Video_Data/" + name + "_" + str(datetime.datetime.now()) +".avi"
                        nama_orang_target = name
                        resultx_raw = cv2.VideoWriter(str(file_name), cv2.VideoWriter_fourcc(*'MJPG'), 5, size)
                        IddalamPerekaman = Id
                        index1 = className[index][0]
                elif name == "Unknown_done" :
                    bool_checked_id = 0
                    for x in list_unkown_terdetek :
                        if x[0] == Id:
                            bool_checked_id = 1
                            x[1] += 1
                            if x[1] > 100: list_name.append([Id, "Tidak_Di_Kenal"])
                    if bool_checked_id == 0 :
                        list_unkown_terdetek.append([Id, 0])
        if IddalamPerekaman == Id : total_frame_koordinat.append([x1, y1, x2, y2])
        if not len(list_name) == 0: 
            for x in list_name :
                if x[0] == Id:
                    name = x[1]      
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(img, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    if IddalamPerekaman in currentList:
        resultx_raw.write(raw_img)
        print("RECORDED")
        framegone = 0 
    elif  not IddalamPerekaman == -1:
        framegone += 1
        if framegone > 5 :
            resultx_raw.release()
            for x in list_name :
                if x[0] == IddalamPerekaman : 
                    x[5] = datetime.datetime.now()
                    x[6] = total_frame_koordinat
                    sql = "INSERT INTO detail_target_pribadi (dtp_start_date, dtp_end_date, dtp_long, dtp_lat, target_pribadi_tp_id) VALUES (%s, %s, %s, %s, %s)"
                    val = (x[4],x[5], dtp_long, dtp_lat, index1)

                    mycursor.execute(sql, val)

                    mydb.commit()
                    with open("File/data_koordinat.txt", 'w') as file:
                        for data in total_frame_koordinat:
                            file.write(str(data) + ', \n')
                        print(f'Data saved successfully.')
                    thread1 = threading.Thread(target=tampilan_evidence_1, args=(mycursor.lastrowid,file_name,nama_orang_target,index1,))
                    thread1.start()
                    
            IddalamPerekaman = -1
    if fps > 30 : fps=0       
    else : fps += 1 
    cv2.imshow("Image", img)
    cv2.waitKey(1)