from deepface import DeepFace
import cv2, mysql.connector
import analisa_jaringan

def analisa_jaringan_1(x,y,m):
    analisa_jaringan.main(x,y,m)

def statistika_emotion(emotions_list):
    emotion_counts = {}
    total_emotions = len(emotions_list)
    for emotion in emotions_list:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    emotion_percentages = {}
    for emotion, count in emotion_counts.items():
        percentage = (count / total_emotions) * 100
        emotion_percentages[emotion] = percentage
    percentage_strings = [f"{int(percentage)}% {emotion}" 
               for emotion, percentage in emotion_percentages.items()]
    result_string = ', '.join(percentage_strings)
    return result_string

def main(dtp_id, filname, index1):
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    database="database_investigator"
    )

    mycursor = mydb.cursor()

    with open("/Users/darrenchailand/Documents/VSCode/OPSI_FINAL_2023/File/data_koordinat.txt", 'r') as file:
            file_contents = file.read()
    koordinat_data = eval("[" + file_contents + "]")

    cap = cv2.VideoCapture("File/Video_Data/DARREN CHAILAND_2023-09-29 22:02:43.010620.avi")
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    face_cascade=cv2.CascadeClassifier(haar_file)
    x = 0
    xMax = len(koordinat_data)
    Total_Emotion_List = []
    while True:
        s,img = cap.read()
        if x > xMax - 1:
            result_string = statistika_emotion(Total_Emotion_List)
            sql = "UPDATE detail_target_pribadi SET dtp_expresi = %s WHERE dtp_id = %s"
            val = (result_string, dtp_id,)
            mycursor.execute(sql, val)

            mydb.commit()
            break
        img = img[int(koordinat_data[x][1]) : int(koordinat_data[x][3]), int(koordinat_data[x][0]) : int(koordinat_data[x][2])]
        print(x, xMax-1)
        x +=1
        faces=face_cascade.detectMultiScale(img,1.3,5)
        for (p,q,r,s) in faces:
            img2 = img[q:q+s,p:p+r]
            cv2.rectangle(img, (int(p), int(q)),
                    (int(p+r), int(q+s)), (255, 0, 255), 1)
            try:
                result = DeepFace.analyze(img2, actions = ['emotion'])
                Total_Emotion_List.append(result[0]['dominant_emotion'])
                cv2.putText(img, str(result[0]['dominant_emotion']), (p,q), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            except:
                pass
        cv2.imshow("img", img)
        cv2.waitKey(1)
        thread1 = threading.Thread(target=analisa_jaringan_1, args=(dtp_id,link,index1,))
        thread1.start()

