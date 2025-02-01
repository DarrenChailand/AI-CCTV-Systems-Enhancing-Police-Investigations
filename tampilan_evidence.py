import cv2
import moviepy.editor as mvp
import analisa_ekspresi
def analisa_ekspresi_1(x,y,m):
    analisa_ekspresi.main(x,y,m)

dtp_id = 35
def save_cuplikan(dtp_id, link, nama_kena, index1):
    with open("File/data_koordinat.txt", 'r') as file:
        file_contents = file.read()
    koordinat_data = eval("[" + file_contents + "]")

    cap = cv2.VideoCapture(link)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    resultx_raw = cv2.VideoWriter("/Applications/XAMPP/xamppfiles/htdocs/video/" + str(dtp_id)+".avi", cv2.VideoWriter_fourcc(*'MJPG'), 8, size)

    x = 0
    xMax = min(len(koordinat_data), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # Ensure xMax is not greater than the number of frames

    while x < xMax:
        ret, img = cap.read()
        if not ret:
            break
        cv2.putText(img, str(nama_kena), (int(koordinat_data[x][0]), int(koordinat_data[x][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (int(koordinat_data[x][0]), int(koordinat_data[x][1])),
                    (int(koordinat_data[x][2]), int(koordinat_data[x][3])), (255, 0, 255), 1)
        resultx_raw.write(img)
        x += 1


    resultx_raw.release()
    cap.release()

    clip = mvp.VideoFileClip("/Applications/XAMPP/xamppfiles/htdocs/video/" + str(dtp_id)+".avi")
    clip.write_videofile("/Applications/XAMPP/xamppfiles/htdocs/video/" + str(dtp_id)+".mp4")
    thread1 = threading.Thread(target=analisa_ekspresi_1, args=(dtp_id,link,index1,))
    thread1.start()
