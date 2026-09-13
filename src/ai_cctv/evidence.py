import ast
import os
import threading
from pathlib import Path

import cv2
import moviepy.editor as mvp
from . import expression_analysis
def analyze_expressions(x,y,m):
    expression_analysis.main(x,y,m)

def save_cuplikan(dtp_id, link, nama_kena, index1):
    with open("data/coordinates.txt", "r", encoding="utf-8") as file:
        file_contents = file.read()
    koordinat_data = ast.literal_eval("[" + file_contents + "]")

    output_dir = Path(os.getenv("EVIDENCE_OUTPUT_DIR", "outputs/evidence"))
    output_dir.mkdir(parents=True, exist_ok=True)
    avi_path = output_dir / f"{dtp_id}.avi"
    mp4_path = output_dir / f"{dtp_id}.mp4"

    cap = cv2.VideoCapture(link)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    resultx_raw = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*'MJPG'), 8, size)

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

    clip = mvp.VideoFileClip(str(avi_path))
    clip.write_videofile(str(mp4_path))
    clip.close()
    thread1 = threading.Thread(
        target=analyze_expressions,
        args=(dtp_id, link, index1),
    )
    thread1.start()
