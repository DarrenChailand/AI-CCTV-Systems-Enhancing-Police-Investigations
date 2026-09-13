import mysql.connector
import base64
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

load_dotenv()

mydb = mysql.connector.connect(
  host=os.getenv("DB_HOST", "localhost"),
  port=int(os.getenv("DB_PORT", "3306")),
  user=os.getenv("DB_USER", "root"),
  password=os.getenv("DB_PASSWORD", ""),
  database=os.getenv("DB_NAME", "database_investigator")
)

mycursor = mydb.cursor()
target_image = os.getenv("TARGET_IMAGE")
if not target_image:
  raise ValueError("Set TARGET_IMAGE to the reference image path before enrollment.")

with open(target_image, 'rb') as file:
    my_string = base64.b64encode(file.read())
string = "data:image/jpeg;base64," + str(my_string)[2:-1]
print(string)

sql = "INSERT INTO target_koneksi (tk_image) VALUES (%s)"

val = (string,)
mycursor.execute(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")
