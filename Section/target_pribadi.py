import mysql.connector
import base64

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  database="database_investigator"
)

mycursor = mydb.cursor()
with open('File/Screenshot 2023-09-29 at 21.25.59.png', 'rb') as file:
    my_string = base64.b64encode(file.read())
string = "data:image/jpeg;base64," + str(my_string)[2:-1]
print(string)

sql = "INSERT INTO target_koneksi (tk_image) VALUES (%s)"

val = (string,)
mycursor.execute(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")