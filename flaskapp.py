from flask import Flask, render_template,redirect,request
import pandas as pd
import time
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view_attendance', methods=['POST'])
def view_attendance():
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    message ="Here is a list of student"
    df = pd.read_csv("attendanceFolder/attendance_" + date + ".csv")
    return render_template('view.html',message=message, df=df.to_html(classes="table table-striped"))

@app.route('/submit_student', methods=['POST'])
def add():
    import cv2, pickle, os
    import numpy as np

    video=cv2.VideoCapture(0)
    facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces_data=[]

    i=0
    name = request.form['studentName']
    ID = request.form['studentID']
    Class = request.form['studentClass']

    while True:
        ret,frame=video.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(gray, 1.3 ,5)
        for (x,y,w,h) in faces:
            crop_img=frame[y:y+h, x:x+w, :]
            resized_img=cv2.resize(crop_img, (50,50))
            if len(faces_data)<=10 and i%10==0:
                faces_data.append(resized_img)
            i=i+1
            cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        cv2.imshow("Frame",frame)
        k=cv2.waitKey(1)
        if k==ord('q') or len(faces_data)==10:
            break
    video.release()
    cv2.destroyAllWindows()

    faces_data=np.asarray(faces_data)
    faces_data=faces_data.reshape(30, -1)


    if 'names.pkl' not in os.listdir('dataset/'):
        names=[name]*30
        with open('dataset/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('dataset/names.pkl', 'rb') as f:
            names=pickle.load(f)
        names=names+[name]*30
        with open('dataset/names.pkl', 'wb') as f:
            pickle.dump(names, f)
        print("NAME ADDED")
    if 'faces_data.pkl' not in os.listdir('dataset/'):
        with open('dataset/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('dataset/faces_data.pkl', 'rb') as f:
            faces=pickle.load(f)
        faces=np.append(faces, faces_data, axis=0)
        with open('dataset/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)
            print("faces added")
    return redirect('/')

@app.route('/record', methods=['POST'])
def record():
    return render_template('add.html')

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    from sklearn.neighbors import KNeighborsClassifier
    import cv2, os, csv, pickle, time
    from datetime import datetime

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model =KNeighborsClassifier(n_neighbors=10)
    columns = ["Student Name", "Arrival Time"]
    bgimage = cv2.imread("ashesi.jpg")
    with open('dataset/names.pkl','rb') as dnames:
        names = pickle.load(dnames)

    with open('dataset/faces_data.pkl','rb') as dfaces:
        faces = pickle.load(dfaces)

    print(faces.shape)
    print(len(names))

    model.fit(faces,names)

    while True:
        ret, frame =video.read()
        grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(grayscale,1.3,5)

        for (x,y,w,h) in faces:
            cropped_image =  frame[y:y+h, x:x+w,:]
            resized_image = cv2.resize(cropped_image, (50,50)).flatten().reshape(1,-1)
            output = model.predict(resized_image)

            current_time= time.time()
            datestamp =  datetime.fromtimestamp(current_time).strftime("%d-%m-%Y")
            timestamp =  datetime.fromtimestamp(current_time).strftime("%H:%M:%S")
            existance = os.path.isfile("attendanceFolder/attendance_"+ datestamp + ".csv")
            color_outer = (128, 0, 128)
            color_inner = (128, 0, 128)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color_outer, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_outer, 3)
            cv2.rectangle(frame, (x, y-40), (x+w, y), color_inner, -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_inner, 5)
            attendance = [str(output[0]),str(timestamp)]
            
        bgimage[162:162+480,55:55+640]= frame
        cv2.imshow("Frame",bgimage)
        k = cv2.waitKey(1)
        
        if k==ord("0"):
            if existance:
                column_index = 0
                column_values = []
                with open("attendanceFolder/attendance_"+ datestamp+ ".csv", 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        if len(row) > column_index:
                            column_values.append(row[column_index])
                with open("attendanceFolder/attendance_"+ datestamp+ ".csv","+a") as attendancefile:
                    writer = csv.writer(attendancefile)
                    if str(output[0]) not in column_values:
                        writer.writerow(attendance)
                    else:
                        print("Attendance already taken")
                attendancefile.close()
            else:
                with open("attendanceFolder/attendance_"+ datestamp+ ".csv","+a") as attendancefile:
                    writer = csv.writer(attendancefile)
                    writer.writerow(columns)
                    writer.writerow(attendance)
                    print("Hurray you are the first in class")
                attendancefile.close()
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)