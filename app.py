import cv2
import random
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy

#### Defining Flask App
app = Flask(__name__)

# creating database model
'''app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:password@localhost/(database_name)'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class Registration(db.Model):
    __tablename__ = 'Registration'
    id = db.Column(db.Integer,primary_key= True)
    name = db.Column(db.String(200))
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    email = db.Column(db.String(200))
    phone_no = db.Column(db.Integer)
    disease = db.Column(db.String(100))

    def __init__(self,name, age, weight, email, phone_no, disease):
        self.name = name
        self.age = age
        self.weight = weight
        self.email = email
        self.phone_no = phone_no
        self.disease = disease
        

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        weight = request.form['weight']
        email = request.form['email']
        phone_no = request.form['phone_no']
        disease = request.form['disease']
        # print(customer, dealer, rating, comments)
        if name == '' or age == '' or email == ''
            or phone_no == '' or disease == '':
            return render_template("index.html",message = 'Please enter the required field')
        if db.session.query(Feedback).filter(Feedback.customer == customer).count() == 0:
            data = Feedback(name, age, weight, email, phone_no, disease)
            db.session.add(data)
            db.session.commit()
            return render_template("success.html")
        return render_template("index.html",message = 'You have already submitted feedback.')
   
'''
# def generate_user_id(name, email, mobile_number):
#     user_id = random.randint(100000, 999999)
#     while user_id_exists(user_id):
#         user_id = random.randint(100000, 999999)
#     save_user_id(name, email, mobile_number, user_id)
#     return user_id

# def user_id_exists(user_id):
#     print('check if user_id exists in database')
#     #return true if it exists, else false

# def save_user_id(name, email, mobile_number, user_id):
#     print('save the user_id to database')

# name = input("Enter your name: ")
# email = input("Enter your email: ")
# mobile_number = input("Enter your mobile number: ")

# user_id = generate_user_id(name, email, mobile_number)

# print("Your User ID is:", user_id)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Patient_Registration'):
    os.makedirs('Patient_Registration')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Patient_Registration-{datetoday}.csv' not in os.listdir('Patient_Registration'):
    with open(f'Patient_Registration/Patient_Registration-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)  #according to suitation n_neighbors ka value increase kar dena
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_data():
    df = pd.read_csv(f'Patient_Registration/Patient_Registration-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_data(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Patient_Registration/Patient_Registration-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Patient_Registration/Patient_Registration-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    # names,rolls,times,l = extract_data()    
    return render_template('index.html')

@app.route('/register',methods = ['GET','POST'])
def register():
    # return redirect(url_for("register.html"))
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        return render_template('register.html')

#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET','POST'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[1]
            add_data(identified_person)
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Validation',frame)
        if cv2.waitKey(1)==ord('y'):
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_data()    
    return render_template('index.html')


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['userName']
    newuserid = request.form['newuserid']
    password = request.form['userPassword']
    weight = request.form['userWeight']
    phonenumber = request.form['phonenumber']
    diseasename = request.form['diseasename']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_data()    
    return render_template('register.html') #changing the location 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)


