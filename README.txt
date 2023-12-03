# Overview
This is a basic Flask web application which employs face recognition to keep track 
of student attendance. Users can make use of the app to record students, take attendance, and examine the day's attendance list.

## Features
Capture and preserve face data of students, as well as their names, IDs, and classes.
Take Attendance: During a class session, use facial recognition to take attendance.
View Attendance: Show a list of students that attended class for the day.

## Requirements
Make sure you have the following installed:

Python >=3.6
Flask 
OpenCV >= 3.4
Pandas 
Scikit-learn

## Getting started

- install dependancies using pip install <dependancy name>
- clone this github folder using git 
- open the folder using your API (preferably VSCode)
- run the app using 'python flaskapp.py'
- navigate to http://127.0.0.1:5000/ in your chosen browser

## Notes

- the training of the model was done OpenCV and KNearest Neighbors
- the trained images and names were packaged into pickled(.pkl) files
- LINK TO VIDE: https://youtu.be/MXPYaM-XLPE
