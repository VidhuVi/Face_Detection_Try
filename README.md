#REQUIREMENTS:

    .CMAKE
    .OPENCV
    .FACE-RECOGNITION (FROM OPENCV)

#ADD:

    .After downloading the code, Add a folder "known_faces" with subfolders which have name of person as subfolder name. 
      eg: "person1" --> subfolder contains all the photos of person1 named as 1.imgtype, 2.imgtype, etc..

#RUNNING:

    .Either run "main.py" or "face_data.py".
    .If running "face_data.py" then run "access_control.py" next.
  
#NOTE:

    .Increasing the number of training data lowers the chances of error.
    .This is a very basic implementation and may contin many errors.
