# Emotion Detection and autocrop Face
copy each charactor face picture .jpg to this folder example bella.jpg, dej.jpg
copy video file contain the charactor we selected to this folder

change name of charactor in auto_crop_from_video.py 
    suggest to ctrl_f and replace all
create folder of each charactor in ./face_database/ example ./face_database/bella
    face_detabase
    -- bella
    -- dej
    

run by : python auto_crop_from_video.py {videoname} ex: python auto_crop_from_video test3.mp4

consider the output image from last program in each emotion and copy image to images/{emotions} 

#Train : python train.py

#run : python model.py videofile
