import os
from datetime import datetime
from time import time, sleep
import cv2
from PIL import Image

if __name__ == "__main__":
    complete_time = 24*60*60
    image_wait_period = 10
    
    folder = str(datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")
    target_folder = os.path.join("timelapse", folder)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
            
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        
    os.makedirs("timelapse", exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)
    
    start_time = time()
    img_number = 0
    while time() - start_time < complete_time:
        cycle_start_time = time()
        
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.save(os.path.join(target_folder, str(img_number).zfill(4) + ".jpeg"))
        img_number += 1
    
        wait_time = image_wait_period - (time() - cycle_start_time)
        sleep(wait_time)
    
    compile_command = f"ffmpeg -r 30 -i {target_folder}/%04d.jpeg -vcodec libx264 timelapse/{folder}.mp4"
    print(compile_command)
    ret = os.system(compile_command)
    # os.system("rm -r %s" % target_folder)
    
    cap.release()
    
# for 5 minute video:
# 9000 images
# for recording 24h:
# 1 omage/10 seconds