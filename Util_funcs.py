# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:21:24 2024

@author: orifl
"""
import numpy as np
from gtts import gTTS
import cv2

# Functions
def create_road_sidewalk_mask (img_shape = (384,640)):
    img_width = img_shape[1]
    img_height = img_shape[0]
    
    far_mask = np.zeros((img_height,img_width))   
    close_mask = np.zeros((img_height,img_width))   

    for x in range(img_width):
        for y in range(img_height):
            if   (y>1.84*x-1.35*img_height) and (y>(-1.84*x+1.72*img_height)):
                if (y>0.57*img_height) and (y<0.75*img_height):
                    far_mask[y,x] = 1
                if (y>0.75*img_height):
                    close_mask[y,x] = 1
    
    
    return [close_mask,far_mask]

def create_column_masks (img_shape = (384,640)):
    img_width = img_shape[1]
    
    right_mask =  np.zeros(img_shape)   
    middle_right_mask = np.zeros(img_shape)   
    center_mask = np.zeros(img_shape)   
    middle_left_mask = np.zeros(img_shape)   
    left_mask = np.zeros(img_shape)
    
    for x in range(img_width):
        if x>img_width*(4/5):
            right_mask[:,x] = 1
        
        if (x>img_width*(3/5)) and (x<img_width*(4/5)):
            middle_right_mask[:,x] = 1
        
        if (x>img_width*(2/5)) and (x<img_width*(3/5)):
            center_mask[:,x] = 1
            
        if (x>img_width*(1/5)) and (x<img_width*(2/5)):
            middle_left_mask[:,x] = 1
            
        if x<img_width*(1/5):
            left_mask[:,x] = 1
            
    return [right_mask, middle_right_mask, center_mask, middle_left_mask, left_mask]
    
    
def create_sounds(texts,sound_path = 'D:/blindAssist/Sound/'):
    for text in texts:
        tts = gTTS(text)
        tts.save(sound_path+'{}.mp3'.format(text))
        
def modify_video_FPS_RES(input_video_path,new_FPS,new_width = 640,new_height = 384):
    #Modify video's FPS and/or Resolution
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if new_width == None:
        new_width = width
    if new_width == None:
        new_height = height
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = fps // new_FPS

    output_video_path = input_video_path.split('.')[0] + '{:.0f}'.format(fps//frame_interval) + '_FPS.' + input_video_path.split('.')[1]
    out = cv2.VideoWriter(output_video_path, fourcc, new_FPS, (new_width, new_height))
    
    frame_count = 0
    while True:
        # Read a frame
        ret, frame = cap.read()
    
        if not ret:
            break
    
        # Write frame to output video every 'frame_interval' frames
        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)
    
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
#%%
# new_FPS = 2
# modify_video_FPS_RES('D:/blindAssist/Videos/Walking_in_Beer_Sheva_Israel_Masada_Crossing_Roads.mp4',new_FPS)


