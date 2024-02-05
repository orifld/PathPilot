# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:00:49 2024

@author: orifl
"""

from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
import pygame
import torch
import time
import Util_funcs
import argparse



#Load YOLO V8 for segments prediction
model_source = 'Models/YOLO_V8/weights/best.pt' #medium
model = YOLO(model_source)
sound_dir = 'Sound/'

#Load MiDaS to predict depth in images
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cpu')
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

#Voice over
pygame.init()
pygame.mixer.init()

#%% Arguments parser
parser = argparse.ArgumentParser('PathPilot')
parser.add_argument('--video_source_path', type=str, default = 'Videos/GH012163_640_2FPS.mp4', help='Input video path. Example: Videos/GH012163_640_2FPS.mp4')
args = parser.parse_args()



#%% Parameters

permute = [2, 1, 0] # For BGR to RGB transformation
frame_counter = 0
frame_with_sound = 0
min_frames_sound_interval = 2     #Eliminate reptitive voice over for same scenarios
clss_thr = 0.4                    #Do not consider obstacles with lower score
img_shape = (384,640)
classes = ['bench', 'bollard', 'bush', 'cabinet', 'car', 'car-barrier', 'chair', 'dirt', 'fence', 'person', 'pole', 'road', 'road-crossing', 'sidewalk', 'stairs', 'trash-bin', 'tree', 'wall']
# video_source_path = 'Videos/GH012163_640_2FPS.mp4'   

#%% Main function



#Predict segments
frames = model.predict(source=args.video_source_path, save=True, save_txt=False, stream=True, boxes=False, show=False)
col_masks = Util_funcs.create_column_masks()
road_sidewalk_masks = Util_funcs.create_road_sidewalk_mask()

start_time = time.time()

for frame in frames:
    
    #Predict depth image
    imgbatch = transform(frame.orig_img).to('cpu')
    
    with torch.no_grad():
        depth_pred_low = midas(imgbatch)
        depth_pred  = torch.nn.functional.interpolate(depth_pred_low.unsqueeze(1),size=frame.masks.data[0].shape[:2],mode="bicubic",align_corners=False).squeeze().cpu().numpy()
    
    #Plot segments, depth and original images
    fig, axs = plt.subplots(nrows = 3,ncols = 2, figsize = (17.5,20))
    gs = axs[0, 0].get_gridspec()
    
    ax1 = plt.subplot(3,2,5)
    ax1.axis('off')
    ax2 = plt.subplot(3,2,6)
    ax2.axis('off')
    ax3 = fig.add_subplot(gs[:2, :])
    ax3.axis('off')

    for ax in axs[:2,:]:
        ax[0].axis('off')
        ax[1].axis('off')

    font_style = {'fontname': 'Arial', 'fontsize': 30}

    ax1.set_title('Segments', **font_style)
    ax1.imshow(frame.plot(boxes=False)[:,:,permute])
    ax2.set_title('Depth Image', **font_style)
    ax2.imshow(depth_pred)
    ax3.set_title('Original Image', **font_style)
    ax3.imshow(frame.orig_img[:,:,permute])
    
    plt.subplots_adjust(hspace=0.25, wspace=0.15, top = 0.9, bottom = 0.1, left = 0, right = 1)
    plt.pause(0.000001)

    depth_pred_norm = depth_pred / 1000
    
    #Check if traversing road or sidewalk
    sidewalk_road_mask_prod_close = np.sum(frame.masks.data.numpy() * road_sidewalk_masks[0], axis=(1,2))/np.sum(road_sidewalk_masks[0])
    sidewalk_road_mask_prod_far = np.sum(frame.masks.data.numpy() * road_sidewalk_masks[1], axis = (1,2))/np.sum(road_sidewalk_masks[1])
    
    sidewalk_road_close_class = classes[int(frame.boxes[np.argmax(sidewalk_road_mask_prod_close)].data[:,5][0].numpy())]
    sidewalk_road_far_class = classes[int(frame.boxes[np.argmax(sidewalk_road_mask_prod_far)].data[:,5][0].numpy())]
    
    if frame_counter - frame_with_sound > min_frames_sound_interval:
        #Sidewalk -> Road
        if (sidewalk_road_far_class == 'road') and (sidewalk_road_close_class == 'sidewalk'):
            sound_file = sound_dir + 'You are about to leave the sidewalk and enter the road.mp3'
            pygame.mixer.Channel(0).queue(pygame.mixer.Sound(sound_file))
            frame_with_sound = frame_counter
        #Road -> Sidewalk     
        elif (sidewalk_road_far_class == 'sidewalk') and (sidewalk_road_close_class == 'road'):
            sound_file = sound_dir + 'You are about to leave the road and enter the sidewalk.mp3'
            pygame.mixer.Channel(0).queue(pygame.mixer.Sound(sound_file))
            frame_with_sound = frame_counter

        
    # Check if obstacles are in front / sides of the way combining segments and depth maps
    masks_sum = np.sum(frame.masks.data.numpy(),axis=(1,2))
    mask_depth_prod =  np.sum(depth_pred_norm * frame.masks.data.numpy(),axis=(1,2))
    mask_depth_prod_norm = mask_depth_prod / masks_sum
    
    max_val_args = np.argsort(mask_depth_prod_norm)[-5:]
    max_val = np.sort(mask_depth_prod_norm)[-5:]
    
    #Find most dominant obstable
    top_class = None
    top_class_arg = None
    for c in range(len(max_val)-1,-1,-1):
        clss = classes[int(frame.boxes[max_val_args[c]].data[:,5][0].numpy())]
        if  (clss != 'sidewalk') and (clss != 'road') and ((clss != 'dirt')):      
            if max_val[c] > clss_thr:
                top_class = clss
                top_class_arg = max_val_args[c]
                print('Top class: {}, Value: {}'.format(top_class,max_val[c]))
                break
    
    #Eliminate reptitive voiceovers for same scenarios
    if frame_counter - frame_with_sound > min_frames_sound_interval:

        if top_class != None:
            
            #Estimate orientation right/center/left of the most dominant obstacle
            orientation_arg = np.argmax(np.sum(np.array(frame.masks[top_class_arg].data.numpy()[0] * \
                                      col_masks) \
                                      , axis=(1,2)))

            # Obstacle in the middle
            if orientation_arg == 2:
                sound_file = sound_dir + top_class + ' ahead.mp3'
                pygame.mixer.Channel(0).queue(pygame.mixer.Sound(sound_file))
                
                #Find proper alternative, whether left or right according to where sidewalk is more present
                sidewalk_mask = np.zeros(img_shape)
                for f in range(len(frame.boxes)):
                    if classes[int(frame.boxes[f].data[:,5][0].numpy())] == 'sidewalk':
                        sidewalk_mask += frame.masks.data[f].numpy()
                
                sidewalk_col_masks_sum_prod = np.sum(sidewalk_mask * col_masks, axis=(1,2))
                if sidewalk_col_masks_sum_prod[1] >= sidewalk_col_masks_sum_prod[3]:
                    sound_file = sound_dir + 'bypass it from the right.mp3'
                else:
                    sound_file = sound_dir + 'bypass it from the left.mp3'

                pygame.mixer.Channel(0).queue(pygame.mixer.Sound(sound_file))
                frame_with_sound = frame_counter
            
            # Obstacle to the right
            elif orientation_arg == 1:
                sound_file = sound_dir + top_class + ', to your right.mp3'
                pygame.mixer.Channel(0).queue(pygame.mixer.Sound(sound_file))
                frame_with_sound = frame_counter
            
            # Obstacle to the left
            elif orientation_arg == 3:
                sound_file = sound_dir + top_class + ', to your left.mp3'
                pygame.mixer.Channel(0).queue(pygame.mixer.Sound(sound_file))
                frame_with_sound = frame_counter

    print('Time elapsed: {}'.format( time.time() - start_time))
    print('\n')
    frame_counter += 1
