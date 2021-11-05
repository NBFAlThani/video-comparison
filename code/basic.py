import numpy as np
import cv2
from PIL import Image
import imagehash
import time

# this is a second
##### Assign the Videos  #####
cap1 = cv2.VideoCapture('../data/vid1.mp4')
cap2 = cv2.VideoCapture('../data/vid2.mp4')



################################### FIRST VIDEO ###################################
dhash_1_list=[] #Array for Dhash
phash_1_list=[] #Array for Phash
frame1_list=[] #Array that holds the frames
counter=0 #To count the current frame number
while(cap1.isOpened()): #While video is playing 
    ret1, frame1 = cap1.read()
    if ret1:
        if counter%5==0: #Checking every five frames
            frame1_list.append(frame1) #Append the MATRIX (Frame) to the Array
            frame1 = Image.fromarray(frame1) #Format the matrix to an object (IMG)
            d_hash1 = imagehash.dhash(frame1) #Calculating Dhash
            p_hash1 = imagehash.phash(frame1) #Calculating Phash
            dhash_1_list.append(d_hash1)
            phash_1_list.append(p_hash1)
        counter+=1
    else:
        break #If ret = 0 then there is no frame so, breake the loop 

################################### SECOND VIDEO ###################################
dhash_2_list=[]
phash_2_list=[]
frame2_list=[]
counter=0
while(cap2.isOpened()): #While video is playing 
    ret2, frame2 = cap2.read()
    if ret2:
        if counter%5==0: #Checking every five frames
            frame2_list.append(frame2) #Append the MATRIX (Frame) to the Array
            frame2 = Image.fromarray(frame2) #Format the matrix to an object (IMG)
            d_hash2 = imagehash.dhash(frame2) #Calculating Dhash
            dhash_2_list.append(d_hash2) 
            phash_2_list.append(imagehash.phash(frame2)) #Calculating and appending Phash
        counter+=1
    else:
        break #If ret = 0 then there is no frame so, breake the loop 


################################### COMPARING P & D HASHES ###################################
matched_indices=[-1 for i in range(len(dhash_1_list))]
hash_diff=[999 for i in range(len(dhash_1_list))]

########################## Calculating D HASHE ##########################

for i in range(len(dhash_1_list)):
    min_hash_dist=9999
    for j in range(len(dhash_2_list)):
        diff_hash_val=abs(dhash_1_list[i]-dhash_2_list[j])
        if diff_hash_val<min_hash_dist:
            min_hash_dist=diff_hash_val
            matched_indices[i]=j
            hash_diff[i]=diff_hash_val


########################## Calculating P HASHE & Compare it ##########################
########################## to D HASHE for The Same Frame ##########################
for i in range(len(phash_1_list)):

    for j in range(len(phash_2_list)):
        diff_hash_val=abs(phash_1_list[i]-phash_2_list[j])
        if diff_hash_val<hash_diff[i]:
            hash_diff[i]=diff_hash_val
            matched_indices[i]=j
           

print("The matched array is ",matched_indices)
print("THe hash differences are ",hash_diff)


################################### GETTING THE SAME FRAMES ###################################

final_match_indices=[-1 for i in range(len(matched_indices))]

threshold_match=14 #Choosen threashold
for i in range(len(matched_indices)):
    if hash_diff[i]<threshold_match:
        final_match_indices[i]=matched_indices[i]

print("Final matches are ",final_match_indices)


################################### CREATING THE OUTPUT VIDEO ###################################


#New height and width
target_h=300
target_w=300


out = cv2.VideoWriter('../Data/out.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (target_w*2,target_h)) #Creating output video, Width*2 because we're combining the 2 frames

# Getting the matched frames
for i in range(len(final_match_indices)):
    if final_match_indices[i]!=-1: #The "i" indicates the frame number from the first video, while the value is the frame of the second video
        print("Taking frame ",final_match_indices[i])
        # resize
        f1=cv2.resize(frame1_list[i],(target_w,target_h),interpolation = cv2.INTER_AREA)
        f2=cv2.resize(frame2_list[final_match_indices[i]],(target_w,target_h),interpolation = cv2.INTER_AREA)
        combo_img=np.concatenate((f1, f2), axis=1) #To combine both frames
        out.write(combo_img)
out.release()

