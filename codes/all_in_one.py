#!/usr/bin/env python
# coding: utf-8

# # 01-Keyframe_Extraction

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://medium.com/@myworldsharma.jay/key-frame-extraction-from-video-9445564eb8ed
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import os
import math


# In[2]:


def create_folders(file_name_only_no_extension):    
    try:
        intermediate_location="../intermediate/"
        if not os.path.isdir(intermediate_location):
            os.mkdir(intermediate_location)
        resized_location=intermediate_location+"resized/"
        if not os.path.isdir(resized_location):
            os.mkdir(resized_location)
        
        

            
        data_dir=intermediate_location+"data/"
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        data_dir=data_dir+file_name_only_no_extension+"/"  
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)    
            

        
            
        kframes_location=intermediate_location+"keyframes/"
        if not os.path.exists(kframes_location):
            os.makedirs(kframes_location)
        kframes_location+=file_name_only_no_extension+"/"
        if not os.path.exists(kframes_location):
            os.makedirs(kframes_location)        
    except OSError:
        print("Error cant make directories")
        
    return resized_location,data_dir,kframes_location
        
def resize_video(source_video_location,destination):
    
    new_dim=(32,24)

    cap = cv2.VideoCapture(source_video_location)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(destination, fourcc, 25.0, new_dim)


    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame,new_dim,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(b)
        else:
            break

    cap.release()
    out.release()
#     cv2.destroyAllWindows()    


    
def extract_all_frames_histogram(video_location,all_frames_location):

    cap = cv2.VideoCapture(video_location)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CV_CAP_PROP_FPS): {0}".format(fps))
    frame_list_norm = []
    hist_list=[]
    print("Start")


    cframe = 0
    while(True):

        ret, frame= cap.read()
        if not ret:
            break
        frame_arr=np.array(frame)
        frame_arr_norm=(frame_arr-0)/255
        frame_list_norm.append(frame_arr_norm)
        hist = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
        hist = cv2.normalize(hist,None).flatten()
        hist_list.append(hist)
        name = all_frames_location +str(cframe) + '.jpg'
        cv2.imwrite(name,frame)

        cframe += 1

    cap.release()
#     cv2.destroyAllWindows()
    print("Frame extraction complete")     
    return hist_list



# whenever there is a 1
# make k frames before and after it also 1

def make_one(list_,center,window,to_keep_new):
#     print("center is at",center,"from",center-window,"to",center+window)
    for i in range(center-window,center+window):
        if i>0 and i<len(list_):
            to_keep_new[i]=1
    return to_keep_new



def generate_keep_list(hist_list):


    hist_list=np.array(hist_list)
#     print(hist_list.shape)

    diff_hist_list=[]

    for i in range(hist_list.shape[0]-1):
        hist1=hist_list[i]
        hist2=hist_list[i+1]
        d = abs(cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT))
#         print(d)
        diff_hist_list.append(d)

    diff_hist_list=np.array(diff_hist_list)
#     print(diff_hist_list.shape)
    sd=math.sqrt(np.var(diff_hist_list))



    differences=[]
    left_image_index=0
    for right_image_index in range(1,diff_hist_list.shape[0]):
        # to check how much right image is similar to the left image
        d1=diff_hist_list[left_image_index]
        d2=diff_hist_list[right_image_index]
        diff_diff = abs(d1-d2)
        differences.append(diff_diff)
        left_image_index=right_image_index


    sd_diff=math.sqrt(np.var(differences))    


    to_keep=[1]

    left_image_index=0
    for right_image_index in range(1,diff_hist_list.shape[0]):
        # to check how much right image is similar to the left image
        d1=diff_hist_list[left_image_index]
        d2=diff_hist_list[right_image_index]
        diff_diff = abs(d1-d2)
#         print(left_image_index,right_image_index,diff_diff,sd_diff)
        if diff_diff>sd_diff:
            to_keep.append(1)
        else:
            to_keep.append(0)
        left_image_index=right_image_index



    wind=1
    to_keep_new=[0 for i in range(len(to_keep))]
    for i in range(len(to_keep)):
        if to_keep[i]==1:
#             print("1 at ",i,to_keep[i])
            to_keep_new=make_one(to_keep,i,wind,to_keep_new)                

    return to_keep_new

def store_keyframes_using_keep_list(video_location,kframes_location,keep_list):
    cap = cv2.VideoCapture(video_location)    
    fps = cap.get(cv2.CAP_PROP_FPS)    
    cframe = 0
    while(True):
        ret, frame= cap.read()
        if not ret:
            break        
        if keep_list[cframe]==1:
            name = kframes_location + str(cframe) + '.jpg'
#             print("creating" +name)
            cv2.imwrite(name,frame)
        cframe += 1
        if cframe>len(keep_list)-1:
            break

    cap.release()
#     cv2.destroyAllWindows()
    print("Data creation complete")          


def make_keyframes(source_video_location):
    
    file_name_only=source_video_location.split("/")[-1]
    file_name_only_no_extension=file_name_only.split(".")[0]

    print(file_name_only,file_name_only_no_extension)
    # create all the requisite folders
    resized_location,all_frames_location,kframes_location=create_folders(file_name_only_no_extension)
    
    # can we put a check to see if the kframes_location already has files in it
    if len(os.listdir(kframes_location))>0:
        print("Keyframes already made for this file")
        return kframes_location
    
    print(resized_location,all_frames_location,kframes_location)
    
    resized_location_destination=resized_location+file_name_only
    
    # resize the video to be same size
    resize_video(source_video_location,resized_location_destination)
    
    # extract all frames, store in data folder and generate histogram
    hist_list=extract_all_frames_histogram(resized_location_destination,all_frames_location)
    
    
    to_keep_new=generate_keep_list(hist_list)
    
    store_keyframes_using_keep_list(resized_location_destination,kframes_location,to_keep_new)
    print("All keyframes at",kframes_location)
    return kframes_location


# ### two references

# In[3]:


path="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/references/"
file_names=os.listdir(path)
klocs=[]
for file_name in file_names:
    file_name=path+file_name
    kloc=make_keyframes(file_name)
    klocs.append(kloc)


    
path="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/setC/positive-c.zip/"
file_names=os.listdir(path)
num_files=len(file_names)
klocs_arch=[]
count=0
for file_name in file_names:
    count+=1
    print(count,"/",num_files)
    file_name=path+file_name
    kloc=make_keyframes(file_name)
    klocs_arch.append(kloc)


# In[10]:


print(klocs_arch)


# In[ ]:





# # 02-CalculateAndStoreHashes

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre
import numpy as np
import pandas as pd
import csv
import datetime
# get_ipython().run_line_magic('matplotlib', 'inline')
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
from matplotlib import pyplot as plt
import pickle

import imagehash
from PIL import Image


# In[2]:


reference_locations=[]
path_to_ref="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/references/"
path_to_ref_kf="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/intermediate/keyframes/"
for f in os.listdir(path_to_ref):
    f=f.split(".")[0]
    reference_locations.append(path_to_ref_kf+f)
    
print(reference_locations,len(reference_locations))


# In[3]:


intermediate_location="../intermediate/"
if not os.path.isdir(intermediate_location):
    os.mkdir(intermediate_location)
hash_location=intermediate_location+"hash_values/"
if not os.path.isdir(hash_location):
    os.mkdir(resized_location)

hash_location=hash_location+"phash/"
if not os.path.isdir(hash_location):
    os.mkdir(resized_location)
    
dic_location=hash_location+"dic_vals.p"    

if not os.path.isfile(dic_location):
    dic_hash_vals={}
else:
    dic_hash_vals = pickle.load( open( dic_location, "rb" ) )








# In[4]:


len(dic_hash_vals.keys())


# In[5]:



for reference_location_path in reference_locations:
    ref_file_frames_list=os.listdir(reference_location_path)
    file_name=reference_location_path.split("/")[-1]
    if file_name in dic_hash_vals:
        print("hash already generated for ",file_name)
        continue
    dic_hash_vals[file_name]={}
#     print(reference_location_path,file_name,ref_file_frames_list)
    for frame_name in ref_file_frames_list:
        f=os.path.join(reference_location_path,frame_name)
        hash_val=imagehash.phash(Image.open(f))
#         print(f,hash_val)
        dic_hash_vals[file_name][frame_name]=hash_val


# In[6]:


dic_hash_vals.keys()


# In[7]:


# dic_hash_vals


# In[8]:


archive_locations=[]
path_to_archive="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/setC/positive-c.zip/"
path_to_archive_kf="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/intermediate/keyframes/"
for f in os.listdir(path_to_archive):
    f=f.split(".")[0]

    archive_locations.append(path_to_archive_kf+f)
    
print(archive_locations,len(archive_locations))    


# In[9]:


for archive_location_path in archive_locations:
    arch_file_frames_list=os.listdir(archive_location_path)
    file_name=archive_location_path.split("/")[-1]
    if file_name in dic_hash_vals:
        print("hash already generated for ",file_name)
        continue    
    dic_hash_vals[file_name]={}
#     print(archive_location_path,file_name,arch_file_frames_list)
    print(archive_location_path,file_name)
    for frame_name in arch_file_frames_list:
        f=os.path.join(archive_location_path,frame_name)
        hash_val=imagehash.phash(Image.open(f))
#         print(f,hash_val)
        dic_hash_vals[file_name][frame_name]=hash_val


# In[10]:


# intermediate_location="../intermediate/"
# if not os.path.isdir(intermediate_location):
#     os.mkdir(intermediate_location)
# hash_location=intermediate_location+"hash_values/"
# if not os.path.isdir(hash_location):
#     os.mkdir(resized_location)

# hash_location=hash_location+"phash/"
# if not os.path.isdir(hash_location):
#     os.mkdir(resized_location)
    
# dic_location=hash_location+"dic_vals.p"    
pickle.dump( dic_hash_vals, open( dic_location, "wb" ) )


# In[11]:


len(dic_hash_vals.keys())


# In[ ]:





# # 03-KF_Similarities_UsingDic

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# #https://betterprogramming.pub/how-to-measure-image-similarities-in-python-12f1cb2b7281

# In[ ]:





# In[1]:


import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre
import numpy as np
import pandas as pd
import csv
import datetime
# get_ipython().run_line_magic('matplotlib', 'inline')
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
from matplotlib import pyplot as plt

import imagehash
from PIL import Image


import pickle


# In[ ]:





# In[2]:


def get_sorted_dic_files(location):
    my_vid_frame_files=os.listdir(location)
    sort_dic_my_vid_frame_files={}
    for my_file in my_vid_frame_files:
        file_num=my_file.split(".")[0]
        sort_dic_my_vid_frame_files[int(file_num)]=my_file

    from collections import OrderedDict


    sort_dic_my_vid_frame_files = OrderedDict(sorted(sort_dic_my_vid_frame_files.items()))
    return sort_dic_my_vid_frame_files


def compare_frames(f1,f2,strategies):
    differences=[]
    for strategy in strategies:
        if strategy=="phash":
            diff=imagehash.phash(Image.open(f1))-imagehash.phash(Image.open(f2))
            differences.append(diff)
        elif strategy=="dhash":
            diff=imagehash.dhash(Image.open(f1))-imagehash.phash(Image.open(f2))
            differences.append(diff)
        elif strategy=="colorhash":
            diff=imagehash.colorhash(Image.open(f1))-imagehash.colorhash(Image.open(f2))
            differences.append(diff)            
        elif strategy=="ssim":
            f1_img = cv2.imread(f1)
            f2_img = cv2.imread(f2)
            diff= ssim(f1_img, f2_img)
            differences.append(diff)
        elif strategy=="sre":
            f1_img = cv2.imread(f1)
            f2_img = cv2.imread(f2)
            diff= sre(f1_img, f2_img)
            differences.append(diff)
        elif strategy=="rmse":
            f1_img = cv2.imread(f1)
            f2_img = cv2.imread(f2)
            diff= rmse(f1_img, f2_img)
            differences.append(diff)
    return differences

def store_in_diff_dic(diff_dic,strategies,diff_scores,arch_kframe_name):
    for i in range(len(strategies)):
        strategy=strategies[i]
        diff_score=diff_scores[i]
        if strategy not in diff_dic:
            diff_dic[strategy]={}
        if arch_kframe_name not in diff_dic[strategy]:
            diff_dic[strategy][arch_kframe_name]=diff_score
    return diff_dic


def get_best_diff(diff_dic,strategies,mappings):
    best_diff_dic={}
    
    for strategy in strategies:   
        best_diff_dic[strategy]={}
        dictn=diff_dic[strategy]
        if strategy in mappings["min"]:
            best_arch_kframe=min(dictn, key=dictn.get)
            best_val=min(list(diff_dic[strategy].values()))
        elif strategy in mappings["max"]:
            best_arch_kframe=max(dictn, key=dictn.get)
            best_val=max(list(diff_dic[strategy].values()))
        best_diff_dic[strategy][best_arch_kframe]=best_val
#     print("best",best_diff_dic)
    return best_diff_dic
        
def store_in_result_dic(all_results,least_diff_dic,strategies,ref_kf_path_name,archive_kf_location):
    for strategy in strategies:
        score=list(least_diff_dic[strategy].values())[0]
        best_match_archive_kf_fname=list(least_diff_dic[strategy].keys())[0]
        all_results["ref_kf_path_name"].append(ref_kf_path_name)
        all_results["arch_kf_path"].append(archive_kf_location)
        all_results["arch_kf_name"].append(best_match_archive_kf_fname)        
        all_results["strategy"].append(strategy)
        all_results["score"].append(score)    
    return all_results
        
        


# In[3]:


strategies=["phash"]
mappings={}
mappings["min"]=["phash","dhash","rmse"]
mappings["max"]=["ssim","sre"]


# In[4]:


hash_locations="../intermediate/hash_values/phash/dic_vals.p"
dic_hash_vals = pickle.load( open( hash_locations, "rb" ) )


# In[5]:


reference_locations=[]
path_to_ref="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/references/"
path_to_ref_kf="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/intermediate/keyframes/"
for f in os.listdir(path_to_ref):
    f=f.split(".")[0]
    reference_locations.append(path_to_ref_kf+f)
    
print(reference_locations,len(reference_locations))


# In[6]:


archive_locations=[]
path_to_archive="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/setC/positive-c.zip/"
path_to_archive_kf="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/intermediate/keyframes/"
for f in os.listdir(path_to_archive):
    f=f.split(".")[0]

    archive_locations.append(path_to_archive_kf+f)
    
print(archive_locations,len(archive_locations))    


# In[7]:


dic_hash_vals.keys()


# In[8]:





def compare_kframes(list_references_kf_locations,list_archives_kf_locations,strategies,mappings,res_location,dic_hash_vals):
    all_results={}
    all_results["ref_kf_path_name"]=[]
    all_results["arch_kf_path"]=[]
    all_results["arch_kf_name"]=[]    
    all_results["strategy"]=[]
    all_results["score"]=[]    
    ref_count=0
    total_ref_count=len(list_references_kf_locations)
    for references_kf_location in list_references_kf_locations:        
        ref_count+=1        
        ref_file_name=references_kf_location.split("/")[-1]
        e = datetime.datetime.now()
        print(e,ref_file_name,ref_count,"/",total_ref_count)
        sorted_dict_reference_kf=get_sorted_dic_files(references_kf_location)
        ref_hash=dic_hash_vals[ref_file_name]
        
        count=0
        for ref_kf_num, ref_kf_file_name in sorted_dict_reference_kf.items():  
            ref_kf_hash_val=ref_hash[ref_kf_file_name]
#             print("start",ref_kf_num,ref_kf_file_name,ref_kf_hash_val)
            count+=1
            percent_complete=int(100*count/len(list(sorted_dict_reference_kf.keys())))
            if percent_complete%10==0:
                print(percent_complete,"% complete at ",datetime.datetime.now())
#             print("\t",ref_kf_num, ref_kf_file_name)
            f1=os.path.join(references_kf_location,ref_kf_file_name)
            for archive_kf_location in list_archives_kf_locations:
                arch_file_name=archive_kf_location.split("/")[-1]
                arch_hash=dic_hash_vals[arch_file_name]
                sorted_dict_arch_kf=get_sorted_dic_files(archive_kf_location)
                diff_dic={}
                # above will contain scores for each frame of reference key frames
                for arch_kf_num,arch_kf_file_name in sorted_dict_arch_kf.items():   
                    arch_kf_hash_val=arch_hash[arch_kf_file_name]
                    f2=os.path.join(archive_kf_location,arch_kf_file_name)
#                     print(f1,f2)
#                     diff_scores=compare_frames(f1,f2,strategies)
                    diff_scores=[ref_kf_hash_val-arch_kf_hash_val]
                    diff_dic=store_in_diff_dic(diff_dic,strategies,diff_scores,arch_kf_file_name)
#                 print(diff_dic)
                # finished with one set of key frames
#                 print("\t ",diff_dic)
                best_diff_dic=get_best_diff(diff_dic,strategies,mappings)
#                 print("\t\t",best_diff_dic)
                all_results=store_in_result_dic(all_results,best_diff_dic,strategies,f1,archive_kf_location)
                df=pd.DataFrame(all_results)
                df.to_csv(res_location,index=False)
               
    
    return all_results
                
    


# In[ ]:



# reference_locations=[
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_202101300630",
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_202101300810"    
# ]

# archive_locations=[
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_20210130063213_C2",
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_20210130081629_C3",    
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_20210130115216_C3"
# ]
    
    
try:
    if not os.path.isdir("../results"):
        os.mkdir("../results")
except OSError:
        print("Error cant make directories")    

location="../results/res.csv"
all_results=compare_kframes(reference_locations,archive_locations,strategies,mappings,location,dic_hash_vals)


# In[ ]:


df=pd.DataFrame(all_results)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


try:
    if not os.path.isdir("../results"):
        os.mkdir("../results")
    df.to_csv("../results/res_all.csv",index=False)
except OSError:
        print("Error cant make directories")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 04-accuracyCalc

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score


# In[41]:


'''
in order to compare accuracy of our algorithm with that of
the ground truth, we need two inputs

input1=list of reference videos
input2=list of archive videos


'''


# In[42]:


reference_locations=[]
path_to_ref="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/references/"
path_to_ref_kf="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/intermediate/keyframes/"
for f in os.listdir(path_to_ref):
    f=f.split(".")[0]
    reference_locations.append(path_to_ref_kf+f)
    
print(reference_locations,len(reference_locations))


# In[43]:


archive_locations=[]
path_to_archive="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/data/setC/positive-c.zip/"
path_to_archive_kf="/ddn/gfxhome/asislam25/projects/other_misc_projects/video_copy/intermediate/keyframes/"
for f in os.listdir(path_to_archive):
    f=f.split(".")[0]

    archive_locations.append(path_to_archive_kf+f)
    
print(archive_locations,len(archive_locations))    


# In[5]:


# # below are the two lists as inputs
# reference_locations=[
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_202101300630",
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_202101300810"    
# ]

# archive_locations=[
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_20210130063213_C2",
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_20210130081629_C3",    
#     "/Users/ashhadulislam/projects/other_misc/video_copy/intermediate/keyframes/c01_20210130115216_C3"
# ]


# In[44]:


reference_location_video_names_only=[]
for full_path in reference_locations:
    just_name=full_path.split("/")[-1]
    reference_location_video_names_only.append(just_name)


# In[45]:


archive_location_video_names_only=[]

for full_path in archive_locations:
    # remove everything before the last /
    just_name=full_path.split("/")[-1]
    
    # get till the last underscore
    just_name=just_name[:-3]
    archive_location_video_names_only.append(just_name)


# In[46]:


print(reference_location_video_names_only,archive_location_video_names_only)
archive_location_video_names_only.sort()


# In[47]:


len(reference_location_video_names_only),len(archive_location_video_names_only)


# In[48]:


df=pd.read_csv("../data/groundtruth.csv",sep=";")


# In[49]:


df.head()


# In[50]:


df["Reference_Video_name_only"]=[nm.split(".")[0] for nm in df["Reference_Video"]]


# In[51]:


df["Positive_Video_name_only"]=[nm.split(".")[0] for nm in df["Positive_Video"]]


# In[52]:


df.head()


# In[53]:


df=df[["Reference_Video_name_only","Positive_Video_name_only"]]


# In[54]:


df.head()


# In[55]:


df.shape


# In[56]:


# apply filter for reference videos
df_new=[]
for a_vid in reference_location_video_names_only:
    df_chosen=df[df["Reference_Video_name_only"]==a_vid]
    print(a_vid,df_chosen.shape)
    df_new.append(df_chosen)


# In[57]:


df_new=pd.concat(df_new)


# In[58]:


df_new.shape


# In[59]:


my_ground_truths={}
my_ground_truths["reference_video"]=[]
my_ground_truths["score_array"]=[]
my_ground_truths["archive_videos"]=[]

for a_vid in reference_location_video_names_only:
    df_chosen=df_new[df_new["Reference_Video_name_only"]==a_vid]
    print(df_chosen.shape)
    uniq_archives=df_chosen["Positive_Video_name_only"].unique()
    print(uniq_archives)
    scores=[]
    for arch_vid in archive_location_video_names_only:
        if arch_vid in uniq_archives:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    my_ground_truths["reference_video"].append(a_vid)
    my_ground_truths["score_array"].append(scores)
    my_ground_truths["archive_videos"].append(archive_location_video_names_only)


# In[60]:


df_gtruth=pd.DataFrame(my_ground_truths)
print(df_gtruth.shape)


# In[61]:


df_gtruth.head()


# In[ ]:





# In[62]:


df_gtruth.to_csv("../results/computer_says.csv",index=False)


# ## now we need to process our output

# In[63]:


# choose a strategy and threshold
strategy="phash"
threshold=10


# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


df_res_us=pd.read_csv("../results/res_all.csv")
print(df_res_us.shape)
df_res_us=df_res_us[df_res_us["strategy"]==strategy]
df_res_us=df_res_us[df_res_us["score"]<=threshold]


# In[65]:


df_res_us.shape


# In[66]:


df_res_us.head()


# In[67]:


df_res_us["ref_vid_name_only"]=[nm.split("/")[-2] for nm in df_res_us["ref_kf_path_name"]]


# In[68]:


df_res_us["arch_vid_name_only"]=[full_path.split("/")[-1][:-3] for full_path in df_res_us["arch_kf_path"]]


# In[69]:


df_res_us=df_res_us[["ref_vid_name_only","arch_vid_name_only"]]


# In[70]:


df_res_us.head()


# In[71]:


# now to calculate the scores
my_calculations={}
my_calculations["reference_video"]=[]
my_calculations["score_array"]=[]
my_calculations["archive_videos"]=[]

for a_vid in reference_location_video_names_only:
    df_chosen=df_res_us[df_res_us["ref_vid_name_only"]==a_vid]
    print(df_chosen.shape)
    uniq_archives=df_chosen["arch_vid_name_only"].unique()
    print(uniq_archives)
    scores=[]
    for arch_vid in archive_location_video_names_only:
        if arch_vid in uniq_archives:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    my_calculations["reference_video"].append(a_vid)
    my_calculations["score_array"].append(scores)
    my_calculations["archive_videos"].append(archive_location_video_names_only)


# In[72]:


df_my_res=pd.DataFrame(my_calculations)


# In[73]:


df_my_res.head()


# In[74]:


df_my_res.to_csv("../results/our_calcul.csv",index=False)


# ### Compare and calculate the scores

# In[75]:


df_given=pd.read_csv("../results/computer_says.csv")
df_calculated=pd.read_csv("../results/our_calcul.csv")


# In[76]:


reference_vids=list(df_given.reference_video)
avg_acc=0
for vid in reference_vids:
    df_given_vid=df_given[df_given["reference_video"]==vid]
    df_calculated_vid=df_calculated[df_calculated["reference_video"]==vid]
#     print(df_given_vid.shape,df_calculated_vid.shape)

    comp_array=list(df_given_vid["score_array"])[0].split(",")
    comp_array=comp_array[1:-2]    
    comp_array=[int(v) for v in comp_array]
    comp_array=np.asarray(comp_array)
    
    calc_array=list(df_calculated_vid["score_array"])[0].split(",")
    calc_array=calc_array[1:-2]    
    calc_array=[int(v) for v in calc_array]
    calc_array=np.asarray(calc_array)

    acc=accuracy_score(calc_array,comp_array)
    avg_acc+=acc
    print(vid,acc)
avg_acc=avg_acc/len(reference_vids)

print("Average accuracy is ",avg_acc)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




