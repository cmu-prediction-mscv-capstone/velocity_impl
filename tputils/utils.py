import numpy as np
import os
import glob
import pdb
import random
def trajectory_matrix_norm(data_file,img_width,img_height,mode=2):
    '''Modify the data file accordingly'''
    return data_file
    

def get_ade(pr,gr,length):
    '''Compute the ground truth trajectory'''
    diff=np.abs(gr-pr)
    diff_norm=np.linalg.norm(diff,axis=2)
    #print(diff_norm.shape)
    ade=diff_norm.sum()/(diff_norm.shape[0]*diff_norm.shape[1])
    return ade
   

def get_fde(pr,gr):
    '''Compute fde'''
    diff=np.abs(gr-pr)
    diff_norm=np.linalg.norm(diff,axis=2)
    #print(diff_norm.shape)
    diff_last=diff[:,-1]
    fde=diff_last.sum()/diff_norm.shape[0]
    return fde


def make_trajectories_array(dataset_paths,traj_length,random_update=False):
    file_names=[]
    trajectories=[]
    for dataset_path in dataset_paths:
        file_names.append(glob.glob(dataset_path+"/*.txt"))
    file_names=[x for file_name in file_names for x in file_name]
    #print(file_names)
    #pdb.set_trace()
    for file_name in file_names:
        all_ped={}
        print(file_name)
        file_pointer=open(file_name)
        lines=file_pointer.readlines()
        for line in lines:
            split_line=line.split("\t")
            frame=int(float(split_line[0]))
            ped=int(float(split_line[1]))
            x=float(split_line[2])
            y=float(split_line[3])
            if ped not in all_ped:
                all_ped[ped]=[[x,y,frame]]
            else:
                #pdb.set_trace()
                if frame-all_ped[ped][-1][2]!=10:
                    print("Pedestrian {} error at frame {}".format(ped,frame))
                    #pdb.set_trace()
                else:
                    all_ped[ped].append([x,y,frame])
        trajectories= trajectories+ build_trajectories(all_ped,traj_length,random_update)
        file_pointer.close()
    out=np.asarray(trajectories,dtype=np.float32)
    return out[:,:,0:2]

def build_trajectories(ped_dict,traj_length,random_update=False):
    '''send the trajectories from the dictionaries. These trajectories will be apppended in the previous trajectories'''
    ''' A list of n_trajxtraj_sizex2'''
    trajectories=[]
    #pdb.set_trace()
    for key in sorted(ped_dict.keys()):
        #print(key)
        index=0
        while index<len(ped_dict[key])-traj_length:
            trajectories.append(ped_dict[key][index:index+traj_length])
            if random_update:
                index+=random.randint(1,traj_length)
            else:
                index+=traj_length
    return trajectories
