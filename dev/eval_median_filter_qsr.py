""" eval_median_filter_qsr
Detects the pair-wise RCC relations of the objects.

Set Parameters:
 - ../__init_paths__/ : the paths of the dataset's information
 - ../init/__init_qsr__/qsr : the qsr value we want to detect
"""

import os, sys
REPO_PATH = os.getcwd()[:-len('dev')]
sys.path.append(REPO_PATH+'init/')
sys.path.append(REPO_PATH+'qsrlib/src/')

import __init_paths__ as init_paths
import __init_objects__ as init_obj
import __init_qsr__ as init_qsr

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace
import qsrlib_qstag.utils as utils
import qsrlib_qstag.qstag
from load_data import get_all_video_data

import cv2, re, itertools
import pandas as pd

global frame
global window_size

# Data directory
DATA_PATH = REPO_PATH + 'data/'
EXP_PATH = DATA_PATH +'exp/'
GROUNDTRUTH_PATH = DATA_PATH + 'groundtruth_relations/'

'''
def qsr_relation_between(obj1_name, obj2_name, obj1, obj2):
    global frame
    global window_size

    qsrlib = QSRlib()
    options = sorted(qsrlib.qsrs_registry.keys()) 
    if init_qsr.qsr not in options:
        raise ValueError("qsr not found, keywords: %s" % options)

    world = World_Trace()

    object_types = {obj1_name: obj1_name,
                    obj2_name: obj2_name}

    dynamic_args = {"filters": {"median_filter": {"window": window_size}},
                    "qstag": {"object_types" : object_types,
                              "params" : {"min_rows" : 1, 
                                          "max_rows" : 1, 
                                          "max_eps" : 3}},

                    "tpcc" : {"qsrs_for": [(obj1_name, obj2_name)]},

                    "rcc2": {"qsrs_for": [(obj1_name, obj2_name)]},

                    "rcc4": {"qsrs_for": [(obj1_name, obj2_name)]},

                    "rcc8": {"qsrs_for": [(obj1_name, obj2_name)]}
                    }

    o1 = [Object_State(name=obj1_name, timestamp=frame, x=obj1[0], y=obj1[1], \
      xsize=obj1[2], ysize=obj1[3])]
    o2 = [Object_State(name=obj2_name, timestamp=frame, x=obj2[0], y=obj2[1], \
      xsize=obj2[2], ysize=obj2[3])]
    world.add_object_state_series(o1)
    world.add_object_state_series(o2)

    qsrlib_request_message = QSRlib_Request_Message(which_qsr=init_qsr.qsr, \
      input_data=world, dynamic_args=dynamic_args)
    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)
    pretty_print_world_qsr_trace(init_qsr.qsr, qsrlib_response_message)#, vis = True)
    qsr_value = find_qsr_value(init_qsr.qsr, qsrlib_response_message)
    return qsr_value
'''

def add_qsr(state, object_name, box):
    global frame
    x, y, xs, ys = box[0], box[1], box[2], box[3]
    state.append(Object_State(name=object_name, timestamp=frame, x=x, y=y, \
      xsize=xs, ysize=ys))
    return state

def dynamic_qsr_arguments(objects):

    global window_size

    object_types = {objects[i]: objects[i] for i in range(len(objects))}
    object_tuples = []
    for pair in itertools.combinations(object_types.keys(), 2):
        object_tuples.append(pair)

    dynamic_args = {
                    "filters": {"median_filter": {"window": window_size}},
                    "qstag": {"object_types" : object_types,
                              "params" : {"min_rows" : 1, "max_rows" : 1, "max_eps" : 3}},

                    "tpcc" : {"qsrs_for": object_tuples},

                    "rcc2": {"qsrs_for": object_tuples},

                    "rcc4": {"qsrs_for": object_tuples},

                    "rcc5": {"qsrs_for": object_tuples},

                    "rcc8": {"qsrs_for": object_tuples}
                    }

    return dynamic_args

def qsr_message(state_series, objects):
    qsrlib = QSRlib()
    options = sorted(qsrlib.qsrs_registry.keys())
    if init_qsr.qsr not in options:
        raise ValueError("qsr not found, keywords: %s" % options)

    world = World_Trace()

    # Create dynamic arguments for every qsr type
    dynamic_args = dynamic_qsr_arguments(objects)
    
    # Add all object states in the World.
    for o in state_series:
        world.add_object_state_series(o)
    # Create request message
    qsrlib_request_message = QSRlib_Request_Message(which_qsr=init_qsr.qsr, input_data=world, dynamic_args=dynamic_args)

    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)

    pretty_print_world_qsr_trace(init_qsr.qsr, qsrlib_response_message, vis = True)


def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message, vis=False):
    if vis:
        print("---")
Last login: Fri Dec  6 14:58:53 on ttys000
toumpa@Alexias-MBP:~$ ssh -X scat@remote-access.leeds.ac.uk
scat@remote-access.leeds.ac.uk's password: 
Last login: Fri Dec  6 16:44:40 2019 from 176.12.107.132
euras01hv.leeds.ac.uk RedHat 6.10 x86_64

Puppet: 3.8.7
Facter: 2.4.6
Uptime: 6 days
[scat@euras01hv ~]$ ssh -X scat@129.11.28.165
scat@129.11.28.165's password: 
Welcome to Ubuntu 16.04.4 LTS (GNU/Linux 4.13.0-38-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

6 packages can be updated.
0 updates are security updates.

Last login: Fri Dec  6 16:45:36 2019 from 129.11.190.34
scat@HPZ820:~$ tmux a -t qsr_detect



        print("Response is:")
        for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
            foo = str(t) + ": "
            for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
              qsrlib_response_message.qsrs.trace[t].qsrs.values()):
                foo += str(k) + ":" + str(v.qsr) + "; "
            print(foo)

def find_qsr_value(which_qsr, qsrlib_response_message):
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        v = qsrlib_response_message.qsrs.trace[t].qsrs.values()
        return str(v[0].qsr[which_qsr])


def video_list(IMAGE_PATH):
    all_tasks = []
    for subject in os.listdir(IMAGE_PATH):
        for activity in os.listdir(IMAGE_PATH+subject+'/'):
            for task in os.listdir(IMAGE_PATH+subject+'/'+activity+'/'):
                PATH = IMAGE_PATH + subject + '/' + activity + '/' + task + '/'
                all_tasks.append(PATH)
    return all_tasks



def main(video_idx):

    correct_detections, all_detections = 0, 0

    global frame

    IMAGE_PATH = init_paths.DATASET_PATH + 'images/'
    all_tasks = video_list(IMAGE_PATH)

    # Groundtruth data
    groundtruth_data = get_all_video_data()
    video_name = re.split('/',all_tasks[video_idx])[-2:-1][0]
    video_data = groundtruth_data[groundtruth_data.videos_id == int(video_name)]

    if video_data.empty:
        return -1

    total_frames = len(os.listdir(all_tasks[video_idx]))/2

    for f in range(2, total_frames+1):
        frame_data = video_data[video_data.frames_id == f]


        # Parse data
        for index,data in frame_data.iterrows():
            video_task = data.videos_id
            frame = data.frames_id
            rcc = data.rcc_resp
            object1 = [data.X_1, data.Y_1, data.W_1, data.H_1]
            object2 = [data.X_2, data.Y_2, data.W_2, data.H_2]

            # Find video
            video_path = [i for i in all_tasks if str(video_task) in i]
            video_path = video_path[0]
            video = re.sub(IMAGE_PATH,'', video_path)
            (subject, activity, task, _) = re.split('/', video)
            PATH = IMAGE_PATH + subject + '/' + activity + '/' + task + '/'

            lendir = len(os.listdir(PATH))/2
            
            # Compute total number of objects in the video
            object_num = sum(1 for i in os.listdir(init_paths.ANNOTATION_PATH+\
              subject[:9]+'annotations/'+activity+'/') if (task in i) and \
              ('obj' in i) )
            object_num = object_num-1 if 'enhanced' in \
              init_paths.ANNOTATION_PATH else object_num
            object_pairs = list(itertools.permutations(range(object_num),2))
            object_states = [[] for o in range(object_num)]
            object_names_list = [o+1 for o in range(object_num)]


            imgname = PATH + 'RGB_' + str(frame) + '.png'
            img = cv2.imread(imgname)
            # For every frame re-init objects' information
            init_obj.init_objects()

            # For every object in the video ...
            for o in range(object_num):
                file = open(init_paths.ANNOTATION_PATH+subject[:9]+'annotations/'+\
                  activity+'/'+task+'_obj'+str(o+1)+'.txt', 'r')
                lines = file.readlines()
                line = lines[frame-1]
                coordinates = re.split(',', line)
                ulx, uly = float(coordinates[2]), float(coordinates[3])
                lrx, lry = float(coordinates[4]), float(coordinates[5])

                b_center_x = int(min(ulx,lrx) + abs(lrx-ulx)/2)
                b_center_y = int(min(uly,lry) + abs(lry-uly)/2)
                nametext = str(o+1) #'obj_' + str(o+1)
                # Keep the object's bounding box information for QSR computations.
                init_obj.object_coord.append([b_center_x, b_center_y, \
                  int(abs(lrx-ulx)), int(abs(lry-uly))])
                init_obj.objects.append([ulx, uly, int(abs(lrx-ulx)), int(abs(lry-uly))])
                init_obj.object_name.append(nametext)

            # Compute RCC relations.
            for (o1,o2) in object_pairs:
                if init_obj.objects[o1] == object1 and \
                  init_obj.objects[o2] == object2:
                    # Keep QSR state of each object
                    object_idx = object_pairs.index([o1,o2])
                    states[object_pair_idx] = add_qsr(states[object_pair_idx], )
                    qsr_relation = (qsr_relation_between(init_obj.object_name[o1], \
                      init_obj.object_name[o2], init_obj.object_coord[o1], \
                      init_obj.object_coord[o2])).upper()
                    #print(qsr_relation, rcc)
                    if qsr_relation == rcc or ((qsr_relation == 'DC' or \
                      qsr_relation == 'DR') and (rcc == 'DC' or rcc == 'DR')):
                        correct_detections += 1
                    all_detections += 1
              
        k = cv2.waitKey(30) & 0xff
        if k ==27:
            break
    # Compute accuracy
    accuracy = float(correct_detections)/float(all_detections)
    return accuracy


if __name__== "__main__":

    num_videos = len(video_list(init_paths.DATASET_PATH + 'images/'))

    global window_size
    for w in [3,5,7,9]:
        window_size = w
        print("window size: %s" % str(window_size))
        video_accuracy, count_videos = 0.0, 0
        for i in range(num_videos):
            if main(i) == -1:
                continue
            video_accuracy += main(i)
            count_videos += 1
        accuracy = video_accuracy / float(count_videos)
        print(accuracy)

