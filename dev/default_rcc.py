""" default_rcc
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

import cv2, re, itertools, csv

global frame

# Data directory
DATA_PATH = REPO_PATH + 'data/'
EXP_PATH = DATA_PATH +'exp/'
GROUNDTRUTH_PATH = DATA_PATH + 'groundtruth_relations/'


def qsr_relation_between(obj1_name, obj2_name, obj1, obj2):
    global frame

    qsrlib = QSRlib()
    options = sorted(qsrlib.qsrs_registry.keys()) 
    if init_qsr.qsr not in options:
        raise ValueError("qsr not found, keywords: %s" % options)

    world = World_Trace()

    object_types = {obj1_name: obj1_name,
                    obj2_name: obj2_name}

    dynamic_args = {"qstag": {"object_types" : object_types,
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

def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message, vis=False):
    if vis:
        print("---")
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



if __name__== "__main__":

    global frame

    IMAGE_PATH = init_paths.DATASET_PATH + 'images/'

    for subject in os.listdir(IMAGE_PATH):
        print("Press ESC key to skip to next video. Press Ctrl+C to exit.\n")
        for activity in os.listdir(IMAGE_PATH+subject+'/'):
            for task in os.listdir(IMAGE_PATH+subject+'/'+activity+'/'):

                PATH = IMAGE_PATH+subject+'/'+activity+'/'+task+'/'
                print(PATH)
                lendir = len(os.listdir(PATH))/2
            
                # Compute total number of objects in the video
                object_num = sum(1 for i in os.listdir(init_paths.ANNOTATION_PATH+\
                  subject[:9]+'annotations/'+activity+'/') if (task in i) and \
                  ('obj' in i) )
                object_num = object_num-1 if 'enhanced' in \
                  init_paths.ANNOTATION_PATH else object_num
                object_pairs = list(itertools.combinations(range(object_num),2))

                ## Skeletal information
                #file = open(init_paths.ANNOTATION_PATH+subject[:9]+'annotations/'+\
                #  activity+'/'+task+'.txt')
                #skeletal_lines = file.readlines()
                #P, ORI,_,_ =readSkeletons(skeletal_lines)

                save_qsr_file = EXP_PATH+subject[:8]+'-'+activity+'-'+task+'.csv'
                if os.path.exists(save_qsr_file): os.remove(save_qsr_file)
                # Write first line of file
                with open(save_qsr_file, mode='a') as csvfile:
                    datawrite = csv.writer(csvfile, delimiter=';', quotechar='"', \
                      quoting=csv.QUOTE_MINIMAL)
                    datawrite.writerow(['frame', object_pairs])

                # Parse frames of video ...
                for frame in range(lendir+1):
                    imgname = PATH+'RGB_'+str(frame+1)+'.png'
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
                        # Visualize object bounding box
                        p1 = (int(ulx), int(uly))
                        p2 = (int(ulx+(lrx-ulx)), int(uly+(lry-uly)))
                        cv2.rectangle(img, p1, p2, (255,0,0),2,1)

                        b_center_x = int(min(ulx,lrx) + abs(lrx-ulx)/2)
                        b_center_y = int(min(uly,lry) + abs(lry-uly)/2)
                        nametext = str(o+1) #'obj_' + str(o+1)
                        # Visualize object name
                        cv2.putText(img, nametext, (b_center_x, b_center_y),\
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,150,30), 2, cv2.LINE_AA)
                        # Keep the object's bounding box information for QSR computations.
                        init_obj.object_coord.append([b_center_x, b_center_y, \
                          int(abs(lrx-ulx)), int(abs(lry-uly))])
                        init_obj.object_name.append(nametext)

                    # Compute RCC relations.
                    all_relations = []
                    for (o1,o2) in object_pairs:
                        qsr_relation = qsr_relation_between(init_obj.object_name[o1], \
                          init_obj.object_name[o2], init_obj.object_coord[o1], \
                          init_obj.object_coord[o2])
                        all_relations.append(qsr_relation)
                    # Save relations in file.
                    with open(save_qsr_file, mode='a') as csvfile:
                        writedata = csv.writer(csvfile,delimiter=';',quotechar='"', \
                          quoting=csv.QUOTE_MINIMAL)
                        writedata.writerow([frame, all_relations])
                      
                    # Visualize
                    cv2.imshow('CAD-120', img)
                    k = cv2.waitKey(30) & 0xff
                    if k ==27:
                        break

