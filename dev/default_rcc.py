import os, sys, cv2, re
REPO_PATH = os.getcwd()[:-len('dev')]
sys.path.append(REPO_PATH+'init/')

import __init_paths__ as init_paths


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


            ## Skeletal information
            #file = open(init_paths.ANNOTATION_PATH+subject[:9]+'annotations/'+\
            #  activity+'/'+task+'.txt')
            #skeletal_lines = file.readlines()
            #P, ORI,_,_ =readSkeletons(skeletal_lines)

            # Parse frames of video ...
            for frame in range(lendir+1):
                imgname = PATH+'RGB_'+str(frame+1)+'.png'
                img = cv2.imread(imgname)

                # For every object in the video ...
                for o in range(object_num-1): # -1 for excluding the table bounding box
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
                    cv2.putText(img, nametext, (b_center_x, b_center_y),\
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,150,30), 2, cv2.LINE_AA)
                # Visualize
                cv2.imshow('CAD-120', img)
                k = cv2.waitKey(30) & 0xff

