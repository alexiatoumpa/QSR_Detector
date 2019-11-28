# Flag setting if the groundtruth or the Faster R-CNN bounding boxes are being used.
use_rcnn = False

def init_video():
    # Object coordinate matrices
    global UL
    global LR
    global T
    global UL_gt
    global LR_gt
    # Object information
    global num_of_objs
    num_of_objs = 0

    global object_data
    object_data = []


def init_objects():
    global object_coord
    object_coord = []
