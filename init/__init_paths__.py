# PATHS
import getpass

if getpass.getuser() == 'scat':
    DATASET_PATH = '/home/scat/Datasets/CAD-120/'
elif getpass.getuser() == 'toumpa':
    DATASET_PATH = '/Users/toumpa/Datasets/CAD-120/'
else:
    DATASET_PATH = 'YOUR DATASET PATH HERE'
    # When you include the DATASET_PATH comment out the following couple of lines.
    print("Error : Dataset paath missing.\nPlease include the CAD-120 dataset path in the init/__init_paths__.py file.\n")
    exit()
ANNOTATION_PATH = DATASET_PATH + 'enhanced_annotations/'
ANNOT_RCNN = DATASET_PATH + 'annotations_rcnn/'

