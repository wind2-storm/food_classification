# no package for independent configuration management
import os

# config.py가 config/ 폴더 안에 있으므로, 두 단계 상위가 프로젝트 루트
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize the parameters
CONF_THRES = 0.1
NMS_THRES  = 0.1

INPWIDTH  = 416
INPHEIGHT = 416

Video_Start_Frame = (21-4)*60*30+(2-39)*30

# model base directory (절대경로)
ModelBaseDir = os.path.join(BASE_DIR, "yolo")

# 테스트 파일
TEST_IMAGE_PATH = os.path.join(ModelBaseDir, "data/food_images/A250101_111424_0032.jpg")
TEST_VIDEO_PATH = ""

SHOW_TEXT_FLAG = 1
PS_FLAG = 1

# classes & codes 파일
CLASSES_FILE = os.path.join(ModelBaseDir, "data/food/food-classes.names")
CLASSES_FILE_CODE = os.path.join(ModelBaseDir, "data/food/food-classes.codes")

# YOLO config + weights
Model_Configuration = os.path.join(ModelBaseDir, "config/food-dark-yolov3-tiny_3l-v3-2.cfg")
Model_Weights = os.path.join(ModelBaseDir, "data/food/weights/food-dark-yolov3-tiny_3l-v3-2_24000.weights")
