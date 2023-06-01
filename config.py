from envbash import load_envbash
import os
import shutil
if not os.path.exists("config.env"):
    # copy "scripts/config.env.default" to "config.env" with shutil.copyfile(src, dst)
    shutil.copyfile("scripts/config.env.default", "config.env")

load_envbash("config.env")
N_FISHES = 24
BATCH_SIZE = 1000
BACK = "back"
sep = ";"
VIS_DIR = "vis"
DIR_TRACES= "traces"
CAM_POS = "cam_pos"
DAY="day"
BATCH="batch"
DATAFRAME="DATAFRAME"
HOURS_PER_DAY = 8
FRAMES_PER_SECOND = 5
projectPath = os.environ["projectPath"]
BLOCK1 = os.environ["BLOCK1"]
BLOCK2 = os.environ["BLOCK2"]
BLOCK = os.environ["BLOCK"]