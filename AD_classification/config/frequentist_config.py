# define the paths to the images directory
# CLASSES = ["CN", "MCI", "AD"]
# CLASSES = ["CN", "AD"]

#slices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          #30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
slices = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
slice_num = "Slice 27-42"
# slices = [36]
DataType = 0
k_fold = 10
ImageSize = 256
BS = 16
NUM_EPOCHS = 320
MODEL = 'B32_newreg_12' # ViT-b32
LabelNum = 2
ImageType = 2


DATA_PATH = "C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\Original_mat"
AdniDir = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\Std_Reg_mat'
LabelDir = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\New folder\\Final\\Label_1402.csv'
DataDir = 'C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset'
DataDir_new = 'C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset_new'

MaskDir = "C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\BrainMasks_Registered"
AdniDirSkull = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\Std_Reg_SkullStripped'
AdniDirSkull_new = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\Std_Reg_SkullStripped_new'
CCNADirSkull = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\CCNA'

CV_PathDir = "C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\CV"
Normal_PathDir = "C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Normal"
NORMAL_PATH_DIR_EQUAL = "C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\NormalEqual"
TrainDir = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset\\3-Class\\Train"

# path to the output model file
MODEL_PATH = "C:\\RyeU\\PhD\\Thesis\\AD_classification\\output\\Xception_slices20.model"
WEIGHT_PATH = "C:\\RyeU\\PhD\\Thesis\\AD_classification\\output"

# define the path to the dataset mean
DATASET_MEAN = "C:\\RyeU\\PhD\\Thesis\\AD_classification\\output\\AD_classification_mean.json"

# define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "C:\\RyeU\\PhD\\Thesis\\Keras\\AD_classification\\output"

