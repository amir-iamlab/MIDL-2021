import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
from tensorflow.keras.preprocessing.image import load_img

from AD_classification.config.frequentist_config import AdniDirSkull, DataDir, Normal_PathDir, ImageSize, LabelNum, ImageType, MODEL, slice_num
import matplotlib.pyplot as plt
import os
import itertools

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import argparse
from AD_classification.preprocessor.aspectawareresize import aspectawareresize
from AD_classification.LR.clr import LRFinder, OneCycleLR

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Convolution2D, AveragePooling2D, BatchNormalization,
                                     Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dropout, Dense,
                                     Flatten, Activation, BatchNormalization,
                                     Lambda, Reshape, Conv2DTranspose, UpSampling2D)
from AD_classification.config.frequentist_config import ImageSize, BS, OUTPUT_PATH, k_fold, LabelNum, NUM_EPOCHS, MODEL
from AD_classification.LR import lr_config
from AD_classification.LR.learningratefinder import LearningRateFinder
from AD_classification.Loss.focal_loss import categorical_focal_loss
from AD_classification.LR.clr_callback import CyclicLR
from tensorflow.keras.models import (Model, Sequential, load_model)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.xception import Xception
from AD_classification.models.vgg16 import VGG16
#from AD_classification.models.vgg19 import VGG19
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from AD_classification.config.frequentist_config import AdniDirSkull, DataDir, Normal_PathDir, ImageSize, LabelNum, ImageType
from AD_classification.models.alexnet import alexnet_model
from AD_classification.preprocessor.centercrop import centercrop
from AD_classification.DataLoader.NormalImageWriter import NormalImageWriter
from keras.backend.tensorflow_backend import set_session
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, matthews_corrcoef, precision_recall_curve, confusion_matrix, plot_roc_curve, RocCurveDisplay



# Load a model
# image_size = ImageSize
testdatagen = ImageDataGenerator(rescale=1./255, featurewise_center=True,
    featurewise_std_normalization=True)

test_dir = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset_new\\" + str(LabelNum) + "-Class\\" + slice_num + "\\test"

if LabelNum == 2:
    activation = 'sigmoid'
else:
    activation = 'softmax'

# filepath = os.path.sep.join([OUTPUT_PATH, MODEL + '-' + str(LabelNum) + "class.hdf5"])
# classes = utils.get_imagenet_classes()
model = vit.vit_b32(
        image_size=ImageSize,
        activation=activation,
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=LabelNum,
        weights="imagenet21k",
        dropout=0.2,
        mlp_dim=3072,  # 3072 hidden layer dimension
        num_heads=24,  # 12 Number of heads in Multi-head Attention layer
        num_layers=12,  # 12 number of transformer
        hidden_size=768  #768 embedding dimension
    )
# model.summary()
# classes = utils.get_imagenet_classes()
# model = Model(inputs=base_model.input, outputs=predictions)
# model.load_weights(OUTPUT_PATH + "\\" + MODEL + "-" + str(LabelNum) + "class - " + str(slice_num) + ".hdf5")
model.load_weights(OUTPUT_PATH + '\\' + MODEL + '-' + str(LabelNum) + "class - " + str(slice_num) + ".hdf5")

testGen = testdatagen.flow_from_directory(test_dir, shuffle=False, target_size=(ImageSize, ImageSize), batch_size=BS)
print("[INFO] predicting on test data ...")

predictions = model.predict_generator(testGen, steps=len(testGen))
# correct_indices = np.nonzero(np.argmax(predictions, axis=1) == testGen.labels)[0]
# incorrect_indices = np.nonzero(np.argmax(predictions, axis=1) != testGen.labels)[0]
# print(len(correct_indices)," classified correctly")
# print(len(incorrect_indices)," classified incorrectly")
# # #Adapt figure size to accomodate 18 subplots plt.rcParams['figure.figsize'] = (7,14) plt.figure() # plot 9 correct predictions
# #
# for i, correct in enumerate(correct_indices[:9]):
#     plt.subplot(6,3,i+1)
#     plt.imshow(testGen.filepaths[correct], cmap='gray', interpolation='none')
#     plt.title("Predicted: {}, Truth: {}".format(predictions[correct], testGen.labels[correct]))
#     plt.xticks([])
#     plt.yticks([])
#
# # plot 9 incorrect predictions
# for i, incorrect in enumerate(incorrect_indices[:9]):
#     plt.subplot(6,3,i+10)
#     plt.imshow(testGen.filepaths[incorrect], cmap='gray', interpolation='none')
#     plt.title( "Predicted {}, Truth: {}".format(predictions[incorrect], testGen.labels[incorrect]))
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
scores = np.mean(np.equal(testGen.labels, np.argmax(predictions, axis=1))) * 100
print(classification_report(testGen.labels, np.argmax(predictions, axis=1), digits=4))
# print("MCC score is: {:.2f}".format(matthews_corrcoef(testGen.labels, np.argmax(predictions, axis=1))*100))
print("Acc score is: {:.2f}".format(accuracy_score(testGen.labels, np.argmax(predictions, axis=1))*100))
print("AUC score is: {:.4f}".format(roc_auc_score(testGen.labels, np.argmax(predictions, axis=1), average='macro')))
roc_auc = roc_auc_score(testGen.labels, np.argmax(predictions, axis=1), average='macro')
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          #cmap=plt.cm.gist_yarg,
                          cmap='Blues',
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    plt.style.use('seaborn-ticks')
    # plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    #plt.grid('on')
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=0)
    plt.yticks(tick_marks, target_names)

    fmt = '.3f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize and cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('\nPredicted labels')


cm = confusion_matrix(testGen.labels, np.argmax(predictions, axis=1))
np.set_printoptions(precision=2)
plot_confusion_matrix(cm=cm, normalize=False, target_names=['AD', 'CN'],
                      title=" ViT-B/32 Confusion Matrix")
#
plt.grid('off')
plt.grid(b=None)
address = os.path.sep.join([OUTPUT_PATH, "CM_" + MODEL + '-' + str(LabelNum) + "class - " + str(slice_num) + ".png"])
plt.savefig(address, dpi = 1000, facecolor='w', edgecolor='w',
orientation='landscape', papertype=None, format=None,
transparent=False, bbox_inches='tight', pad_inches=None)
# plt.show()

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print('Sensitivity/Recall or True Positive Rate =', TPR*100)
# Specificity or true negative rate
TNR = TN/(TN+FP)
print('Specificity or True Negative Rate =', TNR*100)

# Fall out or false positive rate
FPR = FP/(FP+TN)
print('False Positive Rate =', FPR*100)
# False negative rate
FNR = FN/(TP+FN)
print('False Negative Rate =', FNR*100)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print('Precision or Positive Predictive Value =', PPV*100)
# Negative predictive value
NPV = TN/(TN+FN)
print('Negative Predictive Value =', NPV*100)


# False discovery rate
FDR = FP/(TP+FP)
print('False Discovery Rate =', FDR*100)
# display = RocCurveDisplay(fpr=FPR, tpr=TPR, roc_auc=roc_auc)
# display.plot()
# plt.show()

# img = 'C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset\\3-Class\\Slice 27-42\\Test\\AD\\003_S_4136-time1-slices27.png'
# img = testGen.filepaths[0]
# image = load_img(img)
# attention_map = visualize.attention_map(model=model, image=image)
# # print('Prediction:', classes[
# #     model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
# # )  # Prediction: Eskimo dog, husky
#
# # Plot results
# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.axis('off')
# ax2.axis('off')
# ax1.set_title('Original')
# ax2.set_title('Attention Map')
# _ = ax1.imshow(image)
# _ = ax2.imshow(attention_map)
# address = "C:\\RyeU\\PhD\\test"
# plt.savefig(address, dpi = 1000, facecolor='w', edgecolor='w',
# orientation='landscape', papertype=None, format=None,
# transparent=False, bbox_inches='tight', pad_inches=None)
