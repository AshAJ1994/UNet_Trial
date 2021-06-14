import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

import os

# Real Images - 1000 N + 1000 G train, 100 N + 100G valid, 78 N + 78 G test
globalMeanRNFL = []
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation/train/glaucoma/Binary'
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/FinalizedGAN_Severity_GANImages/UNet_Sgementation/valid/normal/Binary'
imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/EXTERNAL_TEST_DATASET_UNet/UNet_Segmented_Results/normal/Binary/'
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/EXTERNAL_TEST_DATASET_UNet/UNet_Segmented_Results/glaucoma/Binary'

for imgFile in os.listdir(imgPath):
    img = cv2.imread(os.path.join(imgPath, imgFile), 0)
    img = img / 255.0

    meanRNFL_EachImage = np.mean(img, axis=0) * 2000
    meanRNFL_EachImage = list(meanRNFL_EachImage)
    globalMeanRNFL.append(np.mean(meanRNFL_EachImage))

    # plt.subplot(121), plt.plot(meanRNFL_EachImage)
    # plt.title('RNFL thickness plot'), plt.xlabel('imageWidth'), plt.ylabel('Avg RNFL thickness/column pixel')
    # plt.subplot(122), plt.imshow(img, cmap='gray')
    # plt.title('Original Image')
    # plt.show()

    # print('')

print(globalMeanRNFL)
plt.hist(globalMeanRNFL, bins=5)
plt.xlabel('GlobalAverage_RNFLThickness')
plt.xlim([20, 160])
plt.ylabel('Validation data (10000)')
plt.ylabel('Test data (150)')
# plt.ylabel('Training data (50000)')
# plt.title('Fake_NORMAL_VALID')
plt.title('Alina_Real_NormalTestImages')
# plt.title('Alina_Real_GlaucomaTestImages')
plt.show()
plt.savefig('Alina_Real_NormalTestImages.jpg')
# plt.savefig('Alina_Real_GlaucomaTestImages.jpg')
print('')

# FLATTENED Real Images - 1000 N + 1000 G train, 100 N + 100G valid, 78 N + 78 G test
# globalMeanRNFL = []
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_Severity_FlattemedImages/UNet_Segmentation_Results/train/glaucoma/Binary/'
# for imgFile in os.listdir(imgPath):
#     img = cv2.imread(os.path.join(imgPath, imgFile), 0)
#     img = img / 255.0
#
#     meanRNFL_EachImage = np.mean(img, axis=0) * 1000
#     meanRNFL_EachImage = list(meanRNFL_EachImage)
#     globalMeanRNFL.append(np.mean(meanRNFL_EachImage))
#
#     # plt.subplot(121), plt.plot(meanRNFL_EachImage)
#     # plt.title('RNFL thickness plot'), plt.xlabel('imageWidth'), plt.ylabel('Avg RNFL thickness/column pixel')
#     # plt.subplot(122), plt.imshow(img, cmap='gray')
#     # plt.title('Original Image')
#     # plt.show()
#
#     # print('')
#
# print(globalMeanRNFL)
# plt.hist(globalMeanRNFL)
# plt.xlabel('GlobalAverage_RNFLThickness')
# plt.xlim([20, 180])
# # plt.ylabel('Validation data (100)')
# # plt.ylabel('Training data (1000)')
# plt.ylabel('Test data (78)')
# plt.title('Flattened_Real_GLAUCOMA_TRAIN')
# # plt.show()
# figName = 'Flattened_Real_Glaucoma_Train_Scaled.jpg'
# plt.savefig('/home/sysadmin/PycharmProjects/UNet_AbhishekThakur/Flattened_RealImages_Results/'+figName)
# print('')

# # FLATTENED Fake Images - 5000 N + 5000 G train, 1000 N + 1000G valid, 100 N + 100 G test
# globalMeanRNFL = []
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/Flattened_GAN_FakeImages_Results/Unet_Seg_Resized/valid/glaucoma/Binary/'
# for imgFile in os.listdir(imgPath):
#     img = cv2.imread(os.path.join(imgPath, imgFile), 0)
#     img = img / 255.0
#
#     meanRNFL_EachImage = np.mean(img, axis=0) * 1000
#     meanRNFL_EachImage = list(meanRNFL_EachImage)
#     globalMeanRNFL.append(np.mean(meanRNFL_EachImage))
#
#     # plt.subplot(121), plt.plot(meanRNFL_EachImage)
#     # plt.title('RNFL thickness plot'), plt.xlabel('imageWidth'), plt.ylabel('Avg RNFL thickness/column pixel')
#     # plt.subplot(122), plt.imshow(img, cmap='gray')
#     # plt.title('Original Image')
#     # plt.show()
#
#     # print('')
#
# print(globalMeanRNFL)
# plt.hist(globalMeanRNFL)
# plt.xlabel('GlobalAverage_RNFLThickness')
# plt.xlim([20, 180])
# plt.ylabel('Validation data (1000)')
# # plt.ylabel('Training data (5000)')
# # plt.ylabel('Test data (100)')
# plt.title('Flattened_Fake_GLAUCOMA_VALID')
# # plt.show()
# figName = 'Flattened_Fake_Glaucoma_Valid_Scaled.jpg'
# plt.savefig('/home/sysadmin/PycharmProjects/UNet_AbhishekThakur/Flattened_FakeImages_Results/'+figName)

# FAKE images - 1000 N + 1000 G (picked from test : 500N + 500G, train: 5000N + 5000G, valid: 1000N + 1000G)
# ************* TRAIN data - 1000N + 1000G, TEST data - 78N + 78G ******************
#
# #normals
# list_meanRNFL_NormalImages = []
# list_Normal_fileNames = []
# # train_normalDF = pd.DataFrame()
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/test/normal/Binary_Renamed/'
# for imgFile in os.listdir(imgPath):
#     img = cv2.imread(os.path.join(imgPath, imgFile), 0)
#     img = img / 255.0
#
#     RNFL_EachSector = np.mean(img, axis=0) * 2000
#     RNFL_EachSector = list(RNFL_EachSector)
#     eachImage_meanRNFL = np.mean(RNFL_EachSector)
#     # train_normalDF['Image_FileName'] = imgFile
#     # train_normalDF['Avg_RNFLThickness'] = eachImage_meanRNFL
#     # train_normalDF['Class'] = 1
#     list_meanRNFL_NormalImages.append(eachImage_meanRNFL)
#     list_Normal_fileNames.append(imgFile)
#
# train_normalDF = pd.DataFrame({'Avg_RNFLThickness' : list_meanRNFL_NormalImages, 'Image_Filename' : list_Normal_fileNames})
# train_normalDF['Actual_Class'] = 1
#
# #glaucoma
# list_meanRNFL_GlaucomaImages = []
# list_Glaucoma_fileNames = []
# # train_glaucomaDF = pd.DataFrame()
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/Model_Combinations/Model4_v2_FNFG/UNet_Segmentation/test/glaucoma/Binary_Renamed/'
# for imgFile in os.listdir(imgPath):
#     img = cv2.imread(os.path.join(imgPath, imgFile), 0)
#     img = img / 255.0
#
#     RNFL_EachSector = np.mean(img, axis=0) * 2000
#     RNFL_EachSector = list(RNFL_EachSector)
#     eachImage_meanRNFL = np.mean(RNFL_EachSector)
#     # train_glaucomaDF['Image_FileName'] = imgFile
#     # train_glaucomaDF['Avg_RNFLThickness'] = eachImage_meanRNFL
#     # train_glaucomaDF['Class'] = 1
#     list_meanRNFL_GlaucomaImages.append(eachImage_meanRNFL)
#     list_Glaucoma_fileNames.append(imgFile)
#
# train_glaucomaDF = pd.DataFrame({'Avg_RNFLThickness' : list_meanRNFL_GlaucomaImages, 'Image_Filename' : list_Glaucoma_fileNames})
# train_glaucomaDF['Actual_Class'] = 0
# #concat normal and glaucoma dataframes
# train_DF = pd.concat([train_normalDF,train_glaucomaDF], ignore_index=True)
# # train_DF.to_csv('FakeTestImages_UnetBasedThickness.csv', sep=',')
# train_DF.to_csv('Renamed_FakeTestImages_UnetBasedThickness.csv', sep=',')
#
# from sklearn import metrics
# from sklearn.metrics import roc_curve,roc_auc_score,auc
# fpr, tpr, thrsh = metrics.roc_curve(train_DF['Actual_Class'], train_DF['Avg_RNFLThickness'], pos_label=1)
# auc_score = auc(fpr, tpr)
# print('Fake test images auc score :', auc_score)

# # ************************************** 78 N + 78 G (SEED) internal test dataset ************************************
# # real Images
# # ************* TEST data - 78 N + 78 G ******************
# #normals
# list_meanRNFL_NormalImages = []
# list_Normal_fileNames = []
# # train_normalDF = pd.DataFrame()
# # imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/test/normal/Binary/' #78 real test images
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Unet_test_70Images/normal/binary/' # selected 70 real normal test images
#
# for imgFile in os.listdir(imgPath):
#     img = cv2.imread(os.path.join(imgPath, imgFile), 0)
#     img = img / 255.0
#
#     RNFL_EachSector = np.mean(img, axis=0) * 2000
#     RNFL_EachSector = list(RNFL_EachSector)
#     eachImage_meanRNFL = np.mean(RNFL_EachSector)
#     # train_normalDF['Image_FileName'] = imgFile
#     # train_normalDF['Avg_RNFLThickness'] = eachImage_meanRNFL
#     # train_normalDF['Class'] = 1
#     list_meanRNFL_NormalImages.append(eachImage_meanRNFL)
#     list_Normal_fileNames.append(imgFile)
#
# test_normalDF = pd.DataFrame({'Avg_RNFLThickness' : list_meanRNFL_NormalImages, 'Image_Filename' : list_Normal_fileNames})
# test_normalDF['Actual_Class'] = 1
#
# #glaucoma
# list_meanRNFL_GlaucomaImages = []
# list_Glaucoma_fileNames = []
# # test_glaucomaDF = pd.DataFrame()
# # imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/FINALIZED_GAN_GLAUCOMA_DATA_Severity/UNet_Segmentation_RealImages/test/glaucoma/Binary/'
# imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/GANINput_SyntheticData_Filtered_v2_600/Unet_test_70Images/glaucoma/binary/' # selected 70 real glaucoma test images
#
# for imgFile in os.listdir(imgPath):
#     img = cv2.imread(os.path.join(imgPath, imgFile), 0)
#     img = img / 255.0
#
#     RNFL_EachSector = np.mean(img, axis=0) * 2000
#     RNFL_EachSector = list(RNFL_EachSector)
#     eachImage_meanRNFL = np.mean(RNFL_EachSector)
#     # train_glaucomaDF['Image_FileName'] = imgFile
#     # train_glaucomaDF['Avg_RNFLThickness'] = eachImage_meanRNFL
#     # train_glaucomaDF['Class'] = 1
#     list_meanRNFL_GlaucomaImages.append(eachImage_meanRNFL)
#     list_Glaucoma_fileNames.append(imgFile)
#
# test_glaucomaDF = pd.DataFrame({'Avg_RNFLThickness' : list_meanRNFL_GlaucomaImages, 'Image_Filename' : list_Glaucoma_fileNames})
# test_glaucomaDF['Actual_Class'] = 0
# #concat normal and glaucoma dataframes
# test_DF = pd.concat([test_normalDF,test_glaucomaDF], ignore_index=True)
# # test_DF.to_csv('RealTestImages_UnetBasedThickness.csv', sep=',')
# test_DF.to_csv('RealTest_70Images_UnetBasedThickness.csv', sep=',')
# from sklearn import metrics
# from sklearn.metrics import roc_curve,roc_auc_score,auc
# fpr, tpr, thrsh = metrics.roc_curve(test_DF['Actual_Class'], test_DF['Avg_RNFLThickness'], pos_label=1)
# auc_score = auc(fpr, tpr)
# #******************************* FINISH ***********************************************************************


# ************************************** 150 N + 150 G (ALINA) external test dataset ************************************
# real Images
# ************* TEST data - 150 N + 150 G ******************

#normals
list_meanRNFL_NormalImages = []
list_Normal_fileNames = []
# below selected 150 real normal test images
imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/EXTERNAL_TEST_DATASET_UNet/UNet_Segmented_Results/normal/Binary/'

for imgFile in os.listdir(imgPath):
    img = cv2.imread(os.path.join(imgPath, imgFile), 0)
    img = img / 255.0
    RNFL_EachSector = np.mean(img, axis=0) * 2000
    RNFL_EachSector = list(RNFL_EachSector)
    eachImage_meanRNFL = np.mean(RNFL_EachSector)
    list_meanRNFL_NormalImages.append(eachImage_meanRNFL)
    list_Normal_fileNames.append(imgFile)

test_normalDF = pd.DataFrame({'Avg_RNFLThickness' : list_meanRNFL_NormalImages, 'Image_Filename' : list_Normal_fileNames})
test_normalDF['Actual_Class'] = 1

#glaucoma
list_meanRNFL_GlaucomaImages = []
list_Glaucoma_fileNames = []
# below : selected 150 real glaucoma test images
imgPath = r'/home/sysadmin/Ashish_PGAN_Validation/EXTERNAL_TEST_DATASET_UNet/UNet_Segmented_Results/glaucoma/Binary/'

for imgFile in os.listdir(imgPath):
    img = cv2.imread(os.path.join(imgPath, imgFile), 0)
    img = img / 255.0
    RNFL_EachSector = np.mean(img, axis=0) * 2000
    RNFL_EachSector = list(RNFL_EachSector)
    eachImage_meanRNFL = np.mean(RNFL_EachSector)
    list_meanRNFL_GlaucomaImages.append(eachImage_meanRNFL)
    list_Glaucoma_fileNames.append(imgFile)

test_glaucomaDF = pd.DataFrame({'Avg_RNFLThickness' : list_meanRNFL_GlaucomaImages, 'Image_Filename' : list_Glaucoma_fileNames})
test_glaucomaDF['Actual_Class'] = 0
#concat normal and glaucoma dataframes
test_DF = pd.concat([test_normalDF,test_glaucomaDF], ignore_index=True)
# test_DF.to_csv('RealTestImages_UnetBasedThickness.csv', sep=',')
test_DF.to_csv('ALINA_TestDataset_150Images_UnetBasedThickness.csv', sep=',')
from sklearn import metrics
from sklearn.metrics import roc_curve,roc_auc_score,auc
fpr, tpr, thrsh = metrics.roc_curve(test_DF['Actual_Class'], test_DF['Avg_RNFLThickness'], pos_label=1)
auc_score = auc(fpr, tpr)
#******************************* FINISH ***********************************************************************

print('finished!')

