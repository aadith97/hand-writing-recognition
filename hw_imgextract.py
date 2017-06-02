import numpy as np
import os
from PIL import Image
from scipy import misc
from skimage import io
import random
CharRepo = misc.imread('dataset_image/0_zero/images0.jpg')
CharRepo = np.array([CharRepo])

CharLabelRepo = np.array([0])
print(CharLabelRepo.shape)
print(CharRepo.shape)
print(os.listdir('dataset_image'))
for label_folder in os.listdir('dataset_image'):
    print(label_folder)
    listImages=[file for file in os.listdir('dataset_image'+'/'+label_folder) if file.endswith('.jpg' or '.jpeg')]
    for image in listImages:
        photo=misc.imread('dataset_image'+'/'+label_folder+'/'+image)
        CharRepo=np.append(CharRepo,[photo],axis=0)
        label=int(label_folder[0])
        CharLabelRepo=np.append(CharLabelRepo,[label],axis=0)
CharRepo = CharRepo[1:]
CharLabelRepo = CharLabelRepo[1:]

print(CharLabelRepo)
print(type(CharLabelRepo))
x=CharRepo
y=CharLabelRepo
randomize = np.arange(len(CharLabelRepo))
np.random.shuffle(randomize)
CharLabelRepo=y = CharLabelRepo[randomize]
CharRepo =x= CharRepo[randomize]
def traintest_extract(test_ratio):
    l=len(x)
    testl=int(test_ratio*l)
    return (x[testl:],y[testl:], x[:testl], y[:testl])
    # return (x, y, x[:testl], y[:testl])

a,b,c,d=traintest_extract(0.1)



# a=[0,1,2,3,4,5,6]
# b=[0.5,1.5,2.5,3.5,4.5,5.5,6.5]
# a=np.array(a)
# b=np.array(b)
# randomize = np.arange(len(a))
# np.random.shuffle(randomize)
# a = a[randomize]
# b = b[randomize]
# print(a)
# print(b)





# for peopleFolder in directories:
#     listImages = [x for x in os.listdir(peopleFolder) if x.endswith('.jpg')]
#
#     for image in listImages:
#         # this is the RGB image array.
#         face = misc.imread(peopleFolder + '/' + image)
#         faceRepo = np.append(faceRepo, [face],axis=0)
#
#         #assigning label
#
#
#
#
#         bit = peopleFolder[8:]
#         if bit == 'dulquer':
#             label = int(10)
#         elif bit == 'jyothika':
#             label = int(11)
#         elif bit == 'mammootty':
#             label = int(12)
#         elif bit == 'mohanlal':
#             label = int(13)
#         elif bit == 'nivin':
#             label = int(14)
#         elif bit == 'shobana':
#             label = int(15)
#         elif bit == 'suriya':
#             label = int(16)
#         elif bit == 'vijay':
#             label = int(17)
#         else:
#             pass
#         # label = peopleFolder[8:]
#
#         faceLabelRepo = np.append(faceLabelRepo, [label],axis=0)
#
#
# # # show dimens.
# # print(faceRepo.shape)
# # print(faceLabelRepo.shape)
# # #
# print(faceLabelRepo)