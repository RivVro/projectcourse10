
from PIL import Image
import numpy as np
import os
import imageio
from keras.models import Sequential

with open ('C:\\Users\\rivvr\\Documents\\informatica\\rontgen\labels.csv') as f:
    f = f.readlines()


Images = []
Label = []
for line in f:
    kolom = line.split(";")
    Images.append(kolom[0].split('.')[0])
    Label.append(kolom[1])

EffusionIndex = [j for j, s in enumerate(Label) if "Effusion" in s]
effusion = []
i = 0
while i < len(EffusionIndex):
    effusion.append("effusion")
    i += 1

PneumoniaIndex = [k for k, s in enumerate(Label) if "Pneumonia" in s]
pneumonia = []
j = 0
while j < len(PneumoniaIndex):
    pneumonia.append("pneumonia")
    j += 1

InfiltrationIndex = [l for l, s in enumerate(Label) if "Infiltration" in s]
infiltration = []
k = 0
while k < len(InfiltrationIndex):
    infiltration.append("infiltration")
    k += 1

EffusionImages = [Images[i] for i in EffusionIndex]
PneumoniaImages = [Images[i] for i in PneumoniaIndex]
InfiltrationImages = [Images[i] for i in InfiltrationIndex]

efdict = dict(zip(EffusionImages,effusion))
pneudict = dict(zip(PneumoniaImages,pneumonia))
infdict = dict(zip(InfiltrationImages,infiltration))

dictionary = {}
for e in [efdict,pneudict,infdict]:
    dictionary.update(e)

sickness = dictionary.values()
sickness_set = set(sickness)
counting_dict = {}
for i in sickness_set:
    counting_dict[i]=0

#for img in os.listdir('C:\\Users\\rivvr\\Documents\\datamining\\Rontgen\\NodigRontgen'):
#    imgName = img.split('.')[0]
#    print(imgName)
#    label = dictionary[str(imgName)]
#    counting_dict[label]+=1
#    path = os.path.join('C:\\Users\\rivvr\\Documents\\datamining\\Rontgen\\NodigRontgen\\', img)
#    saveName = 'C:\\Users\\rivvr\\Documents\\datamining\\Rontgen\\labeled_train\\' + label + '-' + str(counting_dict[label]) + '.png'
#    image_data = np.array(Image.open(path))
#    imageio.imwrite(saveName, image_data)

def label_img(name):
    word_label = name.split('-')[0]
    if word_label == "effusion": return np.array([1,0,0])
    if word_label == "infiltration": return np.array([0,1,0])
    elif word_label == "pneumonia": return np.array([0,0,1])
#werkt

model = Sequential()

