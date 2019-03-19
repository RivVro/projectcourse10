from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_label = []
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('C:\\Users\\rivvr\\Documents\\Classifier\\labeled_train',target_size = (64, 64),batch_size = 1,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\rivvr\\Documents\\Classifier\\labeled_test',target_size = (64, 64),batch_size = 1,class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch=len(training_set), epochs = 25,validation_data = test_set,validation_steps = 2000)
print("klaar!!!")



'''
i = 0
while i < 503:
    train_label.append("0")
    i = i + 1
print(train_label)
classifier.fit(X, train_label, epochs=31)
'''




scores = classifier.evaluate(training_set, steps_per_epoch = 50)



import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\rivvr\\Documents\\Classifier\\RontTest\\00008763_001.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'effusion'
else:
    prediction = 'infiltration'