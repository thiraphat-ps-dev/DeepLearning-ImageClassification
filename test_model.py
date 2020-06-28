import matplotlib.pylab as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from keras.preprocessing import image
import numpy as np
import os
batch_size = 24
image_height = 28
image_width = 28
model = Sequential()
model.add(Conv2D(
    filters=32, kernel_size=(3, 3),
    input_shape=(image_height, image_width, 3),
    activation='relu', padding='same'
))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.load_weights('model.h5') 
img = image.load_img('11743.jpg',
                     target_size=(image_width, image_height))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
pd = model.predict_classes(img_tensor)
catdog = {'cat': 0, 'dog': 1}
for idx, i in enumerate(catdog):
    print(idx, i) if pd[0] == idx else ''

