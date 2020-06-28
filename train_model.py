import matplotlib.pylab as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
batch_size = 24
image_height = 28
image_width = 28

train_data_dir = 'catdog/train'
validation_data_dir = 'catdog/validation'

datagen = ImageDataGenerator(rescale=1./255,)
train_generator = datagen.flow_from_directory(

    directory=train_data_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height),
    class_mode='binary'
)
validation_generator = datagen.flow_from_directory(

    directory=train_data_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height),
    class_mode='binary'
)

label_map = (train_generator.class_indices)
print(label_map)

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

nb_epoch = 30
history = model.fit_generator(train_generator,
                              epochs=nb_epoch,
                              validation_data=validation_generator,
                              verbose=1,
                              steps_per_epoch=10,
                              validation_steps=20,
                              )

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('model.h5')

score = model.evaluate(validation_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
