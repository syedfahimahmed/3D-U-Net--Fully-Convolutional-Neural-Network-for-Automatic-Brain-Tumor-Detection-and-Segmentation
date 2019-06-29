import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import random as r
from keras.optimizers import Adam

import glob

from keras.models import Input, Model
from keras.layers import Conv3D, Concatenate, MaxPooling3D, Reshape
from keras.layers import UpSampling3D, Activation, Permute

import keras.backend as K


def create_data(src, mask, label=False, resize=(128,128,128)):
    files = glob.glob(src + mask, recursive=True)
    imgs = []
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            img[img > 0] = 1
            img = img.astype('float32')
        else:
            img = (img-img.mean()) / img.std()
        img = trans.resize(img, resize, mode='constant')
        imgs.append(img)
    name = 'Y_train' if label else 'X_train'
    np.save(name, np.array(imgs)[..., np.newaxis].astype('float32'))
    print('Saved', len(files), 'to', name)


def level_block_3d(k, dim, depth, factor, acti):
    if depth > 0:
        n = Conv3D(dim, 3, activation=acti, padding='same')(k)
        m = Conv3D(dim, 3, activation=acti, padding='same')(n)
        m = MaxPooling3D()(n)
        m = level_block_3d(m, int(factor*dim), depth-1, factor, acti)
        m = UpSampling3D()(m)
        m = Concatenate(axis=4)([n, m])
        m = Conv3D(dim, 3, activation=acti, padding='same')(m)
    return Conv3D(dim, 3, activation=acti, padding='same')(m)


def UNet_3D(img_shape, n_out=1, dim=64, depth=4, factor=2, acti='elu'):
    i = Input(shape=img_shape)
    o = level_block_3d(i, dim, depth, factor, acti)
    o = Conv3D(n_out, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


def f1_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def f1_loss(y_true, y_pred):
    return 1-f1_score(y_true, y_pred)


create_data('BRATS2015_Training/HGG/', '/**/*T1c*.mha', label=False, resize=(32,32,32))
create_data('BRATS2015_Training/HGG/', '/**/*OT*.mha', label=True, resize=(32,32,32))


x = np.load('/content/gdrive/My Drive/X_train.npy')
print('x: ', x.shape)
y = np.load('/content/gdrive/My Drive/Y_train.npy')
print('y:', y.shape)


import random as r
i = int(r.random() * x.shape[0])
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(x[i, int(x.shape[1]/2), :, :, 0])
plt.subplot(122)
plt.imshow(y[i, int(y.shape[1]/2), :, :, 0])
plt.show()


model = UNet_3D(x.shape[1:], dim=32, factor=2)
model.compile(optimizer=Adam(lr=0.0001), loss=f1_loss, metrics=['accuracy'])


from sklearn.cross_validation import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size = 0.2, random_state = 0)


from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    zoom_range=0.1,
    brightness_range=[0.8,1.2])

 	
datagen.fit(X_train)


history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=8), validation_data=(X_valid, Y_valid),steps_per_epoch=len(X_train)/8,epochs=300)
model.save_weights('demo_weights.h5')
pred = model.predict(x[:60])


import random as r
num = int(x.shape[1]/2)
for n in range(3):
    i = int(r.random() * pred.shape[0])
    plt.figure(figsize=(15,10))

    plt.subplot(131)
    plt.title('Input')
    plt.imshow(x[i, num, :, :, 0])

    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(y[i, num, :, :, 0])

    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(pred[i, num, :, :, 0])

    plt.show()


model.summary()


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Segmentation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid test'], loc='lower right')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Segmentation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid test'], loc='upper right')
plt.show()