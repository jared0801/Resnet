import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.callbacks import TensorBoard
import keras.backend as K
from os import path

from rnmodel import ResNet50
from helpers import convert_to_one_hot, init_data

K.set_image_data_format('channels_last')

# Continue training the network.
# When train = True, the saved model will be overwritten. Change the model_path if you would like to save multiple models.
# Note that training on the same dataset too much will result in overfitting.
train = False

# Path for where to save trained models
model_path = 'models/kerasresnet.h5'

# Number of epochs, or iterations over the whole dataset, to train
# Note: only used when train = True
epochs = 20

# Image to test against
test_img = 'two.jpg'


def main():
    model = ResNet50()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_train, Y_train, X_test, Y_test, classes = init_data()

    if path.exists(model_path):
        model = load_model(model_path)
    if train:
        tb_callback = TensorBoard('./logs', update_freq=1)
        model.fit(X_train, Y_train, epochs = epochs, batch_size = 32, validation_data=(X_test, Y_test), callbacks=[tb_callback])
        model.save(model_path)

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    img_path = 'two.jpg'
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    guess = np.argmax(preds)
    print('Predictions:', preds)
    print('Best guess: ', guess)


if __name__ == '__main__':
    main()
