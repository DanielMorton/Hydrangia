from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

img_height = 299
img_width = 299
img_channels = 3
img_dim = (img_height, img_width, img_channels)

input_tensor = Input(shape=img_dim)
base_model = Xception(weights='imagenet',
                      include_top=False,
                      input_shape=img_dim)

bn = BatchNormalization()(input_tensor)
x = base_model(bn)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='sigmoid',
               kernel_regularizer=l2(0.1))(x)
model = Model(input_tensor, output)

path = '/Users/dmorton/Hydrangia/'
batch_size = 32

trainGen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        preprocessing_function=preprocess_input)

validGen = ImageDataGenerator(preprocessing_function=preprocess_input)

trainBatches = trainGen.flow_from_directory(path+'train', target_size=(img_height, img_width),
                                            class_mode='categorical', shuffle=True, batch_size=batch_size)

valBatches = validGen.flow_from_directory(path+'valid', target_size=(img_height, img_width),
                                          class_mode='categorical', shuffle=False, batch_size=2 * batch_size)


best_model_file = path + 'XC-{}x{}.h5'.format(img_height, img_width)

callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1, min_delta=1e-4),
             ModelCheckpoint(filepath=best_model_file, verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto')]

model.compile(optimizer=Adam(lr=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])


best_model = ModelCheckpoint(filepath=best_model_file,
                             monitor='val_loss', verbose=1, save_best_only=True)


model.fit_generator(generator=trainBatches,
                    verbose=1,
                    steps_per_epoch=trainBatches.n/trainBatches.batch_size,
                    epochs=100,
                    validation_data=valBatches,
                    validation_steps=valBatches.n/valBatches.batch_size,
                    callbacks=callbacks)
