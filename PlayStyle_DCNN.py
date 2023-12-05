import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Softmax, BatchNormalization, Dropout, Add, LeakyReLU
from keras.optimizers import Adam, SGD
from keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, random_rotation



def data_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield np.array(X[batch_indices]), np.array(y[batch_indices])


def prepare_input(moves):
    x = np.zeros((19, 19, 4))
    for move in moves:
        color = move[0]
        column = coordinates.get(move[2])
        row = coordinates.get(move[3])
        if color == 'B':
            x[row, column, 0] = 1
            x[row, column, 2] = 1
        elif color == 'W':
            x[row, column, 1] = 1
            x[row, column, 2] = 1
    if moves:
        last_move = moves[-1]
        if len(last_move) >= 4:
            last_move_column = coordinates.get(last_move[2])
            last_move_row = coordinates.get(last_move[3])
            if last_move_column is not None and last_move_row is not None:
                x[last_move_row, last_move_column, 3] = 1
    x[:, :, 2] = np.where(x[:, :, 2] == 0, 1, 0)
    return x

def load_training_data(file_path):
    df = open(file_path).read().splitlines()
    games = [i.split(',', 2)[-1] for i in df]
    game_styles = [int(i.split(',', 2)[-2]) for i in df]

    x_train = []
    for game in games:
        moves_list = game.split(',')
        x_train.append(prepare_input(moves_list))
    x_train = np.array(x_train)
    y_train = np.array(game_styles) - 1
    y_train_hot = tf.one_hot(y_train, depth=3)

    return x_train, y_train_hot.numpy()

def load_validation_data(file_path):
    df_val = open(file_path).read().splitlines()
    games_val = [i.split(',', 2)[-1] for i in df_val]
    game_styles_val = [int(i.split(',', 2)[-2]) for i in df_val]

    x_val = []
    for game in games_val:
        moves_list = game.split(',')
        x_val.append(prepare_input(moves_list))
    x_val = np.array(x_val)
    y_val = np.array(game_styles_val) - 1
    y_val_hot = tf.one_hot(y_val, depth=3)

    return x_val, y_val_hot.numpy()

def custom_rotation(image):
    angle = np.random.choice([90, 180, 270])
    return random_rotation(image, rg=angle, row_axis=0, col_axis=1, channel_axis=2)

# Data Augmentation: Adding custom rotation
datagen = ImageDataGenerator(
    preprocessing_function=custom_rotation,
    horizontal_flip=True,
    vertical_flip=True
)




#Training
#Simple DCNN6 + BN Model:

def create_model(fs):
    inputs = Input(shape=(19, 19, 4))


    x = Conv2D(kernel_size=7, filters=fs, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 殘差塊
    for _ in range(3):
        identity = x
        x = Conv2D(kernel_size=3, filters=fs, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(kernel_size=3, filters=fs, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, identity])
        x = ReLU()(x)
        x = Dropout(0.3)(x)

    # 最後的卷積層
    x = Conv2D(kernel_size=3, filters=fs, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Flatten()(x)
    x = Dense(fs, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(fs // 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inputs, x)
    opt = Adam(learning_rate=0.0002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model





save_weights = "./h5/playDCDNN7.h5"
save_JPG = "./JPG/playDCDNN6.jpg"

batch_size = 64
epochs = 200
filters_size = 64

df = open('./CSVs/play_style_train.csv').read().splitlines()
games = [i.split(',', 2)[-1] for i in df]
game_styles = [int(i.split(',', 2)[-2]) for i in df]

chars = 'abcdefghijklmnopqrs'
coordinates = {k: v for v, k in enumerate(chars)}
print(coordinates)

# Check how many samples can be obtained
n_games = 0
for game in games:
    n_games += 1
print(f"Total Games: {n_games}")

x = []
for game in games:
    moves_list = game.split(',')
    x.append(prepare_input(moves_list))
x = np.array(x)
y = np.array(game_styles)-1

print(x.shape, y.shape)
print(np.bincount(y))

y_hot = tf.one_hot(y, depth=3)


training_data_path = "./CSVs/play_style_train.csv"
validation_data_path = "./CSVs/Tutorial_play_style_train.csv"

x_train, y_train = load_training_data(training_data_path)
x_val, y_val = load_validation_data(validation_data_path)

#x_train, x_val, y_train, y_val = train_test_split(x, y_hot.numpy(), test_size=0.15)
datagen.fit(x_train)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, horizontal_flip=True, fill_mode="nearest")
# Data Augmentation: Adding custom rotation
datagen = ImageDataGenerator(preprocessing_function=custom_rotation)
datagen.fit(x_train)

model = create_model(filters_size)
model.summary()
#model.load_weights('./weights/ps//PlayStyle_DCNN6_LW2_64_BN_C_Adam00004_20_all_16_best_4_6478_07959.h5')

# construct the callback to save only the *best* model to disk
# based on the validation loss
checkpoint = ModelCheckpoint(save_weights, monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
# H = model.fit(x=x_train, y=y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val),
#              callbacks=callbacks, verbose=2)

train_data_generator = data_generator(x_train, y_train, batch_size)
validation_data_generator = data_generator(x_val, y_val, batch_size)


H = model.fit(
    x=train_data_generator,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)
# filter_size_DG_BN_C_Adam004_epochs_half_games_batch_size
#model.save('./DCNN6/model_playstyle_DCNN6_64_BN_C_Adam00004_40_all_8_last.h5')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on playstyle_DCNN6_LW3_64_BN_C_Adam00004_20_all_16")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(save_JPG)