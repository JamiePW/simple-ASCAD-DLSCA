from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop


def pick_SOAT(data, leakage_model, length, metric, loss, model='MLP', model_size=64):
    if model == 'CNN':
        if 'ASCAD' in data and leakage_model == 'HW':
            batch_size = 50
            epoch = 50
            return ascad_f_hw_cnn(length, metric, loss), batch_size, epoch
        elif 'ASCAD' in data and leakage_model == 'ID':
            batch_size = 128
            epoch = 50
            return ascad_f_id_cnn(length, metric, loss), batch_size, epoch
    elif model == 'MLP':
        if 'ASCAD' in data and leakage_model == 'HW':
            batch_size = 128
            epoch = 10
            return ascad_f_hw_mlp(length, metric, loss), batch_size, epoch
        elif 'ASCAD' in data and leakage_model == 'ID':
            batch_size = 32
            epoch = 10
            return ascad_f_id_mlp(length, metric, loss), batch_size, epoch


################ SOAT MODELS#####################
# epoch 50
def ascad_f_hw_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(16, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model


def ascad_f_hw_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(496, activation='relu')(img_input)
    x = Dense(496, activation='relu')(x)
    x = Dense(136, activation='relu')(x)
    x = Dense(288, activation='relu')(x)
    x = Dense(552, activation='relu')(x)
    x = Dense(408, activation='relu')(x)
    x = Dense(232, activation='relu')(x)
    x = Dense(856, activation='relu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model


def ascad_f_id_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    adam = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer=adam, metrics=metric)
    model.summary()
    return model


def ascad_f_id_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(160, activation='relu')(img_input)
    x = Dense(160, activation='relu')(x)
    x = Dense(624, activation='relu')(x)
    x = Dense(776, activation='relu')(x)
    x = Dense(328, activation='relu')(x)
    x = Dense(968, activation='relu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model
