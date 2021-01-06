import keras
from keras import layers
from keras.layers import Dense, MaxPooling2D, Conv2D, AveragePooling2D, Flatten, Concatenate, Input, \
    GlobalAveragePooling2D
from keras_applications import resnet_v2, nasnet
from keras.applications import mobilenet, resnet50, inception_v3, inception_resnet_v2, xception
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import math, os
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net-name', dest='net_name', type=str, default='mobilenet', help='Choose your nerwork!')
    parser.add_argument('--data', dest='data', type=str, default='./data/class', help='your data path!')
    parser.add_argument('--img-size', dest='img_size', type=int, default=224, help='input net size!')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8)
    parser.add_argument('--classes', dest='classes', type=int, default='3', help='class number!')
    parser.add_argument('--save-path', dest='save_path', type=str, default='./model', help='save model path!')
    args = parser.parse_args()
    return args


def load_model(net_name, img_size, classes):
    model = None
    if net_name in ["mobilenet", "Mobilenet"]:
        base_model = mobilenet.MobileNet(include_top=False, input_shape=((img_size, img_size, 3)))
        input = base_model.output
        x = GlobalAveragePooling2D()(input)
        output = Dense(classes, activation='softmax', use_bias=True)(x)
        model = Model(base_model.input, output)

    elif net_name in ['resnet50', 'Resnet50']:
        base_model = resnet50.ResNet50(include_top=False, input_shape=((img_size, img_size, 3)))
        input = base_model.output
        x = Flatten()(input)
        output = Dense(classes, activation='softmax')(x)
        model = Model(base_model.input, output)

    elif net_name in ['resnet50v2', 'resnet50_v2']:
        base_model = resnet_v2.ResNet50V2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet',
                                          backend=keras.backend, layers=keras.layers, models=keras.models,
                                          utils=keras.utils)
        input = base_model.output
        x = Flatten()(input)
        output = Dense(classes, activation='softmax')(x)
        model = Model(base_model.input, output)

    elif net_name in ['inception', 'Inception']:
        base_model = inception_v3.InceptionV3(include_top=False, input_shape=(img_size, img_size, 3))
        input = base_model.output
        x = GlobalAveragePooling2D()(input)
        output = Dense(classes, activation='softmax')(x)
        model = Model(base_model.input, output)

    elif net_name in ['inceptino_resnet_v2', 'inceptionresnetv2']:
        base_model = inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=(img_size, img_size, 3))
        input = base_model.output
        x = GlobalAveragePooling2D()(input)
        output = Dense(classes, activation='softmax')(x)
        model = Model(base_model.input, output)

    elif net_name in ['xception', 'Xception']:
        base_model = xception.Xception(include_top=False, input_shape=(img_size, img_size, 3))
        input = base_model.output
        x = GlobalAveragePooling2D()(input)
        output = Dense(classes, activation='softmax')(x)
        model = Model(base_model.input, output)

    elif net_name in ['nasnetmobile', 'nasnetmobilenet']:
        base_model = nasnet.NASNetMobile(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet',
                                         backend=keras.backend, layers=keras.layers, models=keras.models,
                                         utils=keras.utils)
        input = base_model.output
        x = Flatten()(input)
        output = Dense(3, activation='softmax')(x)
        model = Model(base_model.input, output)
    else:
        assert model, 'please input right net_name or make yourself net!'
    model.summary()
    return model


def load_data(img_path, input_size, batch):
    train_datagen = ImageDataGenerator(rotation_range=10,
                                       width_shift_range=0.01,
                                       height_shift_range=0.01,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       rescale=1 / 255.)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(img_path, 'train'),
        target_size=(input_size, input_size),
        batch_size=batch,
        shuffle=True,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(img_path, 'val'),
        target_size=(input_size, input_size),
        batch_size=batch,
        shuffle=True,
        class_mode='categorical'
    )

    return train_gen, test_gen


def train(model, train_gen, test_gen, epoches, save_path):
    print("starting train!")
    print("-" * 100)
    optimer = keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer=optimer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model_path = os.path.join(save_path, 'weights.epoch:{epoch:02d}_loss:{val_loss:.2f}.hdf5')
    # model_path = './checkpoint/weights.{epoch:02d}-{val_loss:.4f}-{val_categorical_accuracy:.4f}.hdf5'
    callback = keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True,
                                               save_weights_only=False)

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen),  # 訓練數據樣本數
        epochs=epoches,
        validation_data=test_gen,
        validation_steps=len(test_gen),  # 測試數據樣本數
        shuffle=True,
        callbacks=[callback]
    )


if __name__ == '__main__':
    args = parser_args()
    train_gen, test_gen = load_data(args.data, args.img_size, args.batch_size)  # 加載數據
    model = load_model(args.net_name, args.img_size, args.classes)  # 加載模型
    train(model, train_gen, test_gen, args.epochs, args.save_path)  # 訓練
