import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Add
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Veri kümesini yükleme ve ön işleme
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)


# Model oluşturma
def create_resnet_model():
    def residual_block(x, filters, kernel_size=3, strides=1):
        y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
        y = BatchNormalization()(y)

        if strides > 1:
            x = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)

        out = Add()([x, y])
        out = Activation('relu')(out)
        return out

    input_shape = (32, 32, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    num_res_blocks = 3
    filters = 64
    for i in range(num_res_blocks):
        x = residual_block(x, filters)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    num_res_blocks = 4
    filters *= 2
    for i in range(num_res_blocks):
        strides = 2 if i == 0 else 1
        x = residual_block(x, filters, strides=strides)

    num_res_blocks = 6
    filters *= 2
    for i in range(num_res_blocks):
        strides = 2 if i == 0 else 1
        x = residual_block(x, filters, strides=strides)

    num_res_blocks = 3
    filters *= 2
    for i in range(num_res_blocks):
        strides = 2 if i == 0 else 1
        x = residual_block(x, filters, strides=strides)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Modeli derleme ve eğitim
def train_resnet_model():
    model = create_resnet_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=10, batch_size=128, verbose=1)
    return model


# Modeli eğitme ve değerlendirme
def evaluate_resnet_model():
    model = train_resnet_model()
    y_pred = model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
    acc = accuracy_score(y_test, y_pred_classes)
    print(f"ResNet model doğruluğu: {acc}")


# ResNet modelini eğitme ve değerlendirme
evaluate_resnet_model()
