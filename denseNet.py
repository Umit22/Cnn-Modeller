import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Veri kümesini yükleme ve ön işleme
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# Yoğun blok oluşturma
def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x

# Convolutional blok oluşturma
def conv_block(x, growth_rate):
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x1)
    x = tf.keras.layers.concatenate([x, x1])
    return x

# Geçiş katmanı oluşturma
def transition_layer(x, reduction):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), (1, 1), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

# Model oluşturma
def create_densenet():
    growth_rate = 12
    depth = 40  # Daha az yoğun blok sayısı
    num_dense_blocks = (depth - 4) // 6

    input_shape = (32, 32, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = Conv2D(2 * growth_rate, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = dense_block(x, num_dense_blocks, growth_rate)
    x = transition_layer(x, 0.5)
    x = dense_block(x, num_dense_blocks, growth_rate)
    x = transition_layer(x, 0.5)
    x = dense_block(x, num_dense_blocks, growth_rate)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Modeli derleme ve eğitim
def train_densenet():
    model = create_densenet()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=10, batch_size=128, verbose=1)
    return model

# Modeli eğitme ve değerlendirme
def evaluate_densenet():
    model = train_densenet()
    y_pred = model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
    acc = accuracy_score(y_test, y_pred_classes)
    print(f"DenseNet model doğruluğu: {acc}")

# DenseNet modelini eğitme ve değerlendirme
evaluate_densenet()
