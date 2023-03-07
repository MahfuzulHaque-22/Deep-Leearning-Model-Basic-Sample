import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate random data
X_train = tf.random.normal((1000, 10))
Y_train = tf.random.uniform((1000, 1), maxval=2, dtype=tf.int32)
X_test = tf.random.normal((100, 10))
Y_test = tf.random.uniform((100, 1), maxval=2, dtype=tf.int32)

# Create a deep learning model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
