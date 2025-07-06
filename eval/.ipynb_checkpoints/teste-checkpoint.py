import tensorflow as tf

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Save weights
model.save_weights('test_weights.h5')

# Load weights
model.load_weights('test_weights.h5')
