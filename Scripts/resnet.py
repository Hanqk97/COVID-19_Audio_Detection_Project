import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load your data
# Assuming X_train, X_val, X_test are of shape (num_samples, sequence_length)
# You will need to adjust based on your actual data pipeline

# Example dummy data for demonstration
X_train = np.random.rand(1000, 128, 1)  # Replace with actual data loading
X_val = np.random.rand(200, 128, 1)
X_test = np.random.rand(200, 128, 1)
y_train = np.random.randint(0, 2, 1000)  # Binary classification example
y_val = np.random.randint(0, 2, 200)
y_test = np.random.randint(0, 2, 200)

# Reshape data to match ResNet input expectations (height, width, channels)
# Example: Expand from (128, 1) to (128, 128, 1) for 2D CNN compatibility
X_train = np.expand_dims(X_train, axis=-1)  # (num_samples, sequence_length, 1, 1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Repeat the channel dimension to make the input (sequence_length, sequence_length, channels)
X_train = np.repeat(X_train, 128, axis=2)
X_val = np.repeat(X_val, 128, axis=2)
X_test = np.repeat(X_test, 128, axis=2)

# Build the ResNet50 model with modified input shape
input_shape = (128, 128, 1)  # Adjust to match expanded dimensions
base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)  # Set weights=None for training from scratch

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Example of a fully connected layer
predictions = Dense(1, activation='sigmoid')(x)  # Adjust for binary classification

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint("best_resnet_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train the model
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"ResNet Test Accuracy: {test_accuracy * 100:.2f}%")
