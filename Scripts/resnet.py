import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your data
# Replace with actual data loading pipeline
X_train = np.random.rand(1000, 128, 1)
X_val = np.random.rand(200, 128, 1)
X_test = np.random.rand(200, 128, 1)
y_train = np.random.randint(0, 2, 1000)
y_val = np.random.randint(0, 2, 200)
y_test = np.random.randint(0, 2, 200)

# Reshape data for ResNet input
X_train = np.repeat(np.expand_dims(X_train, axis=-1), 128, axis=2)  # (num_samples, 128, 128, 1)
X_val = np.repeat(np.expand_dims(X_val, axis=-1), 128, axis=2)
X_test = np.repeat(np.expand_dims(X_test, axis=-1), 128, axis=2)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build the ResNet50 model using pretrained weights
input_shape = (128, 128, 1)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))  # 3 channels needed for ImageNet weights

# Modify the input to have 3 channels by repeating the 1 channel 3 times
X_train = np.repeat(X_train, 3, axis=-1)
X_val = np.repeat(X_val, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for training stability
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint("best_resnet_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# Train the model with augmentation
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"ResNet Test Accuracy: {test_accuracy * 100:.2f}%")
