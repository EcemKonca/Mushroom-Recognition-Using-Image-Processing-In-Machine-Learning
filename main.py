import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Unzip the file and extract its contents
zip_path = "C:/Users/ECEM/Downloads/archive (2).zip"
extract_to = 'data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

data_dir = os.path.join(extract_to, 'mushroom')

# Load and preprocess the dataset - Including Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False)

# Create the InceptionV3 model and freeze the top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in base_model.layers:
    layer.trainable = False

# Modifications in the Model Architecture and Applying Dropout
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Increase the number of neurons
x = Dropout(0.3)(x)  # Increase dropout rate
x = Dense(512, activation='relu')(x)  # Additional Dense layer
x = Dropout(0.3)(x)  # New dropout layer
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Adjust the Learning Rate Dynamically
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=20,  # You can increase the number of epochs
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          validation_steps=validation_generator.samples // validation_generator.batch_size,
          callbacks=[reduce_lr, early_stop, model_checkpoint])  # Add learning rate reduction callback

# Fine-Tuning: Freeze the first 249 layers and unfreeze the remaining
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

# Recompile the Model (After Fine-Tuning)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model (With Fine-Tuning)
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=20,  # Adjust the number of epochs for fine-tuning
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_steps=validation_generator.samples // validation_generator.batch_size,
                    callbacks=[reduce_lr, early_stop, model_checkpoint])

# Evaluate the Model on the Validation Set
validation_generator.reset()
y_pred = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
y_pred = np.round(y_pred).astype(int)

# Calculate the True Labels and Performance Metrics
y_true = validation_generator.classes[:len(y_pred)]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")
print(f"Validation F1 Score: {f1}")

# Data for Graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
lr = history.history['lr']

epochs_range = range(len(acc))

# Accuracy and Loss Graphs
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.show()

# Learning Rate Change Graph
plt.figure(figsize=(7, 5))
plt.plot(epochs_range, lr, label='Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.legend(loc='upper right')
plt.show()

# Confusion Matrix and Classification Report
predictions = model.predict(validation_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)
true_classes = validation_generator.classes

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'],
            yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(true_classes, predicted_classes, target_names=['Edible', 'Poisonous'])
print(report)

# Export the Keras model in SavedModel format
# model.save("my_model")
tf.saved_model.save(model, "my_model")
# Convert the model using TensorFlow Lite Converter
converter = tf.lite.TFLiteConverter.from_saved_model("my_model")  # or load the .h5 file
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # TF Lite native supported operations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

labels = ["edible", "poisonous"]  # Example label list
with open("labels.txt", "w") as f:
    f.write("\n".join(labels))
