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

# Extract zip file
zip_path = "C:/Users/ECEM/Downloads/archive (2).zip"  # Path to zip file
extract_to = 'data'  # Target folder to extract

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

data_dir = os.path.join(extract_to, 'mushroom')  # Path to extracted files

# Load and preprocess dataset - including data augmentation
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

# Modify model architecture and apply Dropout
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Increase number of neurons
x = Dropout(0.3)(x)  # Increase Dropout rate
x = Dense(512, activation='relu')(x)  # Additional Dense layer
x = Dropout(0.3)(x)  # New Dropout layer
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Dynamic learning rate adjustment
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=20,  # Increase the number of epochs if needed
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          validation_steps=validation_generator.samples // validation_generator.batch_size,
          callbacks=[reduce_lr, early_stop, model_checkpoint])  # Add learning rate reduction callback

# Fine-tuning: freeze the first 249 layers and unfreeze the rest
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

# Re-compile the model (after fine-tuning)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (with fine-tuning)
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=20,  # Adjust number of epochs for fine-tuning
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_steps=validation_generator.samples // validation_generator.batch_size,
                    callbacks=[reduce_lr, early_stop, model_checkpoint])

# Evaluate the model on the validation set
validation_generator.reset()
y_pred = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
y_pred = np.round(y_pred).astype(int)

# Calculate true labels and performance metrics
y_true = validation_generator.classes[:len(y_pred)]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")
print(f"Validation F1 Score: {f1}")

# Data for plotting graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
lr = history.history['lr']

epochs_range = range(len(acc))

# Plot Accuracy and Loss graphs
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

# Plot Learning Rate Change
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

# Export Keras model in SavedModel format
tf.saved_model.save(model, "my_model")
# Convert model using TensorFlow Lite Converter
converter = tf.lite.TFLiteConverter.from_saved_model("my_model")  # or load .h5 file
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # TF Lite native supported operations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
with open("model5.tflite", "wb") as f:
    f.write(tflite_model)

labels = ["edible", "poisonous"]  # Example label list
with open("labels.txt", "w") as f:
    f.write("\n".join(labels))
