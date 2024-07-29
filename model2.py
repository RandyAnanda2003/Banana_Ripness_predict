import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
from tkinter import filedialog, Label

# Define paths (DEFINE UR PATH !)
train_dir = 'D:/Randy/semester 4 randy/citra digital/uas5/train2/'
val_dir = 'D:/Randy/semester 4 randy/citra digital/uas5/valid2/'
test_dir = 'D:/Randy/semester 4 randy/citra digital/uas5/test2/'
model_dir = 'D:/Randy/semester 4 randy/citra digital/uas5/'
model_path = os.path.join(model_dir, 'banana_ripeness_model3.h5')

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Instantiate data generators with minimal augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    fill_mode='nearest'
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(416, 416),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Check if model already exists
if os.path.exists(model_path):
    print("Loading the trained model...")
    model = load_model(model_path)
else:
    # Prepare the VGG19 model for transfer learning
    print("Preparing the VGG19 model...")
    num_classes = 6  # Number of categories

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(416, 416, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new classification layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='swish')(x)
    x = Dropout(0.5)(x)  # Adding dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    print("Compiling the model...")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    epochs = 13

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[early_stopping]
    )

    # Save the model
    print("Saving the model...")
    model.save(model_path)

# Function to predict ripeness of a banana image
def predict_ripeness(image_path):
    img = load_img(image_path, target_size=(416, 416))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = list(train_generator.class_indices.keys())[np.argmax(predictions)]
    return predicted_class

# Function to open file dialog and predict ripeness
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_ripeness = predict_ripeness(file_path)
        result_label.config(text=f"The predicted ripeness of the banana is: {predicted_ripeness}")

# Create the Tkinter window
root = tk.Tk()
root.title("Banana Ripeness Predictor")

# Create a label and button
label = Label(root, text="Select a banana image to predict its ripeness:")
label.pack(pady=20)

button = tk.Button(root, text="Open Image", command=open_file)
button.pack(pady=10)

result_label = Label(root, text="")
result_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
