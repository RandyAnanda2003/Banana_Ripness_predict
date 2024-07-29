import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation      
from tensorflow.keras.utils import get_custom_objects

# Define MISH activation function
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

get_custom_objects().update({'mish': Activation(mish)})

# Define paths
train_dir = 'C:/Users/Lemonilo/Documents/pcd/dataset/train'
val_dir = 'C:/Users/Lemonilo/Documents/pcd/dataset/valid'
test_dir = 'C:/Users/Lemonilo/Documents/pcd/dataset/test'
model_dir = 'C:/Users/Lemonilo/Documents/pcd/dataset/'
model_path = os.path.join(model_dir, 'banana_ripeness_model4.h5')

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Instantiate data generators with minimal augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    fill_mode='nearest'  # Mengisi piksel yang hilang dengan nilai piksel terdekat
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

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
    num_classes = 7  # Number of categories

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(416, 416, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new classification layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='mish')(x)  # Use MISH activation here
    x = Dropout(0.5)(x)  # Adding dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    print("Compiling the model...")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    epochs = 10

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

# # Evaluate the model
# print("Evaluating the model...")
# test_loss, test_accuracy = model.evaluate(
#     test_generator, 
#     steps=test_generator.samples // test_generator.batch_size
# )
# print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

# Function to predict ripeness of a banana image
# def predict_ripeness(image_path):
#     img = load_img(image_path, target_size=(416, 416))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     predictions = model.predict(img_array)
#     predicted_class = list(train_generator.class_indices.keys())[np.argmax(predictions)]
#     return predicted_class

# Example usage of the prediction function
# image_path = 'C:/Users/Lemonilo/Documents/pcd/uji/pisang2.jpg'  # Change this to your image path
# predicted_ripeness = predict_ripeness(image_path)
# print(f"The predicted ripeness of the banana is: {predicted_ripeness}")

def predict_ripeness(image_paths):
    predictions = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(416, 416))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = list(train_generator.class_indices.keys())[np.argmax(prediction)]
        predictions.append((image_path, predicted_class))
    return predictions

# Example usage of the prediction function for multiple images
image_paths = [
    'C:/Users/Lemonilo/Documents/pcd/uji/pisang.jpg',
    'C:/Users/Lemonilo/Documents/pcd/uji/pisang2.jpg',
    'C:/Users/Lemonilo/Documents/pcd/uji/pisang3.jpg',
    'C:/Users/Lemonilo/Documents/pcd/uji/pisang4.jpg',
    'C:/Users/Lemonilo/Documents/pcd/uji/pisang6.jpg'
]

predicted_ripeness = predict_ripeness(image_paths)
for image_path, ripeness in predicted_ripeness:
    print(f"The predicted ripeness of the banana in {image_path} is: {ripeness}")