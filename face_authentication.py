import cv2
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display/image_collect")
mode = ap.parse_args().mode

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    # axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    # axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy']) / 10))
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) / 10))

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1), len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()


train_dir = 'data/train'
#test_dir = 'data/test'
validate_dir = 'data/validate'

train_datagen = ImageDataGenerator(rescale = 1/255)
val_datagen = ImageDataGenerator(rescale = 1/255)

num_train = 1530  # enter number of images for training
num_val = 627  # enter number of images for validation
batch_size = 10
num_epoch = 3

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size =(88,168),
    batch_size = batch_size,
    color_mode='grayscale',
    class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    validate_dir,
    target_size = (88,168),
    color_mode = 'grayscale',
    class_mode = 'binary'
)

model = Sequential()

model.add(Conv2D(32, kernel_size= (3,3), activation = 'relu', input_shape=(88,168,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


if mode=="image_collect":
    # Define the directory where you want to save the detected face images
    output_directory = "D:/sem 7/csd/face images"  ## Create a folder names face images and give folder path in the double qoutes
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera, change if you have multiple cameras

    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Set the frame rate and calculate the total number of frames to capture for 5 seconds
    frame_rate = 30  # Adjust this as needed (e.g., 30 frames per second)
    duration = 2  # Duration in seconds
    total_frames = int(frame_rate * duration)

    # Capture images for 5 seconds
    start_time = time.time()
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Save the detected face images
        for (x, y, w, h) in faces:
            face_image = frame[y:y + (h//2)+2, x:x + w]
            image_filename = os.path.join(output_directory, f"face_{i:04d}.jpg")
            cv2.imwrite(image_filename, face_image)

        # Calculate and display the elapsed time and remaining time
        elapsed_time = time.time() - start_time
        remaining_time = duration - elapsed_time
        timer_text = f"Elapsed Time: {elapsed_time:.2f} seconds | Remaining Time: {remaining_time:.2f} seconds"

        # Add the timer text to the frame
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Captured Frame", frame)

        # Break the loop if 'q' is pressed or the 5 seconds are up
        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= duration:
            break

        # Sleep to control the frame rate
        time.sleep(1 / frame_rate)

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
elif mode == "train":
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  #optimizer = RMSprop(learning_rate =0.001),
                  metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        steps_per_epoch = num_train//batch_size,
        epochs = num_epoch,
        validation_data = validation_generator,
        validation_steps = num_val // batch_size
    )
    plot_model_history(model_info)
    model.save_weights('model2.h5')
elif mode == 'start':
    model.load_weights('model1.h5')

    cv2.ocl.setUseOpenCL(False)

    verif = {1:"undefined user",
             0: "vishnu"}
    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find Haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            #roi_gray = gray[y:y + h, x:x + w]
            roi_gray = frame[y:y + (h//2)+2, x:x + w]
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (168,88)), -1), 0)
            prediction = model.predict(cropped_img)
            #maxindex = int(np.argmax(prediction))
            val = " "
            if prediction == 1:
                val =  "defined user"
            else:
                val = 'undefined user'
            cv2.putText(frame, val, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            print(prediction[0])
        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


