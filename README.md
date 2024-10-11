# Image-sentiment-classification-using-CNN

## 1. Setting Up the Environment

Start by installing the required libraries and dependencies:

\`\`\`bash
%pip install tensorflow opencv-python matplotlib
\`\`\`

- Imported libraries include TensorFlow, OpenCV for image processing, and Matplotlib for visualization.

---

## 2. Data Loading and Preprocessing

We load the image dataset using TensorFlow's \`image_dataset_from_directory\` function, automatically labeling the images.

\`\`\`python
data = tf.keras.utils.image_dataset_from_directory('data')
\`\`\`

### Data Cleaning
We ensure all images meet specific format requirements (\`jpeg\`, \`jpg\`, \`bmp\`, \`png\`). Unsupported images are automatically removed:

\`\`\`python
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        img = cv2.imread(image_path)
        tip = imghdr.what(image_path)
        if tip not in image_exts:
            os.remove(image_path)
\`\`\`

### Data Normalization
All images are normalized by dividing pixel values by 255, ensuring better model training:

\`\`\`python
data = data.map(lambda x, y: (x / 255, y))
\`\`\`

---

## 3. Data Splitting

We split the dataset into training (70%), validation (20%), and test (10%) sets:

\`\`\`python
train_size = int(len(data) * .7)
test_size = int(len(data) * .1)
val_size = int(len(data) * .2)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
\`\`\`

---

## 4. Building the CNN Model

A sequential model is built with multiple convolutional layers, max pooling layers, and dense layers. The model is compiled with the Adam optimizer and binary cross-entropy loss for binary classification tasks.

\`\`\`python
model = Sequential([
    Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
\`\`\`

---

## 5. Training the Model

We train the model for 20 epochs using TensorBoard for visualization:

\`\`\`python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
\`\`\`

### Performance Visualization
Loss and accuracy are plotted after training for both the training and validation sets:

\`\`\`python
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.show()

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.show()
\`\`\`

---

## 6. Model Evaluation

We evaluate the trained model using Precision, Recall, and Accuracy metrics on the test set:

\`\`\`python
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    predictions = model.predict(X)
    pre.update_state(y, predictions)
    re.update_state(y, predictions)
    acc.update_state(y, predictions)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
\`\`\`

---

## 7. TensorBoard Logs

Training performance can be monitored in real-time using TensorBoard:

\`\`\`bash
tensorboard --logdir logs
\`\`\`

---

### Summary

This project successfully demonstrates how to:
* Load, clean, and preprocess image data.
* Build a CNN using TensorFlow and Keras.
* Train and evaluate the model using accuracy, precision, and recall metrics.
* Visualize performance using Matplotlib and TensorBoard.

Feel free to explore and adapt the model to different datasets for image classification tasks!
"""