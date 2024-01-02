# Cats vs. Dogs Image Classification using Convolutional Neural Networks (CNN)

## Overview
This project aims to build a deep-learning model to classify images of cats and dogs using Convolutional Neural Networks (CNN). The dataset utilized is sourced from the Dogs vs. Cats dataset available on Kaggle.

<img src="https://github.com/BK-KAVIYA/Zaara.lk/blob/main/PHOTO/logo/zara.png" alt="Data set">

## Key Steps
1. **Data Collection and Preprocessing:** 
   - The Kaggle dataset is downloaded using the Kaggle API and divided into training and validation sets.
   - Images are resized to 256x256 pixels and normalized for enhanced model performance.
   - Code snippet:
     ```python
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     ```

2. **Model Architecture:** 
   - A CNN architecture is employed, comprising convolutional, pooling, batch normalization, and dense layers with ReLU activation.
   - The model uses a sigmoid activation for binary classification.
   - Code snippet:
     ```python
      model = Sequential()

      model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
      
      model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
      
      model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
      
      model.add(Flatten())
      
      model.add(Dense(128,activation='relu'))
      model.add(Dropout(0.1))
      model.add(Dense(64,activation='relu'))
      model.add(Dropout(0.1))
      model.add(Dense(1,activation='sigmoid'))
     ```

3. **Model Training:** 
   - The CNN model is compiled with Adam optimizer and binary cross-entropy loss.
   - Training is conducted for 10 epochs, monitoring accuracy and loss on the validation set.
   - Code snippet:
     ```python
     model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
     ```

4. **Performance Evaluation:** 
   - The training and validation accuracies and losses are visualized using matplotlib.
   - Additionally, a sample test image is loaded and passed through the trained model for prediction.
   - Code snippet:
     ```python
      import matplotlib.pyplot as plt

      plt.plot(history.history['accuracy'],color='red',label='train')
      plt.plot(history.history['val_accuracy'],color='blue',label='validation')
      plt.legend()
      plt.show()
     ```

5. **Mitigating Overfitting:** 
   - Strategies to address overfitting, such as adding more data, data augmentation, regularization techniques (L1/L2), dropout, batch normalization, and reducing model complexity, are suggested for implementation.


## Next Steps
- Explore and implement techniques to reduce overfitting, improving model generalization.
- Experiment with hyperparameters, model architectures, and augmentation strategies for enhanced performance.
- Deploy the trained model for real-time predictions or integrate it into applications for cat vs. dog image classification tasks.

This project serves as a foundation for understanding and implementing image classification using CNNs, with potential for further enhancements and applications in various domains.

---

<img src="https://github.com/BK-KAVIYA/Image-Classification-using-Convolutional-Neural-Networks-/blob/main/Images/cat-test.png" alt="cat testing">
*Sample Cat Image*

<img src="https://github.com/BK-KAVIYA/Image-Classification-using-Convolutional-Neural-Networks-/blob/main/Images/dos-test.png" alt="dog testing">
*Sample Dog Image*

 
