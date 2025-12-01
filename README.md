# MNIST Handwritten Digit Recognizer using CNN

## 1. Project Overview
This project builds a Convolutional Neural Network (CNN) to classify handwritten digit images (0–9) from the MNIST dataset. [web:49][web:59]  
It is intended as a portfolio project for demonstrating basic deep learning and computer vision skills.

**Key points:**
- **Dataset:** MNIST (60,000 training images, 10,000 test images, 28×28 grayscale). [web:49][web:59]
- **Frameworks:** TensorFlow/Keras, NumPy, Matplotlib, Seaborn, Scikit-learn. [web:49][web:59]
- **Result:** Test accuracy ≈ **95.1%** on the held‑out test set.

---

## 2. Dataset

The MNIST dataset is a classic benchmark for handwritten digit recognition and is available directly through `tensorflow.keras.datasets`. [web:49][web:59]  
Each sample is a 28×28 grayscale image of a single digit labeled from 0 to 9.

---

## 3. Model Architecture

The CNN model used in this project has the following structure (similar to common MNIST CNN examples on GitHub). [web:86][web:87][web:95]

- Conv2D layer: 32 filters, 3×3 kernel, ReLU activation  
- MaxPooling2D layer: 2×2 pool size  
- Conv2D layer: 64 filters, 3×3 kernel, ReLU activation  
- MaxPooling2D layer: 2×2 pool size  
- Flatten layer  
- Dense layer: 64 units, ReLU activation  
- Output Dense layer: 10 units, softmax activation  

The model is trained with the Adam optimizer and `sparse_categorical_crossentropy` loss for multi‑class classification. [web:49][web:59]

---

## 4. Training and Evaluation

- **Preprocessing:**
  - Normalize pixel values to the range \([0, 1]\).  
  - Reshape images to \((28, 28, 1)\) to add the channel dimension. [web:49][web:59]

- **Training setup:**
  - Epochs: 5  
  - Batch size: 128  
  - Validation split: 0.1  

- **Results:**
  - Test accuracy: **≈ 95.1%**  
  - Test loss: **≈ 0.17**  
  - A confusion matrix and classification report are generated to analyze per‑class performance. [web:39][web:41]

---

## 5. How to Run

1. Open the provided Google Colab notebook (`mnist_handwritten_digit_recognizer.ipynb`). [web:48][web:49]  
2. Run all cells sequentially (`Runtime → Run all`).  
3. At the end of the notebook, you will see:
   - Training/validation curves in the logs  
   - Test accuracy and loss  
   - Confusion matrix heatmap  
   - Sample images with predicted and true labels

---

## 6. Future Improvements

Possible next steps inspired by common MNIST CNN projects: [web:88][web:92][web:95]

- Train for more epochs and tune learning rate.  
- Add regularization (Dropout, Batch Normalization).  
- Use data augmentation (rotation, shift, zoom) to improve generalization.  
- Experiment with deeper architectures or apply the same pipeline to Fashion‑MNIST. [web:97]
