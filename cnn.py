
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import dataset_creator
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])




# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
# test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')

train_labels=np.load("/home/santhosh/vs_code/labels.npy")
train_images=np.load("/home/santhosh/vs_code/dataset_array.npy")
#dataset_creator.dataset_creator("/home/santhosh/vs_code/dataset2")

# classifier.fit_generator(training_set,
#                          steps_per_epoch = 8000,
#                          epochs = 25,
#                          validation_data = test_set,
#                          validation_steps = 2000)

classifier.fit(train_images, train_labels, epochs=5)
classifier.save("classi.h5")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model

# Assuming you have a trained model saved as 'model.h5'
model = load_model('model.h5')

# Assuming you have test data X_test, y_test ready for evaluation
X_test = ...
y_test = ...

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate precision, recall, F1-score, specificity
report = classification_report(y_true, y_pred_classes, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()
specificity = tn / (tn + fp)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')

# Print evaluation results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("Specificity:", specificity)
print("ROC-AUC:", roc_auc)

