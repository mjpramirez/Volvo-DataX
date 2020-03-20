import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from keras.utils import to_categorical
import h5py
import numpy as np
import matplotlib.pyplot as plt
from conv3d_net_working import DenseNet3D_121
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


#FIRST GET INPUT DATA

def array_to_color(array, cmap="Oranges"):
  s_m = plt.cm.ScalarMappable(cmap=cmap)
  return s_m.to_rgba(array)[:,:-1]

def rgb_data_transform(data):
  data_t = []
  for i in range(data.shape[0]):
    data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
  return np.asarray(data_t, dtype=np.float32)
# Load the HDF5 data file
with h5py.File("./full_dataset_vectors.h5", "r") as hf:    
    # Split the data into training/test features/targets
    X_train = hf["X_train"][:]
    targets_train = hf["y_train"][:]
    X_test = hf["X_test"][:] 
    targets_test = hf["y_test"][:]
    # Determine sample shape
    sample_shape = (16, 16, 16, 3)
    # Reshape data into 3D format
    X_train = rgb_data_transform(X_train)
    X_test = rgb_data_transform(X_test)
    # Convert target vectors to categorical targets
    targets_train = to_categorical(targets_train).astype(np.integer)
    targets_test = to_categorical(targets_test).astype(np.integer)


#print('train',len(X_train),len(X_train[0]),len(X_train[0][0]),len(X_train[0][0][0]),len(X_train[0][0][0][0]))
#print('test',len(targets_train),len(targets_train[0]))

#THE BUILD THE MODEL
checkpoint = ModelCheckpoint("saved-weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model = DenseNet3D_121((16, 16, 16, 3))
#model.summary()

# Model configuration
batch_size = 100
no_epochs = 30
learning_rate = 0.001
no_classes = 10
validation_split = 0.2
verbosity = 1

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            metrics=['accuracy'])

# Fit data to model
'''
history = model.fit(X_train, targets_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split,callbacks=[checkpoint])
'''
#Visualize
model.load_weights("saved-weights.hdf5")

# Generate generalization metrics
score = model.evaluate(X_test, targets_test, verbose=0)
print('Test loss:',score[0],'/Test accuracy:',score[1])

# Plot history: Categorical crossentropy & Accuracy
'''
plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Model performance for 3D MNIST Keras Conv3D example')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
'''