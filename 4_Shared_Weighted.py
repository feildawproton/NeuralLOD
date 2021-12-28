from matplotlib import image
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf

def define_model(width = 128, drop_ratio = 0.1):
	initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 1.0)
	#hard coded input shape because it will always be this
	input = tf.keras.layers.Input(shape = (2,))
	
	layer_1 = tf.keras.layers.Dense(width, activation = "relu")(input)
	drop_1 = tf.keras.layers.Dropout(drop_ratio)(layer_1)
	
	layer_2 = tf.keras.layers.Dense(width, activation = "relu")(drop_1)
	drop_2 = tf.keras.layers.Dropout(drop_ratio)(layer_2)
	
	layer_3 = tf.keras.layers.Dense(width, activation = "relu")(drop_2)
	drop_3 = tf.keras.layers.Dropout(drop_ratio)(layer_3)
	
	layer_4 = tf.keras.layers.Dense(width, activation = "relu")(drop_3)
	drop_4 = tf.keras.layers.Dropout(drop_ratio)(layer_4)
	
	
	shared_out = tf.keras.layers.Dense(1, activation = "relu")
	
	output_1 = shared_out(drop_1)
	output_2 = shared_out(drop_2)
	output_3 = shared_out(drop_3)
	output_4 = shared_out(drop_4)

	return tf.keras.Model(inputs = input, outputs = [output_1, output_2, output_3, output_4])

#adjustments made for multiple outputs
image		= image.imread(os.path.join("shapes", "leto.png"))
image		= np.expand_dims(image, axis = -1)
print(image.shape)
img_repeat	= np.repeat(image, 4, axis = -1)
y_flat		= np.reshape(img_repeat, (img_repeat.shape[0]*img_repeat.shape[1], img_repeat.shape[2]))


#the input data X
x1 = np.arange(0.0, 1.0, 1/image.shape[0])
x2 = np.arange(0.0, 1.0, 1/image.shape[1])
xx1, xx2 = np.meshgrid(x1, x2)
xx1 = np.expand_dims(xx1, axis = -1)
xx2 = np.expand_dims(xx2, axis = -1)
X = np.concatenate((xx1, xx2), axis = -1)
#and then flatten
X_flat = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))


model = define_model()
model.summary()

#use default learning rate of 0.001 for Adam optimizer
model.compile(loss = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()], loss_weights = [.25, 1, 4, 16], optimizer = tf.keras.optimizers.Adam(), metrics = ["mse"])
epochs = 32768 #16384
batch_size = 4096
history = model.fit(X_flat, y_flat, epochs = epochs, batch_size = batch_size, shuffle = True)

#now evaluate
x1_test = np.arange(0.0, 1.0, 1/256)
x2_test = np.arange(0.0, 1.0, 1/256)
xx1_test, xx2_test = np.meshgrid(x1_test, x2_test) 
xx1_test = np.expand_dims(xx1_test, axis = -1)
xx2_test = np.expand_dims(xx2_test, axis = -1)
X_test = np.concatenate((xx1_test, xx2_test), axis = -1)

X_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
print(X_test.shape)
predictions = model.predict(X_test)
print(len(predictions))

#show stuffconda activ


fig, ax = plt.subplots(nrows = 4, ncols = 2)


ax[0,0].imshow(np.reshape(predictions[0], (256,256,1)))
ax[0,1].plot(history.history["dense_4_mse"])


ax[1,0].imshow(np.reshape(predictions[1], (256,256,1)))
ax[1,1].plot(history.history["dense_4_1_mse"])


ax[2,0].imshow(np.reshape(predictions[2], (256,256,1)))
ax[2,1].plot(history.history["dense_4_2_mse"])

ax[3,0].imshow(np.reshape(predictions[3], (256,256,1)))
ax[3,1].plot(history.history["dense_4_3_mse"])


plt.show()

model_name = os.path.join("models", "reference")
model.save(model_name)
