from matplotlib import image
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf

def define_model(width = 256, drop_ratio = 0.1):
	#hard coded input shape because it will always be this
	input = tf.keras.layers.Input(shape = (3,))
	
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
image1		= image.imread(os.path.join("shapes", "tank1.png"))
image1		= np.expand_dims(image1, axis = -1)
print(image1.shape)
img1_repeat	= np.repeat(image1, 4, axis = -1)
y_flat1		= np.reshape(img1_repeat, (img1_repeat.shape[0]*img1_repeat.shape[1], img1_repeat.shape[2]))
print(y_flat1.shape)

image2		= image.imread(os.path.join("shapes", "tank3.png"))
image2		= np.expand_dims(image2, axis = -1)
print(image2.shape)
img2_repeat	= np.repeat(image2, 4, axis = -1)
y_flat2		= np.reshape(img2_repeat, (img2_repeat.shape[0]*img2_repeat.shape[1], img2_repeat.shape[2]))
print(y_flat2.shape)

y_flat_concat = np.concatenate((y_flat1, y_flat2))
print(y_flat_concat.shape)


#the input data X
x1 = np.arange(0.0, 1.0, 1/image1.shape[0])
x2 = np.arange(0.0, 1.0, 1/image1.shape[1])
xx1, xx2 = np.meshgrid(x1, x2)
xx1 = np.expand_dims(xx1, axis = -1)
xx2 = np.expand_dims(xx2, axis = -1)
X = np.concatenate((xx1, xx2), axis = -1)
#and then flatten
X_flat = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
identifier1 = np.zeros((X_flat.shape[0],1))
X_flat1 = np.concatenate((X_flat, identifier1), axis = -1)

identifier2 = np.ones((X_flat.shape[0],1))
X_flat2 = np.concatenate((X_flat, identifier2), axis = -1)

X_flat_concat = np.concatenate((X_flat1, X_flat2))
print(X_flat_concat.shape)


model = define_model()
model.summary()

#use default learning rate of 0.001 for Adam optimizer
model.compile(loss = tf.keras.losses.MeanSquaredError(), loss_weights = [1, 2, 4, 8], optimizer = tf.keras.optimizers.Adam(), metrics = ["mse"])
epochs = 32768 #16384
batch_size = 4096*2
history = model.fit(X_flat_concat, y_flat_concat, epochs = epochs, batch_size = batch_size, shuffle = True)


#now evaluate
x1_test = np.arange(0.0, 1.0, 1/256)
x2_test = np.arange(0.0, 1.0, 1/256)
xx1_test, xx2_test = np.meshgrid(x1_test, x2_test)
xx1_test = np.expand_dims(xx1_test, axis = -1)
xx2_test = np.expand_dims(xx2_test, axis = -1)

ident_zero = np.zeros((xx1_test.shape[0], xx1_test.shape[1], 1))
X_test_zero = np.concatenate((xx1_test, xx2_test, ident_zero), axis = -1)
X_test_zero = np.reshape(X_test_zero, (X_test_zero.shape[0]*X_test_zero.shape[1], X_test_zero.shape[2]))

print(X_test_zero)

predictions1 = model.predict(X_test_zero)
print(len(predictions1))

ident_one = np.ones((xx1_test.shape[0], xx1_test.shape[1], 1))
X_test_one = np.concatenate((xx1_test, xx2_test, ident_one), axis = -1)
X_test_one = np.reshape(X_test_one, (X_test_one.shape[0]*X_test_one.shape[1], X_test_one.shape[2]))

print(X_test_one)

predictions2 = model.predict(X_test_one)
print(len(predictions2))

halves = ident_one * 0.5
X_test_half = np.concatenate((xx1_test, xx2_test, halves), axis = -1)
X_test_half = np.reshape(X_test_half, (X_test_half.shape[0]*X_test_half.shape[1], X_test_half.shape[2]))

print(X_test_half)

predictions3 = model.predict(X_test_half)
print(len(predictions3))

#show stuff

fig, ax = plt.subplots(nrows = 4, ncols = 3)

ax[0,0].imshow(np.reshape(predictions1[0], (256,256,1)))
ax[0,1].imshow(np.reshape(predictions2[0], (256,256,1)))
ax[0,2].imshow(np.reshape(predictions3[0], (256,256,1)))

ax[1,0].imshow(np.reshape(predictions1[1], (256,256,1)))
ax[1,1].imshow(np.reshape(predictions2[1], (256,256,1)))
ax[1,2].imshow(np.reshape(predictions3[1], (256,256,1)))

ax[2,0].imshow(np.reshape(predictions1[2], (256,256,1)))
ax[2,1].imshow(np.reshape(predictions2[2], (256,256,1)))
ax[2,2].imshow(np.reshape(predictions3[2], (256,256,1)))

ax[3,0].imshow(np.reshape(predictions1[3], (256,256,1)))
ax[3,1].imshow(np.reshape(predictions2[3], (256,256,1)))
ax[3,2].imshow(np.reshape(predictions3[3], (256,256,1)))

plt.show()

model_name = os.path.join("models", "4Out_multiimage_sharedOut_weighted")
model.save(model_name)
