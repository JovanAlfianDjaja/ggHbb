import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import scienceplots
#plt.style.use("science")

from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2,l1


file_path = "C:/Users/jovan/Downloads/dataggHbb.txt"

data = np.loadtxt(file_path)
np.random.shuffle(data)
print(data.shape)

train_split_ratio = 0.8
val_split_ratio = 0.1
train_split_count = int(data.shape[0] * train_split_ratio)
val_split_count = int(data.shape[0] * val_split_ratio)
num_features = data.shape[1] - 1

x = data[:, :num_features]/1000000
y = np.log(data[:, num_features:])

#pt = QuantileTransformer(output_distribution='uniform')
#pt = pt.fit(x[:train_split_count], y[:train_split_count])
#x = pt.transform(x)
#
#scaler = MinMaxScaler()
#x = scaler.fit_transform(x)

data = np.concatenate([x, y], axis=-1)

data_train = data[:train_split_count]
data_val = data[train_split_count:train_split_count + val_split_count]
data_test = data[train_split_count + val_split_count:]
print(data_train.shape)
print(data_val.shape)
print(data_test.shape)

x_train = data_train[:, :num_features]
y_train = data_train[:, num_features:]

x_val = data_val[:, :num_features]
y_val = data_val[:, num_features:]

x_test = data_test[:, :num_features]
y_test = data_test[:, num_features:]


print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

#def create_model(input_shape):
#    input = tf.keras.layers.Input(shape=input_shape)
#    x = tf.keras.layers.Dense(1000)(input)
#    x = tf.keras.layers.Dense(1000)(x)
#    x = tf.keras.layers.Dense(1000)(x)
#    x = tf.keras.layers.Dense(1000, activation='sigmoid')(x)
#    x = tf.keras.layers.Dense(1000, activation='relu')(x)
#    x = tf.keras.layers.Dense(1000, activation='relu')(x)
#    x = tf.keras.layers.Dense(1000, activation='tanh')(x)
#    x = tf.keras.layers.Dense(1000, activation='exponential')(x)
#    x = tf.keras.layers.Dense(1000, activation='relu')(x)
#    x = tf.keras.layers.Dense(1000, activation='sigmoid')(x)
#    x = tf.keras.layers.Dense(1000, activation='relu')(x)
#    x = tf.keras.layers.Dense(1000)(x)
#
#    output = tf.keras.layers.Dense(1)(x)
#
#    model = tf.keras.Model(inputs=input, outputs=output)
#
#    return model

#def create_model(input_shape):
#    input = tf.keras.layers.Input(shape=input_shape)
#    x = tf.keras.layers.Dense(500)(input)
#    x = tf.keras.layers.Dense(500)(x)
#    x = tf.keras.layers.Dense(500)(x)
#    x = tf.keras.layers.Dense(512, activation='selu')(x)
#    x = tf.keras.layers.Dense(512, activation='relu')(x)
#    x = tf.keras.layers.Dense(512, activation='relu')(x)
#    x = tf.keras.layers.Dense(500, activation='tanh')(x)
#    x = tf.keras.layers.Dense(500, activation='exponential')(x)
#    x = tf.keras.layers.Dense(500, activation='relu')(x)
#    x = tf.keras.layers.Dense(500, activation='sigmoid')(x)
#    x = tf.keras.layers.Dense(512, activation='relu')(x)
#    x = tf.keras.layers.Dense(512, activation='selu')(x)
#    x = tf.keras.layers.Dense(512, activation='selu')(x)
#    x = tf.keras.layers.Dense(500)(x)
#
#    output = tf.keras.layers.Dense(1)(x)
#
#    model = tf.keras.Model(inputs=input, outputs=output)
#
#    return model

def create_model(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(250)(input)
    x = tf.keras.layers.Dense(512, activation='selu')(x)
    x = tf.keras.layers.Dense(250)(x)
    x = tf.keras.layers.Dense(250)(x)
    x = tf.keras.layers.Dense(512, activation='selu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(250, activation='tanh')(x)
    x = tf.keras.layers.Dense(250, activation='exponential')(x)
    x = tf.keras.layers.Dense(250, activation='relu')(x)
    x = tf.keras.layers.Dense(250, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='selu')(x)
    x = tf.keras.layers.Dense(512, activation='selu')(x)
    x = tf.keras.layers.Dense(250)(x)

    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model

input_shape = (num_features,)
model = create_model(input_shape)
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss val_loss
                              factor=0.75,         # Reduce learning rate by half
                              patience=10, #10         # Wait for 5 epochs before reducing LR
                              verbose=1,          # Print a message when reducing LR
                              mode='min',         # Reduce if the loss doesn't improve
                              min_lr=1e-30)        # Lower bound for learning rate

lr_schedule = reduce_lr

optimizer = tf.keras.optimizers.Nadam(learning_rate=8e-4)
loss = tf.keras.losses.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=loss)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=50, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=125, epochs=5000, callbacks=[early_stopping,lr_schedule], verbose=1)

model.save("model8.keras")

losses = np.array(history.history['loss'])
val_losses = np.array(history.history['val_loss'])

plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.plot(np.arange(losses.shape[0]), losses, label="Train MSE")
plt.plot(np.arange(losses.shape[0]), val_losses, label="Validation MSE")
plt.legend()
# plt.savefig("Images/Full-Data-Train.png", dpi=300)

print(losses[-1])
print(val_losses[-1])

y_train_pred = model.predict(x_train)[:, 0]
y_val_pred = model.predict(x_val)[:, 0]
y_test_pred = model.predict(x_test)[:, 0]

min_ = np.min(np.concatenate((y_train[:, 0], y_val[:, 0], y_test[:, 0], y_train_pred, y_val_pred, y_test_pred)))
max_ = np.max(np.concatenate((y_train[:, 0], y_val[:, 0], y_test[:, 0], y_train_pred, y_val_pred, y_test_pred)))
# r2 = r2_score(np.concatenate((y_train[:, 0], y_val[:, 0], y_test[:, 0])), np.concatenate((y_train_pred, y_val_pred, y_test_pred)))
r2_test = r2_score(y_test, y_test_pred)

# plt.scatter(y_train[:, 0], y_train_pred, label="Train Data")
# plt.scatter(y_val[:, 0], y_val_pred, label="Validation Data")
plt.scatter(y_test[:, 0], y_test_pred, label="Test Data")
plt.plot([min_, max_], [min_, max_])

# print(r2)
print(f"r2 score : {r2_test}")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.legend(bbox_to_anchor=(1.2, 0.95))

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')




