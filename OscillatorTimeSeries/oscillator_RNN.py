import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from Oscillators import Oscillator
import time
keras = tf.keras
#loeading testing values 
tf.random.set_seed(42)
np.random.seed(42)

def collect_data(sample_number,length,intervals,  osc_class):
    steps = np.linspace(0, length, intervals)
    val_data = np.zeros((sample_number, 1, intervals))
    init_var_data = np.zeros((sample_number, 6))

    for i in range(sample_number):
        init_conditions = np.random.rand(6)

        init_conditions[0] = 0.1 + 0.9 * init_conditions[0]  # mass: 0.1-1.0
        init_conditions[1] = 0.5 + 1.5 * init_conditions[1]  # k: 0.5-2.0
        init_conditions[2] = 0.1 + 0.9 * init_conditions[2]  # A0: 0.1-1.0
        init_conditions[3] = 0.1 * init_conditions[3]        # b: 0-0.1
        init_conditions[4] = 0.1 * init_conditions[4]        # F0: 0-0.1
        init_conditions[5] = 0.5 * init_conditions[5]  

        osc_obj = osc_class(*init_conditions)
        
        val_data[i] = np.array(osc_obj.get_x_pos(steps))
        init_var_data[i] = init_conditions

    return init_var_data, val_data


def preparing_data(init_var, series, batch_size = 32, shuffle_buffer = 1000, training_data = True):
    dataset = tf.data.Dataset.from_tensor_slices((init_var, series))
    if (training_data) : dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


model = keras.models.Sequential([
    keras.layers.Input(shape = (6,)),
    keras.layers.RepeatVector(500),
    keras.layers.SimpleRNN(100, return_sequences=True),
    keras.layers.SimpleRNN(100),
    keras.layers.Dense(1)
    ])

# lr_schedule = keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-7 * 10**(epoch / 20))

initial_values, series_values = collect_data(1000, 50, 500, Oscillator)
train_set = preparing_data(initial_values, series_values, batch_size= 128)


optimizer = keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

early_stopping = keras.callbacks.EarlyStopping(patience=50)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint", save_best_only=True)

#history = model.fit(train_set, epochs = 50, callbacks=[ea])

model.fit(train_set, epochs=500,
          callbacks=[early_stopping, model_checkpoint])

model = keras.models.load_model("my_checkpoint")



#export_path_keras = "./{}.h5".format(int(time.time()))
#print(export_path_keras)

#model.save(export_path_keras)

# plt.semilogx(history.history["lr"], history.history["loss"])
# #plt.axis([1e-7, 1e-4, 0, 30])
# plt.show()

# init_v, s = collect_data(2, 10, 10, Oscillator)
# preparing_data(init_v, s)