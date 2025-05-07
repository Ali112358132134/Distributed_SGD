import os
from pathlib import Path
import shutil
import signal
from threading import Lock
import time
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

model_lock = Lock()
timer_lock = Lock()
batch_size = 16
step = 0
shutdown_flag = False

training_steps = 1024
training_type = 'Probing'
model_update = 1

# training_steps = 4096
# training_type = 'Executing'
# model_update = 1

parallelism_level = 4
start_time = 0


output_file = open("output.txt", "a")

def center_print(text):
    width = shutil.get_terminal_size().columns
    output_file.write(text.center(width) + "\n")
    output_file.flush()

def create_model_cifar_10():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def create_model_cifar_100():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(100, activation='softmax')
    ])
    return model

def load_data(model_type):
    if model_type.lower() == 'cifar_10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test, x_train = x_test / 255.0, x_train /255.0
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_test, x_train = x_test / 255.0, x_train /255.0
    
    return x_train, y_train, x_test, y_test

# Global model (initialized randomly)
global_model = create_model_cifar_100()
file_path = Path("model_weights.weights.h5")
if file_path.exists():
    global_model.load_weights("model_weights.weights.h5")
global_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
x_train, y_train, x_test, y_test = load_data("cifar_100")
predictions = global_model(x_test, training=False)
initial_loss = loss_object(y_true=y_test, y_pred=predictions)

@app.route('/start_timer', methods=['POST'])
def start_timer():
    global start_time

    with timer_lock:
        if start_time == 0:
            start_time = time.time()
    
    return jsonify({'status': 'success'})

@app.route('/get_task', methods=['GET'])
def get_task():
    global model_version, request_time
    indices = np.random.choice(len(x_train), batch_size)
    with model_lock:
        return jsonify({
            'weights': [w.tolist() for w in global_model.get_weights()],
            'x': x_train[indices].tolist(),  # Already preprocessed
            'y': y_train[indices].flatten().tolist(),  # Explicit flattening
        })


@app.route('/submit_update', methods=['POST'])
def submit_update():
    global step, shutdown_flag, global_optimizer
    if shutdown_flag:
        return jsonify({'status': 'rejected', 'message': 'Server shutting down'}), 503
    
    serialized_grads = request.json['gradients']
    gradients = []
    for grad in serialized_grads:
        gradients.append(np.array(grad, dtype=np.float32))  # Ensure correct dtype
    
    # Apply gradients with thread safety
    with model_lock:
        global_optimizer.apply_gradients(zip(gradients, global_model.trainable_variables))
        step += 1
        if step % 1024 == 0:
            training_time = time.time() - start_time
            predictions = global_model(x_test, training=False)
            loss = loss_object(y_test, predictions)
            accuracy(y_test, predictions)
            convergence_rate = (loss - initial_loss)/training_time
            center_print(f"{training_type}: Loss: {loss:.4f}, Accuracy: {accuracy.result():.4f}, training_time {training_time}, convergence_rate: {convergence_rate}, m: {parallelism_level}")
        if step >= training_steps:
            step = 0
            shutdown_flag = True
            if model_update or training_type == 'Executing':
                global_model.save_weights("model_weights.weights.h5")
            os.kill(os.getpid(), signal.SIGTERM)

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='localhost', port=12345)