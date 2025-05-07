import os
import requests
import numpy as np
import tensorflow as tf
import sys

CONTROLLER_URL = 'http://localhost:12345'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

def train_locally(weights, x_batch, y_batch):
    model.set_weights(weights)
    x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int64)
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss = loss_object(y_batch, preds)
    
    # Compute gradients (but do NOT apply them)
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients



model = create_model_cifar_100()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)



def worker_loop():
    _ = requests.post(
        f'{CONTROLLER_URL}/start_timer'
    ).json()
    while True:
        response = requests.get(
            f'{CONTROLLER_URL}/get_task?worker_id=worker_{sys.argv[1]}'
        ).json()
        
        weights = [np.array(w) for w in response['weights']]
        x_batch = response['x']
        y_batch = response['y']
        
        gradients = train_locally(weights, x_batch, y_batch)
        
        # Changed key from 'weights' to 'gradients' to match controller
        serialized_grads = []
        for grad in gradients:
            serialized_grads.append(grad.numpy().tolist())  # Convert to list
        
        # 4. Send updates back to controller
        requests.post(
            f'{CONTROLLER_URL}/submit_update',
            json={
                'gradients': serialized_grads,
            }
        )

if __name__ == '__main__':
    worker_loop()