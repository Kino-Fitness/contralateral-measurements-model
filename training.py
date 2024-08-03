import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import process_data, s2g
import os
from data_preprocessing import get_data
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_error

# Check if the file exists
path_train_df = 'saved/dataframes/train_df.pkl'
path_test_df = 'saved/dataframes/test_df.pkl'
if not os.path.isfile(path_train_df) and not os.path.isfile(path_test_df):
    get_data()

train_df = pd.read_pickle(path_train_df)
test_df = pd.read_pickle(path_test_df)

# organize training and testing data into input streams
X_train_front_images, X_train_back_images, X_train_tabular, Y_train_right_bicep, Y_train_left_bicep, Y_train_chest, Y_train_right_forearm, Y_train_left_forearm, Y_train_right_quad, Y_train_left_quad, Y_train_right_calf, Y_train_left_calf, Y_train_waist, Y_train_hips, Y_train_bodypose, Y_train_joints = process_data(train_df)
X_test_front_images, X_test_back_images, X_test_tabular, Y_test_right_bicep, Y_test_left_bicep, Y_test_chest, Y_test_right_forearm, Y_test_left_forearm, Y_test_right_quad, Y_test_left_quad, Y_test_right_calf, Y_test_left_calf, Y_test_waist, Y_test_hips, Y_test_bodypose, Y_test_joints = process_data(test_df)

# defining the model
image_shape = (224, 224, 3)
num_tabular_features = 4

base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')

# Front Image branch
front_image_input = Input(shape=image_shape, name='front_image_input')
front_image_x = base_model(front_image_input)
front_image_x = GlobalAveragePooling2D()(front_image_x)  # Use the output of base_model(front_image_input)
front_image_features = Flatten()(front_image_x)
front_image_features = BatchNormalization()(front_image_features)

# Back Image branch
back_image_input = Input(shape=image_shape, name='back_image_input')
back_image_x = base_model(back_image_input)
back_image_x = GlobalAveragePooling2D()(back_image_x)  # Use the output of base_model(back_image_input)
back_image_features = Flatten()(back_image_x)
back_image_features = BatchNormalization()(back_image_features)

# Tabular data branch
tabular_input = Input(shape=(num_tabular_features,), name='tabular_input')
tabular_features = Flatten()(tabular_input)
tabular_features = Dense(32, activation='relu')(tabular_features)
tabular_features = BatchNormalization()(tabular_features)

# Combine image, categorical, and numerical features
combined_features = concatenate([front_image_features, back_image_features, tabular_features])

# Final output layers
outputs = ['right_bicep', 'left_bicep', 'chest', 'right_forearm', 'left_forearm', 'right_quad', 'left_quad', 'right_calf', 'left_calf', 'waist', 'hips']
output_layers = []

for output in outputs:
    output_layer = Dense(1000, activation='softmax', name='output_' + output)(combined_features)
    output_layers.append(output_layer)

output_bodypose = Dense(63, name='output_bodypose')(combined_features)
output_joints = Dense(354, name='output_joints')(combined_features)

# Create the model
model = Model(
    inputs=[
        front_image_input, 
        back_image_input, 
        tabular_input
    ], 
    outputs=output_layers + [output_bodypose]
)


def l1_l2_loss(y_true, y_pred, output):
    l1_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)  # Softmax Crossentropy loss
    decoded_y_true = s2g[output].decode_tensor(vector=y_true)
    decoded_y_pred = s2g[output].decode_tensor(vector=y_pred)
    # Ensure decoded outputs have at least one dimension
    decoded_y_true = tf.expand_dims(decoded_y_true, -1)
    decoded_y_pred = tf.expand_dims(decoded_y_pred, -1)
    
    l2_loss = tf.keras.losses.MeanSquaredError()(decoded_y_true, decoded_y_pred)  # MSE loss
    total_loss = l1_loss + l2_loss
    return total_loss

def loss_wrapper(output):
    def loss_fn(y_true, y_pred):
        return l1_l2_loss(y_true, y_pred, output)
    return loss_fn

# Compile the model
losses = {'output_' + output: loss_wrapper(output) for output in outputs}
losses['output_bodypose'] = 'mean_squared_error'

optimizer = Adam(learning_rate=.0001)

model.compile(optimizer=optimizer,
              loss=losses,
              metrics=['mae'] * (len(outputs) + 1))

# Print model summary
model.summary()

# Train the model
checkpoint = ModelCheckpoint('/home/aavsi/multitask/model_checkpoint.keras', 
                             monitor='val_loss', 
                             save_best_only=True, 
                             mode='min', 
                             verbose=1
)
early_stopping = EarlyStopping(monitor='loss', 
                               patience=10, 
                               mode='min', 
                               verbose=1
)

result = model.fit(
    [
        X_train_front_images,
        X_train_back_images,
        X_train_tabular
    ],
    [
        Y_train_right_bicep,
        Y_train_left_bicep,
        Y_train_chest,
        Y_train_right_forearm,
        Y_train_left_forearm,
        Y_train_right_quad,
        Y_train_left_quad,
        Y_train_right_calf,
        Y_train_left_calf,
        Y_train_waist,
        Y_train_hips,
        Y_train_bodypose
    ],
    epochs=1000,
    batch_size=8,
    verbose=1,
    callbacks=[checkpoint, early_stopping],
)

# Plot the loss over epochs
plt.plot(result.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
evaluation = model.evaluate(
    [
        X_test_front_images, 
        X_test_back_images, 
        X_test_tabular
    ], 
    [
        Y_test_right_bicep,
        Y_test_left_bicep,
        Y_test_chest,
        Y_test_right_forearm,
        Y_test_left_forearm,
        Y_test_right_quad,
        Y_test_left_quad,
        Y_test_right_calf,
        Y_test_left_calf,
        Y_test_waist,
        Y_test_hips,
        Y_test_bodypose
    ], 
    verbose=1
)


# Get the model's predictions on the test data
predictions_right_bicep, predictions_left_bicep, predictions_chest, predictions_right_forearm, predictions_left_forearm, predictions_right_quad, predictions_left_quad, predictions_right_calf, predictions_left_calf, predictions_waist, predictions_hips, predictions_body_pose = model.predict(
    [
        X_test_front_images, 
        X_test_back_images,
        X_test_tabular
    ])

def decode_scalar(vector, output):
    return [np.round(s2g[output].decode(prediction), 1) for prediction in vector]

predictions = {
    'right_bicep': decode_scalar(predictions_right_bicep, 'right_bicep'),
    'left_bicep': decode_scalar(predictions_left_bicep, 'left_bicep'),
    'chest': decode_scalar(predictions_chest, 'chest'),
    'right_forearm': decode_scalar(predictions_right_forearm, 'right_forearm'),
    'left_forearm': decode_scalar(predictions_left_forearm, 'left_forearm'),
    'right_quad': decode_scalar(predictions_right_quad, 'right_quad'),
    'left_quad': decode_scalar(predictions_left_quad, 'left_quad'),
    'right_calf': decode_scalar(predictions_right_calf, 'right_calf'),
    'left_calf': decode_scalar(predictions_left_calf, 'left_calf'),
    'waist': decode_scalar(predictions_waist, 'waist'),
    'hips': decode_scalar(predictions_hips, 'hips')
}

ground_truth = {
    'right_bicep': decode_scalar(Y_test_right_bicep, 'right_bicep'),
    'left_bicep': decode_scalar(Y_test_left_bicep, 'left_bicep'),
    'chest': decode_scalar(Y_test_chest, 'chest'),
    'right_forearm': decode_scalar(Y_test_right_forearm, 'right_forearm'),
    'left_forearm': decode_scalar(Y_test_left_forearm, 'left_forearm'),
    'right_quad': decode_scalar(Y_test_right_quad, 'right_quad'),
    'left_quad': decode_scalar(Y_test_left_quad, 'left_quad'),
    'right_calf': decode_scalar(Y_test_right_calf, 'right_calf'),
    'left_calf': decode_scalar(Y_test_left_calf, 'left_calf'),
    'waist': decode_scalar(Y_test_waist, 'waist'),
    'hips': decode_scalar(Y_test_hips, 'hips')
}

mae = {}
for output in outputs:
    mae[output] = mean_absolute_error(decode_scalar(eval(f'Y_test_{output}'), output), predictions[output])

total_mae = sum(mae.values()) / len(mae)
print(f"Total MAE: {total_mae}")
for output in outputs:
    print(f"MAE {output.capitalize()}: {round(mae[output], 2)}")

print(predictions)
print(ground_truth)
# Save the model
# model.save('/home/aavsi/multitask/saved/exported_models/86.keras')
