"""
CME Multi-View Parameter Prediction using CNNs + Metadata
---------------------------------------------------------

This script performs cross-validation to train and evaluate a convolutional neural network (CNN)
that predicts 3D CME parameters using synthetic coronagraph images from three viewpoints (C3, Cor2A, Cor2B)
and spacecraft metadata. The pipeline consists of:

1. Data Loading and Preprocessing
   - Loads grayscale images (C3, Cor2A, Cor2B) and stacks them into RGB-format tensors.
   - Reads and aligns image-metadata-output from CSVs.
   - Computes sin/cos of longitude to handle circularity.

2. Data Organization
   - Structures each CME sequence into a dictionary containing images, metadata, outputs, and filenames.

3. Model Architecture
   - CNN feature extractor with configurable depth and width.
   - Metadata sub-network concatenated with image features.
   - Dense layers for regression output predicting:
       *Longitude (sin/cos), Latitude, Orientation, Heights from SOHO, STA, STB.

4. Training Setup
   - Uses K-Fold cross-validation (outer loop) with 5 repeats per fold.
   - Applies StandardScaler to normalize outputs and metadata.
   - Uses Huber loss + MAE + MSE as evaluation metrics.
   - Saves best-performing model (lowest validation MAE) for each fold.

5. Evaluation & Results
   - Predicts on the test set using best fold model.
   - Decodes sin/cos back to longitude degrees.
   - Saves predictions, ground truth, scalers, model weights, and plots.

Output:
-------
- Best model weights per fold (`*.keras`, `*.h5`)
- Scalers for metadata and output (`*.save`)
- Predictions and ground truths as CSVs
- Prediction vs. actual scatter plots
- Normalized image arrays for Grad-CAM

Dependencies:
-------------
- TensorFlow / Keras
- NumPy, pandas, matplotlib
- scikit-learn (StandardScaler, KFold)
- joblib for saving scalers
- Data must be pre-generated using `synth_cme` pipeline

Author: Harshita Gandhi, written at Aberystwyth University, 2025 credit: gpu slurm
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import Huber, MeanSquaredError, MeanAbsoluteError
import csv
import gc


np.random.seed(42)

# 1) Data Loading and Preprocessing

# Define paths, image parameters, and column names.
csv_folder = 'csv_path to the metadata csvs for each CME event'
base_image_folder = 'path to the folders containing images of C3, COR2A and COR2B'
output_directory = 'path to save the outputs of this code'
valmetrics_csv_path = os.path.join(output_directory, 'cv_val_metrics.csv')
testmetrics_csv_path = os.path.join(output_directory, 'cv_test_metrics.csv')

image_height, image_width = 256, 256
num_channels = 3  # 3 channels: C3, Cor2a, Cor2b


output_cols = ["LonRel2Earth_sin", "LonRel2Earth_cos", "LatRel2Earth", "Orient", "Height_soho", "Height_sta", "Height_stb"]

metadata_cols = [
    "lon_soho", "lat_soho", "pixperrs_soho",
    "lon_sta",  "lat_sta",  "pixperrs_sta",
    "lon_stb",  "lat_stb",  "pixperrs_stb",
    "time_soho", "time_sta", "time_stb"
]
metrics_records = []

# Helper functions to generate filenames based on the CSV composite string.
def extract_c3_filename(csv_filename):
    parts = csv_filename.split("_")
    date_time = parts[4] + "_" + parts[6]
    return f"cme_c3_lasco_soho_dynamics_{date_time}.png"

def extract_cor2a_filename(csv_filename):
    parts = csv_filename.split("_")
    date_time = parts[4] + "_" + parts[8]
    return f"cme_cor2_a_stereo_dynamics_{date_time}.png"

def extract_cor2b_filename(csv_filename):
    parts = csv_filename.split("_")
    date_time = parts[4] + "_" + parts[10]
    return f"cme_cor2_b_stereo_dynamics_{date_time}"

# Build a dictionary of sequences with images, metadata, outputs, and filenames.
sequences = {}

for folder in os.listdir(base_image_folder):
    folder_path = os.path.join(base_image_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    # Define view subdirectories.
    c3_folder_path = os.path.join(folder_path, 'C3_extracted')
    cor2a_folder_path = os.path.join(folder_path, 'Cor2a_extracted')
    cor2b_folder_path = os.path.join(folder_path, 'Cor2b_extracted')
    
    if not (os.path.isdir(c3_folder_path) and os.path.isdir(cor2a_folder_path) and os.path.isdir(cor2b_folder_path)):
        print(f"[WARNING] One or more view folders missing in folder: {folder}")
        continue
    
    # The corresponding CSV is assumed to be named <folder>_linked.csv.
    csv_path = os.path.join(csv_folder, folder + '_linked.csv')
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found for folder: {folder}")
        continue

    df_cme = pd.read_csv(csv_path)
    if df_cme.empty:
        print(f"[WARNING] CSV is empty for folder: {folder}")
        continue

    sequence_id = folder  
    sequences[sequence_id] = {"images": [], "metadata": [], "outputs": [], "filenames": []}

    # Process each row in the CSV file.
    for idx, row in df_cme.iterrows():
        # Optionally, process only a subset of rows (e.g., idx > 1 to skip extra rows)
        if idx ==0:
            continue

        csv_row = row["CompositeImage"]
        # Generate filenames for each view.
        c3_filename = extract_c3_filename(csv_row)
        cor2a_filename = extract_cor2a_filename(csv_row)
        cor2b_filename = extract_cor2b_filename(csv_row)
        
        # Construct full image paths.
        c3_image_path = os.path.join(c3_folder_path, c3_filename)
        cor2a_image_path = os.path.join(cor2a_folder_path, cor2a_filename)
        cor2b_image_path = os.path.join(cor2b_folder_path, cor2b_filename)
        
        if not (os.path.exists(c3_image_path) and os.path.exists(cor2a_image_path) and os.path.exists(cor2b_image_path)):
            print(f"[WARNING] Missing image in folder {folder}: one or more images not found.")
            continue

        # Load each image in grayscale and resize.
        img_c3 = tf.keras.preprocessing.image.load_img(c3_image_path, color_mode="grayscale", target_size=(image_height, image_width))
        img_cor2a = tf.keras.preprocessing.image.load_img(cor2a_image_path, color_mode="grayscale", target_size=(image_height, image_width))
        img_cor2b = tf.keras.preprocessing.image.load_img(cor2b_image_path, color_mode="grayscale", target_size=(image_height, image_width))
        
        # Convert images to arrays.
        img_array_c3 = tf.keras.preprocessing.image.img_to_array(img_c3)
        img_array_cor2a = tf.keras.preprocessing.image.img_to_array(img_cor2a)
        img_array_cor2b = tf.keras.preprocessing.image.img_to_array(img_cor2b)
        
        # Stack the three images along the channel dimension.
        combined_img = np.concatenate([img_array_c3, img_array_cor2a, img_array_cor2b], axis=-1)
        
        try:
            meta_vals = row[metadata_cols].values.astype(float)
            # --- Compute sin/cos of LonRel2Earth ---
            lon_deg = row["LonRel2Earth"]
            lon_rad = np.deg2rad(lon_deg)
            lon_sin = np.sin(lon_rad)
            lon_cos = np.cos(lon_rad)

    # --- Collect the rest of the outputs manually ---
            lat = row["LatRel2Earth"]
            orient = row["Orient"]
            height_soho = row["Height_soho"]
            height_sta = row["Height_sta"]
            height_stb = row["Height_stb"]

            output_vals = np.array([lon_sin, lon_cos, lat, orient, height_soho, height_sta, height_stb], dtype=np.float32)
        except Exception as e:
            print(f"[WARNING] Error extracting output columns for row {idx} in folder {folder}: {e}")
            continue

        sequences[sequence_id]["images"].append(combined_img)
        sequences[sequence_id]["metadata"].append(meta_vals)
        sequences[sequence_id]["outputs"].append(output_vals)
        sequences[sequence_id]["filenames"].append(csv_row)

# Helper function: prepares split data from a list of sequence IDs.
def prepare_split_data(sequence_ids):
    X_list, md_list, Y_list, filenames_list = [], [], [], []
    for seq_id in sequence_ids:
        # Normalize images to [0, 1].
        seq_images = np.array(sequences[seq_id]["images"]) / 255.0  
        seq_metadata = np.array(sequences[seq_id]['metadata'])
        seq_outputs = np.array(sequences[seq_id]["outputs"])
        seq_filenames = sequences[seq_id]["filenames"]
        
        X_list.append(seq_images)
        md_list.append(seq_metadata)
        Y_list.append(seq_outputs)
        filenames_list.append(seq_filenames)
    return X_list, md_list, Y_list, filenames_list

# Functions for NaN and numerical stability checks.
def check_nans(data, name):
    if np.isnan(data).any():
        print(f"[ERROR] NaN detected in {name}")
    if np.isinf(data).any():
        print(f"[ERROR] Inf detected in {name}")

def fill_nan_with_same_sequence(md_sequences):
    """
    For each metadata sequence, replace any NaN row by copying values from the previous
    (or next if it is the first row) row in the same sequence.
    """
    for seq_index, seq in enumerate(md_sequences):
        nan_rows = np.isnan(seq).any(axis=1)
        if np.any(nan_rows):
            for i in np.where(nan_rows)[0]:
                if i > 0:
                    seq[i] = seq[i - 1]
                elif seq.shape[0] > 1:
                    seq[i] = seq[i + 1]
                else:
                    print(f"Warning: Sequence {seq_index} has only one row, cannot replace NaNs!")
    return md_sequences


# Model Definition & Loss

def model_block(x, features):
    x = Conv2D(features, (3, 3), padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    return x

def model_out_block(x, dropout):
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def create_model(image_shape, features, dropout, depth, n_outputs, md):
    input_img = Input(shape=image_shape, name='image_input')
    x = input_img
    for _ in range(depth):
        x = model_block(x, features)
        features *= 2
    x = model_out_block(x, dropout)
    
    md_input = Input(shape=(md,), name='md_input')
    y = Dense(md, activation='relu')(md_input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    x = Concatenate()([x, y])
    
    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output = Dense(n_outputs, activation='linear')(x)
    return Model(inputs=[input_img, md_input], outputs=output)

#def custom_loss(y_true, y_pred):
#    delta = 1.0
#    huber_loss = Huber(delta=delta)(y_true, y_pred)
#    mse_loss = MeanSquaredError()(y_true, y_pred)
#    mae_loss = MeanAbsoluteError()(y_true, y_pred)
#    return huber_loss + mse_loss + mae_loss


# Cross Validation Setup

all_sequence_ids = list(sequences.keys())
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists for storing overall metrics and predictions for ensembling.
outer_metrics = []
outer_predictions = []
outer_actuals = []
outer_filenames = []
validation_metrics = []

def scale_outputs(Y_list, scaler):
    scaled = []
    for Y in Y_list:
        Y_scaled = np.zeros_like(Y)
        Y_scaled[:, :2] = Y[:, :2]  # sin and cos untouched
        Y_scaled[:, 2:] = scaler.transform(Y[:, 2:])
        scaled.append(Y_scaled)
    return scaled

def unscale_outputs(Y_scaled, scaler):
    Y_unscaled = np.zeros_like(Y_scaled)
    Y_unscaled[:, :2] = Y_scaled[:, :2]
    Y_unscaled[:, 2:] = scaler.inverse_transform(Y_scaled[:, 2:])
    return Y_unscaled

def normalize_sin_cos(sin_pred, cos_pred):
    norm = np.sqrt(sin_pred**2 + cos_pred**2) + 1e-8  # avoid div by 0
    return sin_pred / norm, cos_pred / norm
        
fold_num = 1
print("All sequence IDs:", all_sequence_ids)

for outer_train_idx, outer_test_idx in outer_cv.split(all_sequence_ids):
    # --- Outer Fold Split ---
    outer_train_ids = [all_sequence_ids[i] for i in outer_train_idx]
    outer_test_ids = [all_sequence_ids[i] for i in outer_test_idx]
    print("\nFold", fold_num)
    print("Outer Train IDs:", outer_train_ids)
    print("Outer Test IDs:", outer_test_ids)
   
    # --- Inner Split: Use a train/validation split on the outer training IDs (e.g., 80/20)
    inner_train_ids, inner_val_ids = train_test_split(outer_train_ids, test_size=0.2, random_state=42)
    print("Inner Train IDs:", inner_train_ids)
    print("Inner Val IDs:", inner_val_ids)
    # --- Prepare data for each set ---
    X_train, md_train, Y_train, f_train = prepare_split_data(inner_train_ids)
    X_val, md_val, Y_val, f_val = prepare_split_data(inner_val_ids)
    X_test, md_test, Y_test, f_test = prepare_split_data(outer_test_ids)
    
    print("Number of sequences in X_train:", len(X_train))
    if len(X_train) > 0:
        print("Shape of first sequence in X_train:", X_train[0].shape)
    print("Number of sequences in md_train:", len(md_train))
    if len(md_train) > 0:
        print("Shape of first sequence in md_train:", md_train[0].shape)
    print("Number of sequences in Y_train:", len(Y_train))
    if len(Y_train) > 0:
        print("Shape of first sequence in Y_train:", Y_train[0].shape)
    print("Number of sequences in f_train:", len(f_train))
    if len(f_train) > 0:
        print("First filenames list:", f_train[0])
        
    # Filter out sequences that might be empty.
    X_train = [seq for seq in X_train if seq.shape[0] > 0]
    md_train = [seq for seq in md_train if seq.shape[0] > 0]
    Y_train = [seq for seq in Y_train if seq.shape[0] > 0]
    X_val = [seq for seq in X_val if seq.shape[0] > 0]
    md_val = [seq for seq in md_val if seq.shape[0] > 0]
    Y_val = [seq for seq in Y_val if seq.shape[0] > 0]
    X_test = [seq for seq in X_test if seq.shape[0] > 0]
    md_test = [seq for seq in md_test if seq.shape[0] > 0]
    Y_test = [seq for seq in Y_test if seq.shape[0] > 0]
    
    # --- Scaling Outputs ---
    Y_train_concat = np.concatenate(Y_train, axis=0)
    output_scaler = StandardScaler()
    output_scaler.fit(Y_train_concat[:,2:])
    

    
    Y_train_scaled = scale_outputs(Y_train, output_scaler)
    Y_val_scaled   = scale_outputs(Y_val, output_scaler)
    Y_test_scaled  = scale_outputs(Y_test, output_scaler)
    
    Y_train_all = np.concatenate(Y_train_scaled, axis=0)
    Y_val_all   = np.concatenate(Y_val_scaled, axis=0)
    Y_test_all  = np.concatenate(Y_test_scaled, axis=0)
    
    # --- Scaling Metadata ---
    md_train_concat = np.concatenate(md_train, axis=0)
    meta_scaler = StandardScaler()
    meta_scaler.fit(md_train_concat)
    
    md_train_scaled = [meta_scaler.transform(seq) for seq in md_train]
    md_val_scaled   = [meta_scaler.transform(seq) for seq in md_val]
    md_test_scaled  = [meta_scaler.transform(seq) for seq in md_test]
    
    md_train_all = np.concatenate(md_train_scaled, axis=0)
    md_val_all   = np.concatenate(md_val_scaled, axis=0)
    md_test_all  = np.concatenate(md_test_scaled, axis=0)
    
    fold_output_dir = os.path.join(output_directory, f"fold_{fold_num}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    
    output_scaler_path = os.path.join(fold_output_dir, f'output_scaler.save')
    meta_scaler_path = os.path.join(fold_output_dir, f'meta_scaler.save')

    joblib.dump(output_scaler, output_scaler_path)
    joblib.dump(meta_scaler, meta_scaler_path)

    print(f"[INFO] Saved output scaler to {output_scaler_path}")
    print(f"[INFO] Saved metadata scaler to {meta_scaler_path}")
    
    # --- NaN and numerical stability checks ---
    X_train_all = np.concatenate(X_train, axis=0)
    check_nans(X_train_all, "X_train_all")
    check_nans(md_train_all, "md_train_all")
    check_nans(Y_train_all, "Y_train_all")
    
    check_nans(np.concatenate(X_val, axis=0), "X_val_all")
    check_nans(md_val_all, "md_val_all")
    check_nans(Y_val_all, "Y_val_all")
    
    check_nans(np.concatenate(X_test, axis=0), "X_test_all")
    check_nans(md_test_all, "md_test_all")
    check_nans(Y_test_all, "Y_test_all")
    
    # Fill NaNs in metadata sequences for validation and test sets.
    md_train_filled = fill_nan_with_same_sequence(md_train_scaled)
    md_val_filled = fill_nan_with_same_sequence(md_val_scaled)
    md_test_filled = fill_nan_with_same_sequence(md_test_scaled)
    md_train_all  = np.concatenate(md_train_filled, axis=0)
    md_val_all = np.concatenate(md_val_filled, axis=0)
    md_test_all = np.concatenate(md_test_filled, axis=0)
    print(f"NaN in md_train_all after fixing: {np.isnan(md_train_all).sum()}")
    print(f"NaN in md_val_all after fixing: {np.isnan(md_val_all).sum()}")
    print(f"NaN in md_test_all after fixing: {np.isnan(md_test_all).sum()}")
    
    # --- Prepare Images: Concatenate all image arrays ---
    X_train_all = np.concatenate(X_train, axis=0)
    X_val_all   = np.concatenate(X_val, axis=0)
    X_test_all  = np.concatenate(X_test, axis=0)
    
    # ===============================
    # 4) Model Definition, Compilation, and Training for This Fold
    # ===============================
    print("Shape of X_train_all:", X_train_all.shape)
    number_of_samples = X_train_all.shape[0]
    batch_size = 128   
    steps_per_epoch = number_of_samples // batch_size
    decay_frequency_in_epochs = 5
    decay_steps = steps_per_epoch * decay_frequency_in_epochs
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=decay_steps,
        decay_rate=0.90,
        staircase=False
    )
    best_test_loss = float('inf')
    best_Y_pred = None
    best_Y_actual = None
    best_filenames = None
    best_val_mae = float('inf')
    best_repeat = None
 
    for repeat_num in range(5):  # Repeat each fold 5 times
        print(f"Fold {fold_num}, Repeat {repeat_num + 1}")
        n_outputs = len(output_cols)
  
        model = create_model((image_height, image_width, num_channels), 32, 0.5, 3, n_outputs, len(metadata_cols))
        model.compile(optimizer=Adam(learning_rate=learning_rate_schedule, clipnorm=1.0),
                       loss=Huber(delta=1.0),
                       metrics=['mae', 'mse'])
                      
        model.summary()
    
    # Set up callbacks with a fold-specific output directory.

        callbacks_list = [
             ModelCheckpoint(os.path.join(fold_output_dir, f'repeat_{repeat_num}_model.keras'),
                             monitor='val_loss', save_best_only=True, mode='min'),
             EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
             TensorBoard(log_dir=os.path.join(fold_output_dir, f'logs_repeat_{repeat_num}'))
         ]
        print("Starting training...")
        history = model.fit(
             {'image_input': X_train_all, 'md_input': md_train_all}, Y_train_all,
             validation_data=({'image_input': X_val_all, 'md_input': md_val_all}, Y_val_all),
             epochs=100, batch_size=batch_size, callbacks=callbacks_list
             )
    
    # ===============================
    # 5) Evaluation, Prediction, and Saving Results for This Fold
    # ===============================
    
        # Compute validation loss
        val_loss, val_mae, val_mse = model.evaluate({'image_input': X_val_all, 'md_input': md_val_all}, Y_val_all)
        
        # Store validation loss
        validation_metrics.append([fold_num, repeat_num, val_loss, val_mae, val_mse])        
             
        test_loss, test_mae, test_mse = model.evaluate({'image_input': X_test_all, 'md_input': md_test_all}, Y_test_all)
        print(f"Fold {fold_num} - Test loss: {test_loss}, MAE: {test_mae}, MSE: {test_mse}")
        metrics_records.append([fold_num, repeat_num, test_loss, test_mae, test_mse])  
        print(f'Metrics {metrics_records}')

        
        # Save predictions if this is the best model for the fold
        if val_mae < best_val_mae:
             best_val_mae = val_mae
             best_repeat = repeat_num
             
             Y_pred_scaled = model.predict({'image_input': X_test_all, 'md_input': md_test_all})
             Y_pred_unscaled = unscale_outputs(Y_pred_scaled, output_scaler)
             Y_actual_unscaled = unscale_outputs(Y_test_all, output_scaler)
             print("Predicted sin/cos stats:")
             print("  sin: min", Y_pred_unscaled[:, 0].min(), "max", Y_pred_unscaled[:, 0].max())
             print("  cos: min", Y_pred_unscaled[:, 1].min(), "max", Y_pred_unscaled[:, 1].max())
 
 # === Decode longitude from sin/cos ===
             lon_sin_pred = Y_pred_unscaled[:, 0]
             lon_cos_pred = Y_pred_unscaled[:, 1]
             sin_pred, cos_pred = normalize_sin_cos(lon_sin_pred, lon_cos_pred)
             lon_pred_rad = np.arctan2(sin_pred, cos_pred)
             lon_pred_deg = np.rad2deg(lon_pred_rad) % 360
 
             lon_sin_actual = Y_actual_unscaled[:, 0]
             lon_cos_actual = Y_actual_unscaled[:, 1]
             sin_actual, cos_actual = normalize_sin_cos(lon_sin_actual, lon_cos_actual)
             lon_actual_rad = np.arctan2(sin_actual, cos_actual)
             lon_actual_deg = np.rad2deg(lon_actual_rad) % 360
 

             best_filenames = [fname for seq in f_test for fname in seq]
             
             model.save_weights(os.path.join(fold_output_dir, f"repeat_{repeat_num}_weights.h5"))
             model.save(os.path.join(fold_output_dir, f"repeat_{repeat_num}_savedmodel"), save_format="tf")
             
        del model
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
            
        # Save best predictions per fold

    Y_pred_with_lon = np.concatenate([lon_pred_deg[:, None], Y_pred_unscaled], axis=1)
    Y_actual_with_lon = np.concatenate([lon_actual_deg[:, None], Y_actual_unscaled], axis=1)

   
    output_cols_final = ["LonRel2Earth_deg", "LonRel2Earth_sin","LonRel2Earth_cos","LatRel2Earth","Orient","Height_soho", "Height_sta","Height_stb"]
    pred_df = pd.DataFrame(Y_pred_with_lon, columns=output_cols_final)
    actual_df = pd.DataFrame(Y_actual_with_lon, columns=output_cols_final)


    pred_df.insert(0, 'filename', best_filenames)
    actual_df.insert(0, 'filename', best_filenames)
    pred_csv_path = os.path.join(output_directory, f'best_test_predictions_fold_{fold_num}.csv')
    actual_csv_path = os.path.join(output_directory, f'best_test_actuals_fold_{fold_num}.csv')
    pred_df.to_csv(pred_csv_path, index=False)
    actual_df.to_csv(actual_csv_path, index=False)
    print(f"Saved best predictions to {pred_csv_path} and best actuals to {actual_csv_path}")
    X_test_save_path = os.path.join(fold_output_dir, "X_test_all.npy")
    md_test_save_path = os.path.join(fold_output_dir, "md_test_all.npy")
    np.save(X_test_save_path, X_test_all)
    np.save(md_test_save_path, md_test_all)
    print(f"Saved test data for Grad-CAM to:\n  {X_test_save_path}\n  {md_test_save_path}")
    # Create scatter plots for each output.
    
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, output_name in enumerate(output_cols_final):
        ax = axes[i]
        actual = Y_actual_with_lon[:, i]
        predicted = Y_pred_with_lon[:, i]
        ax.scatter(actual, predicted, alpha=0.5, label='Predicted vs Actual')
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Ideal')
        ax.set_xlabel(f"Actual {output_name}")
        ax.set_ylabel(f"Predicted {output_name}")
        ax.set_title(f"Fold {fold_num}, Repeat {best_repeat} - {output_name}")
        ax.legend()
    plt.tight_layout()
    fold_plot_path = os.path.join(fold_output_dir, "pred_vs_actual.png")
    plt.savefig(fold_plot_path)
    plt.close()
    
    outer_metrics.append({
    'fold': fold_num,
    'repeat': best_repeat,
    'loss': test_loss,
    'mae': test_mae,
    'mse': test_mse
    })
    outer_predictions.append(Y_pred_with_lon)
    outer_actuals.append(Y_actual_with_lon)
    outer_filenames.extend(best_filenames)
    

    fold_num += 1

# Save all test metrics across folds

val_df = pd.DataFrame(validation_metrics, columns=["Fold", "Repeat", "Val_Loss","Val_MAE","Val_MSE" ])
val_df.to_csv(valmetrics_csv_path, index=False)
metrics_df = pd.DataFrame(metrics_records, columns=['Fold', 'Repeat', 'Test Loss', 'MAE', 'MSE'])
metrics_df.to_csv(testmetrics_csv_path, index=False)


