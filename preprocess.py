import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(data_dir='data/raw', target_size=(128, 128), train_split=0.8):
    print(f"[INFO] Starting preprocessing for train/val split from: {data_dir}")
    images = []
    labels = []
    #
    # Iterate through class folders
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"[WARN] Skipping non-directory: {class_path}")
            continue
        print(f"[INFO] Processing class: {class_name}")
        
        # Process each image in class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(class_name)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    print(f"[INFO] Total images: {len(X)}, Total labels: {len(y)}")
    
    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        train_size=train_split,
        stratify=y,
        random_state=42
    )
    print(f"[INFO] Split: {X_train.shape[0]} train, {X_val.shape[0]} val")
    return X_train, X_val, y_train, y_val


def preprocess_and_save_images(data_dir='data/raw', processed_dir='data/processed', target_size=(128, 128), augment=True, augment_count=2):
    """
    Preprocess images from data_dir and save them as JPGs in processed_dir/<class>_new/ folders.
    Naming convention: <class>_new/img_<index>.jpg
    If augment=True, generate augment_count augmented images per original using Keras ImageDataGenerator.
    All images are converted to grayscale and then to 3-channel before saving.
    """
    print(f"[INFO] Starting preprocessing and saving to: {processed_dir}")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Set up augmentation
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = None

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"[WARN] Skipping non-directory: {class_path}")
            continue
        new_class_name = f"{class_name}_new"
        new_class_path = os.path.join(processed_dir, new_class_name)
        os.makedirs(new_class_path, exist_ok=True)
        print(f"[INFO] Processing and saving class: {class_name} -> {new_class_name}")

        for idx, img_name in enumerate(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue
            img = cv2.resize(img, target_size)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
            img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Back to 3-channel
            img_norm = (img_gray_3ch.astype(np.float32) / 255.0)  # normalize for augmentation
            # Save original
            save_name = f"{class_name}_img_{idx+1}.jpg"
            save_path = os.path.join(new_class_path, save_name)
            cv2.imwrite(save_path, img_gray_3ch)
            # Augment and save
            if augment and datagen is not None:
                img_exp = np.expand_dims(img_norm, 0)
                aug_iter = datagen.flow(img_exp, batch_size=1)
                for aug_idx in range(augment_count):
                    aug_img = next(aug_iter)[0]
                    aug_img_uint8 = (aug_img * 255).astype(np.uint8)
                    # Ensure augmented image is also grayscale 3-channel
                    aug_img_gray = cv2.cvtColor(aug_img_uint8, cv2.COLOR_BGR2GRAY)
                    aug_img_gray_3ch = cv2.cvtColor(aug_img_gray, cv2.COLOR_GRAY2BGR)
                    aug_save_name = f"{class_name}_img_{idx+1}_aug{aug_idx+1}.jpg"
                    aug_save_path = os.path.join(new_class_path, aug_save_name)
                    cv2.imwrite(aug_save_path, aug_img_gray_3ch)
            if (idx + 1) % 100 == 0:
                print(f"[INFO] Saved {idx+1} images for class {class_name}")
        print(f"[INFO] Finished class {class_name}, total images: {idx+1 if 'idx' in locals() else 0}")


if __name__ == '__main__':
    print("[INFO] Running preprocess_images (train/val split)...")
    X_train, X_val, y_train, y_val = preprocess_images()
    print(f"[RESULT] Training set shape: {X_train.shape}")
    print(f"[RESULT] Validation set shape: {X_val.shape}")
    print(f"[RESULT] Number of classes: {len(np.unique(y_train))}")

    print("[INFO] Running preprocess_and_save_images (save to processed folders with augmentation and grayscaling)...")
    preprocess_and_save_images()
    print("[DONE] Processing and saving completed.")
