import os
import json
import cv2
import numpy as np
import requests
import tempfile
import shutil
from tqdm import tqdm
import mediapipe as mp

# --- Config ---
DATA_DIR = 'MS-ASL'
OUT_DIR = 'MS_train/DATAASL/processed_data'
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = [
    ('MSASL_train.json', 'landmarks_train.npy', 'labels_train.npy'),
    ('MSASL_val.json', 'landmarks_val.npy', 'labels_val.npy'),
    ('MSASL_test.json', 'landmarks_test.npy', 'labels_test.npy'),
]

mp_hands = mp.solutions.hands

# --- Helper Functions ---
def download_video(url, out_path):
    if os.path.exists(out_path):
        return out_path
    r = requests.get(url, stream=True)
    with open(out_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return out_path

def extract_clip(video_path, start_time, end_time, fps):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, min(end_frame, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_landmarks_from_frame(frame, hands_detector, num_landmarks=21):
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Each hand: 21 landmarks, each with x, y, z
    left = np.zeros(num_landmarks * 3)
    right = np.zeros(num_landmarks * 3)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                left = coords
            elif label == 'Right':
                right = coords
    return left, right

def process_split(json_file, out_landmarks, out_labels):
    with open(os.path.join(DATA_DIR, json_file), 'r') as f:
        samples = json.load(f)
    features = []
    labels = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for sample in tqdm(samples, desc=f'Processing {json_file}'):
            url = sample['url']
            start_time = sample['start_time']
            end_time = sample['end_time']
            label = sample['label']
            fps = sample.get('fps', 30)
            # Download video to temp file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                video_path = download_video(url, tmp.name)
                # Extract frames for the clip
                frames = extract_clip(video_path, start_time, end_time, fps)
            os.remove(tmp.name)
            # For each frame, extract landmarks
            clip_landmarks = []
            for frame in frames:
                left, right = extract_landmarks_from_frame(frame, hands)
                clip_landmarks.extend(left)
                clip_landmarks.extend(right)
            features.append(np.array(clip_landmarks, dtype=np.float32))
            labels.append(label)
    # Pad features to the same length
    max_len = max(len(f) for f in features)
    features_padded = np.zeros((len(features), max_len), dtype=np.float32)
    for i, f in enumerate(features):
        features_padded[i, :len(f)] = f
    np.save(os.path.join(OUT_DIR, out_landmarks), features_padded)
    np.save(os.path.join(OUT_DIR, out_labels), np.array(labels, dtype=np.int32))

if __name__ == '__main__':
    for json_file, out_landmarks, out_labels in SPLITS:
        process_split(json_file, out_landmarks, out_labels) 