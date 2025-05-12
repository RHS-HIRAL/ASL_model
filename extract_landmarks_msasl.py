import os
import sys
import json
import cv2
import numpy as np
import tempfile
import shutil
from tqdm import tqdm
import mediapipe as mp
import random  # For sampling subset
import time
import re
import subprocess
import hashlib
from yt_dlp import YoutubeDL

SUBSET_PERCENTAGE = 1

# --- Config ---
WORKSPACE_ROOT = 'C:/Hiral/Projects/Sign Language Projects/model_train'  # Absolute path from user info
DATA_DIR = os.path.join(WORKSPACE_ROOT, 'MS_train', 'raw_data')
OUT_DIR = os.path.join(WORKSPACE_ROOT, 'MS_train', 'processed_data')
os.makedirs(OUT_DIR, exist_ok=True)
ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg-7.1.1-essentials_build', 'bin')
os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']

SPLITS = [
    ('MSASL_train.json', 'landmarks_train.npy', 'labels_train.npy'),
    ('MSASL_val.json', 'landmarks_val.npy', 'labels_val.npy'),
    ('MSASL_test.json', 'landmarks_test.npy', 'labels_test.npy'),
]

mp_hands = mp.solutions.hands

def install_package(package):
    try:
        import pkg_resources
        pkg_resources.get_distribution(package)
        print(f"{package} is already installed")
    except pkg_resources.DistributionNotFound:
        print(f"Installing {package}...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} has been installed successfully")
        except Exception as e:
            print(f"Failed to install {package}: {e}")
            return False
    return True

def extract_clip(video_path, start_time, end_time, fps):
    print(f"Extracting clip from {video_path} (start: {start_time}s, end: {end_time}s)...")
    
    # Verify the file exists and has content
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist at {video_path}")
        return []
        
    if os.path.getsize(video_path) == 0:
        print(f"Error: Video file at {video_path} is empty (0 bytes)")
        return []
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        
        # Print additional debugging info
        print(f"Video file size: {os.path.getsize(video_path)} bytes")
        print(f"Attempting to read first few bytes to check format...")
        
        try:
            with open(video_path, 'rb') as f:
                header = f.read(20)
                print(f"File header (hex): {header.hex()}")
        except Exception as e:
            print(f"Failed to read file header: {e}")
        
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties - Total frames: {total_frames}, FPS: {video_fps}, Duration: {total_frames/video_fps if video_fps > 0 else 'unknown'} seconds")
    
    if video_fps <= 0:
        print(f"Warning: Invalid FPS ({video_fps}). Using default FPS={fps}")
        video_fps = fps
    
    start_frame = max(0, int(start_time * video_fps))
    end_frame = min(total_frames, int(end_time * video_fps))
    
    if start_frame >= end_frame:
        print(f"Warning: Invalid frame range. Start frame ({start_frame}) >= End frame ({end_frame})")
        return []
    
    frames = []
    
    # First try using frame position
    success = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not success:
        print(f"Warning: Could not set start frame position. Trying to read from beginning...")
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        # Read and discard frames until start_frame
        for _ in range(start_frame):
            cap.read()
    
    # Read desired frames
    frame_count = 0
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}. Stopping clip extraction.")
            break
        frames.append(frame)
        frame_count += 1
        
        # Print progress periodically
        if frame_count % 30 == 0:
            print(f"Read {frame_count} frames so far...")
    
    cap.release()
    print(f"Extracted {len(frames)} frames from the clip.")
    return frames

def extract_landmarks_from_frame(frame, hands_detector, num_landmarks=21):
    results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

def hash_sample(sample):
    """Generate a unique hash for a sample based on URL and time range."""
    key = f"{sample['url']}_{sample['start_time']}_{sample['end_time']}"
    return hashlib.md5(key.encode()).hexdigest()

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': True
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        print(f"yt-dlp failed: {e}")
        return None

def process_split(json_file, out_landmarks, out_labels):
    print(f"Starting processing for {json_file}...")
    json_path = os.path.join(DATA_DIR, json_file)

    with open(json_path, 'r') as f:
        samples = json.load(f)

    # Select 10% subset
    random.seed(42)
    subset_size = int(len(samples) * SUBSET_PERCENTAGE)
    subset_samples = random.sample(samples, subset_size)
    print(f"Selected {subset_size} samples out of {len(samples)} for processing.")

    landmarks_path = os.path.join(OUT_DIR, out_landmarks)
    labels_path = os.path.join(OUT_DIR, out_labels)
    processed_hashes = set()
    features = []
    labels = []

    # Load existing data if available
    if os.path.exists(landmarks_path) and os.path.exists(labels_path):
        features = list(np.load(landmarks_path, allow_pickle=True))
        labels = list(np.load(labels_path, allow_pickle=True))
        print(f"Loaded {len(features)} previously processed samples.")
        with open(landmarks_path + ".hashes", 'r') as hf:
            processed_hashes = set(line.strip() for line in hf)

    hash_file = open(landmarks_path + ".hashes", 'a')
    successful_samples = 0
    download_failures = 0
    extraction_failures = 0
    processing_failures = 0

    batch_features, batch_labels, batch_hashes = [], [], []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for sample_idx, sample in enumerate(subset_samples):
            sample_hash = hash_sample(sample)
            if sample_hash in processed_hashes:
                print(f"[{sample_idx+1}/{len(subset_samples)}] Already processed, skipping.")
                continue

            print(f"\n[Sample {sample_idx+1}/{len(subset_samples)}] Processing {sample['url']}")
            url = sample['url']
            start_time = sample['start_time']
            end_time = sample['end_time']
            label = sample['label']
            fps = sample.get('fps', 30)

            temp_filename = f"temp_{int(time.time())}_{random.randint(1000,9999)}.mp4"
            temp_filepath = os.path.join(tempfile.gettempdir(), temp_filename)

            try:
                video_path = download_video(url, temp_filepath)

                if video_path is None:
                    print("Download failed, marking for deletion.")
                    download_failures += 1
                    # Immediately delete the sample from the list
                    samples = [s for s in samples if s != sample]
                    with open(json_path, 'w') as f:
                        json.dump(samples, f, indent=2)
                    continue  # Skip to next sample

                frames = extract_clip(video_path, start_time, end_time, fps)
                cv2.destroyAllWindows()

                if not frames:
                    print("No frames extracted, skipping.")
                    extraction_failures += 1
                    continue

                clip_landmarks = []
                for frame_idx, frame in enumerate(frames):
                    if frame_idx % 10 == 0:
                        print(f"Extracting landmarks from frame {frame_idx+1}/{len(frames)}")
                    left, right = extract_landmarks_from_frame(frame, hands)
                    clip_landmarks.extend(left)
                    clip_landmarks.extend(right)

                batch_features.append(np.array(clip_landmarks, dtype=np.float32))
                batch_labels.append(label)
                batch_hashes.append(sample_hash)
                successful_samples += 1

                if len(batch_features) >= 10:
                    features.extend(batch_features)
                    labels.extend(batch_labels)
                    np.save(landmarks_path, np.array(features, dtype=object))
                    np.save(labels_path, np.array(labels, dtype=object))
                    for h in batch_hashes:
                        hash_file.write(h + '\n')
                    hash_file.flush()
                    print(f"Checkpoint saved at {len(features)} samples.")
                    batch_features.clear()
                    batch_labels.clear()
                    batch_hashes.clear()

            except Exception as e:
                print(f"Error: {e}")
                processing_failures += 1
            finally:
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                        print(f"Removed temp file {temp_filepath}")
                    except Exception as e:
                        print(f"Could not remove temp file: {e}")

    # Save any remaining samples
    if batch_features:
        features.extend(batch_features)
        labels.extend(batch_labels)
        np.save(landmarks_path, np.array(features, dtype=object))
        np.save(labels_path, np.array(labels, dtype=object))
        for h in batch_hashes:
            hash_file.write(h + '\n')
        hash_file.flush()

    hash_file.close()

    # Summary
    print("\n--- ✅ Processing Complete ---")
    print(f"Total attempted: {len(subset_samples)}")
    print(f"Successful: {successful_samples}")
    print(f"Download failures: {download_failures}")
    print(f"Extraction failures: {extraction_failures}")
    print(f"Processing failures: {processing_failures}")

if __name__ == '__main__':
    # Make sure required packages are installed
    required_packages = ['pytube', 'opencv-python', 'numpy', 'mediapipe', 'tqdm']
    for package in required_packages:
        install_package(package)
    
    # Try to fix common pytube issues
    try:
        print("Applying potential fixes to pytube...")
        # This is a common fix for pytube regex issues
        from pytube import cipher
        if hasattr(cipher, "get_initial_function_name"):
            original_get_initial_function_name = cipher.get_initial_function_name
            def patched_get_initial_function_name(js):
                try:
                    return original_get_initial_function_name(js)
                except Exception:
                    # Common pattern in recent YouTube responses
                    function_patterns = [
                        r'(?:c&&d\.set\([^,]+,encodeURIComponent\(([a-zA-Z0-9$]+)\(',
                        r'(?:=[a-zA-Z0-9$]+\.get\("n"\)\)&&\([a-zA-Z0-9$]+=([a-zA-Z0-9$]+)',
                        r'a\.[a-zA-Z]\?([a-zA-Z0-9$]+)'
                    ]
                    for pattern in function_patterns:
                        regex = re.compile(pattern)
                        function_match = regex.search(js)
                        if function_match:
                            print(f"Applied pytube regex fix with pattern: {pattern}")
                            return function_match.group(1)
                    raise
            cipher.get_initial_function_name = patched_get_initial_function_name
            print("Applied patch to pytube cipher functions")
    except Exception as e:
        print(f"Could not apply pytube fixes: {e}")
        print("Continuing anyway...")
    
    # Now process the splits
    for json_file, out_landmarks, out_labels in SPLITS:
        process_split(json_file, out_landmarks, out_labels)
