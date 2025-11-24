"""
Emergency MediaPipe Feature Extraction from PNG Frames
Extracts features from Phoenix dataset image sequences
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    """Extract all landmarks from MediaPipe results."""

    # Pose landmarks (33 keypoints x 3 coordinates)
    pose = np.zeros(33 * 3)
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            pose[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Face landmarks (468 keypoints x 3 coordinates)
    face = np.zeros(468 * 3)
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            face[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Left hand landmarks (21 keypoints x 3 coordinates)
    left_hand = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Right hand landmarks (21 keypoints x 3 coordinates)
    right_hand = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    return np.concatenate([pose, face, left_hand, right_hand])

def process_frame_sequence(frame_dir):
    """Process a sequence of PNG frames and extract features."""
    frame_dir = Path(frame_dir)

    # Find the subdirectory (usually '1')
    subdirs = [d for d in frame_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    frame_folder = subdirs[0]  # Usually '1'

    # Get all PNG files sorted by frame number
    frame_files = sorted(list(frame_folder.glob('*.png')))

    if not frame_files:
        return None

    features = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for frame_file in frame_files:
            # Read image
            image = cv2.imread(str(frame_file))
            if image is None:
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = holistic.process(image_rgb)

            # Extract landmarks
            frame_features = extract_landmarks(results)
            features.append(frame_features)

    if features:
        return np.array(features)
    return None

def main():
    """Extract features from Phoenix dataset PNG frames."""

    # Paths
    base_dir = Path('data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px')
    output_dir = Path('data/teacher_features/mediapipe_full')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'dev', 'test']:
        logger.info(f"Processing {split} split...")
        split_dir = base_dir / split

        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist")
            continue

        # Get all sample directories
        sample_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        # Process first 10 samples for testing
        sample_dirs = sample_dirs[:10]

        logger.info(f"Found {len(sample_dirs)} samples in {split}")

        for sample_dir in tqdm(sample_dirs, desc=f"Processing {split}"):
            sample_name = sample_dir.name

            # Process frames
            features = process_frame_sequence(sample_dir)

            if features is not None:
                # Save features
                output_file = output_dir / f"{split}_{sample_name}.npy"
                np.save(output_file, features)
                logger.debug(f"Saved features for {sample_name}: shape {features.shape}")
            else:
                logger.warning(f"Failed to extract features for {sample_name}")

    logger.info("Feature extraction complete!")

    # Check what was created
    created_files = list(output_dir.glob('*.npy'))
    logger.info(f"Created {len(created_files)} feature files in {output_dir}")

if __name__ == "__main__":
    main()