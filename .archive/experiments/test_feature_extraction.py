"""
Test Feature Extraction Pipeline

Quick test to verify the feature extraction pipeline works before running on full dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import glob
from experiments.extract_cnn_features import GoogLeNetFeatureExtractor, load_annotations


def test_single_video():
    """Test feature extraction on a single video"""
    print("="*60)
    print("TEST 1: Single Video Feature Extraction")
    print("="*60)

    # Initialize extractor
    print("\n1. Initializing GoogLeNet feature extractor...")
    extractor = GoogLeNetFeatureExtractor()
    print(f"   Device: {extractor.device}")
    print(f"   Model loaded: [OK]")

    # Load annotations to get a sample video
    print("\n2. Loading sample video from dev set...")
    samples = load_annotations('dev')
    if len(samples) == 0:
        print("   ERROR: No samples found in dev set")
        return False

    sample = samples[0]
    video_id = sample['id']
    folder = sample['folder']
    annotation = sample['annotation']

    print(f"   Video ID: {video_id}")
    print(f"   Annotation: {annotation}")

    # Build frame pattern
    base_path = "data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/dev"
    frame_pattern = os.path.join(base_path, folder)

    print(f"\n3. Checking frames...")
    print(f"   Frame pattern: {frame_pattern}")

    frame_paths = sorted(glob.glob(frame_pattern))
    print(f"   Found {len(frame_paths)} frames")

    if len(frame_paths) == 0:
        print("   ERROR: No frames found")
        return False

    print(f"   First frame: {frame_paths[0]}")
    print(f"   Last frame: {frame_paths[-1]}")

    # Extract features
    print(f"\n4. Extracting features...")
    features = extractor.extract_video_features(frame_pattern)

    if features is None:
        print("   ERROR: Feature extraction failed")
        return False

    print(f"   [OK] Features extracted successfully!")
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature dtype: {features.dtype}")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Feature mean: {features.mean():.3f}")
    print(f"   Feature std: {features.std():.3f}")

    # Verify dimensions
    expected_frames = len(frame_paths)
    expected_dim = 1024

    if features.shape[0] != expected_frames:
        print(f"   WARNING: Expected {expected_frames} frames, got {features.shape[0]}")
        return False

    if features.shape[1] != expected_dim:
        print(f"   ERROR: Expected dimension {expected_dim}, got {features.shape[1]}")
        return False

    print("\n   [OK] All checks passed!")
    return True


def test_batch_extraction():
    """Test extraction on multiple videos"""
    print("\n" + "="*60)
    print("TEST 2: Batch Extraction (5 videos)")
    print("="*60)

    extractor = GoogLeNetFeatureExtractor()
    samples = load_annotations('dev')[:5]  # First 5 videos

    print(f"\n1. Processing {len(samples)} videos...")

    success_count = 0
    total_frames = 0
    feature_shapes = []

    for i, sample in enumerate(samples):
        video_id = sample['id']
        folder = sample['folder']

        base_path = "data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/dev"
        frame_pattern = os.path.join(base_path, folder)

        features = extractor.extract_video_features(frame_pattern)

        if features is not None:
            success_count += 1
            total_frames += len(features)
            feature_shapes.append(features.shape)
            print(f"   [{i+1}/5] {video_id}: {features.shape} [OK]")
        else:
            print(f"   [{i+1}/5] {video_id}: FAILED [FAIL]")

    print(f"\n2. Summary:")
    print(f"   Success rate: {success_count}/{len(samples)}")
    print(f"   Total frames: {total_frames}")
    print(f"   Avg frames per video: {total_frames/success_count:.1f}")
    print(f"   All dimensions 1024: {all(s[1] == 1024 for s in feature_shapes)}")

    return success_count == len(samples)


def test_gpu_availability():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("TEST 3: GPU Availability")
    print("="*60)

    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"3. CUDA version: {torch.version.cuda}")
        print(f"4. GPU count: {torch.cuda.device_count()}")
        print(f"5. GPU name: {torch.cuda.get_device_name(0)}")

        # Test GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"6. GPU memory: {gpu_mem:.1f} GB")

        if gpu_mem < 4:
            print("\n   WARNING: Low GPU memory. Feature extraction may be slow.")
            print("   Consider using CPU or reducing batch size.")
    else:
        print("\n   INFO: No GPU available. Using CPU.")
        print("   Feature extraction will be slower (~3-4x).")

    return True


def test_disk_space():
    """Test available disk space"""
    print("\n" + "="*60)
    print("TEST 4: Disk Space Check")
    print("="*60)

    import shutil

    # Check space in data directory
    total, used, free = shutil.disk_usage("data/")

    print(f"\n1. Total disk space: {total / 1e9:.1f} GB")
    print(f"2. Used space: {used / 1e9:.1f} GB")
    print(f"3. Free space: {free / 1e9:.1f} GB")

    # Estimate required space
    # ~3000 videos * 100 frames * 1024 features * 4 bytes = ~1.2 GB per split
    estimated_gb = 3 * 1.5  # 3 splits * 1.5 GB

    print(f"\n4. Estimated space needed: ~{estimated_gb:.1f} GB")

    if free / 1e9 < estimated_gb * 1.5:
        print("\n   WARNING: Low disk space!")
        print(f"   Recommended: {estimated_gb * 1.5:.1f} GB free")
        return False

    print(f"\n   [OK] Sufficient disk space available")
    return True


def test_dependencies():
    """Test all dependencies are installed"""
    print("\n" + "="*60)
    print("TEST 5: Dependency Check")
    print("="*60)

    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'h5py': 'HDF5 for Python',
        'PIL': 'Pillow',
        'tqdm': 'Progress bars',
        'numpy': 'NumPy'
    }

    all_installed = True

    for module_name, description in dependencies.items():
        try:
            __import__(module_name)
            print(f"   [OK] {description:20s} ({module_name})")
        except ImportError:
            print(f"   [FAIL] {description:20s} ({module_name}) - NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\n   ERROR: Missing dependencies. Install with:")
        print("   pip install torch torchvision h5py pillow tqdm numpy")
        return False

    print("\n   [OK] All dependencies installed")
    return True


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# Feature Extraction Pipeline - Test Suite")
    print("#"*60)

    results = {}

    # Test dependencies first
    results['dependencies'] = test_dependencies()

    if not results['dependencies']:
        print("\n" + "="*60)
        print("FATAL: Missing dependencies. Install required packages first.")
        print("="*60)
        return

    # Test GPU
    results['gpu'] = test_gpu_availability()

    # Test disk space
    results['disk_space'] = test_disk_space()

    # Test single video
    results['single_video'] = test_single_video()

    # Test batch extraction
    if results['single_video']:
        results['batch'] = test_batch_extraction()
    else:
        print("\n   Skipping batch test (single video test failed)")
        results['batch'] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("[PASS] ALL TESTS PASSED - Ready for feature extraction!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run full extraction:")
        print("   python experiments/extract_cnn_features.py --split train")
        print("\n2. Monitor progress (will take 2-4 hours)")
        print("\n3. Validate results:")
        print("   python experiments/extract_cnn_features.py --split train --validate_only")
    else:
        print("[FAIL] SOME TESTS FAILED - Please fix issues before extraction")
        print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
