"""
Data Quality Monitoring Module

ROLE IN OBSERVABILITY LAYER:
-----------------------------
This module monitors the quality of input brain MRI images to detect data issues
that could degrade model performance. It supports:
- MLO 8.5: Production-grade code with quality checks
- MLO 8.7: Deployment monitoring (detecting bad inputs in production)
- MLO 14.1: Demonstrating data quality awareness in final presentation

WHY DATA QUALITY MATTERS:
--------------------------
Machine learning models are only as good as their input data. For medical imaging:

1. BRIGHTNESS ISSUES:
   - Too dark/bright images → poor model performance
   - Indicates scanner calibration issues or preprocessing errors
   - Can detect distribution shift between training and production data

2. CORRUPTED/INVALID IMAGES:
   - File corruption during transfer
   - Unsupported formats
   - Truncated files
   - Can cause inference failures or silent errors

3. DISTRIBUTION SHIFT:
   - Production images differ from training images
   - Different scanners, protocols, patient populations
   - Detected by comparing brightness distributions, dimensions, etc.

DATA QUALITY IN PRODUCTION:
----------------------------
When deployed, this module can:
- Reject images that don't meet quality standards
- Alert radiologists to potential scanner issues
- Trigger model retraining if data distribution shifts significantly
- Log quality metrics for audit trails

ARCHITECTURE-AGNOSTIC:
----------------------
This module operates on RAW IMAGES before they reach the model.
It makes NO assumptions about what model processes these images.
Works with any CNN architecture or preprocessing pipeline.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings


# =============================================================================
# IMAGE QUALITY CHECKS
# =============================================================================

def check_image_brightness(
    image: Union[np.ndarray, Image.Image, str, Path],
    return_histogram: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Calculate average brightness of an image.

    Brightness is a simple but effective proxy for image quality and consistency.
    Significant brightness changes between training and production can indicate
    data distribution shift or scanner calibration issues.

    Args:
        image (array, PIL.Image, or str/Path): Input image.
            - If numpy array: assumed to be in [0, 255] or [0, 1] range
            - If PIL Image: will be converted to numpy
            - If string/Path: will be loaded from file
        return_histogram (bool): If True, also return brightness histogram.

    Returns:
        float: Average brightness (0-255 scale).
        tuple (optional): (brightness, histogram) if return_histogram=True.

    Example:
        >>> # From file
        >>> brightness = check_image_brightness("mri_scan.jpg")
        >>> print(f"Average brightness: {brightness:.1f}")
        >>>
        >>> # From numpy array
        >>> img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> brightness = check_image_brightness(img_array)
        >>>
        >>> # With histogram
        >>> brightness, hist = check_image_brightness("scan.jpg", return_histogram=True)

    Clinical Note:
        MRI images typically have specific brightness ranges depending on the
        sequence type (T1, T2, FLAIR). Significant deviations may indicate
        preprocessing issues or different scanner protocols.
    """
    # Load image if it's a path
    if isinstance(image, (str, Path)):
        try:
            image = Image.open(image)
        except Exception as e:
            raise ValueError(f"Could not load image from {image}: {e}")

    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure image is a numpy array at this point
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be numpy array, PIL Image, or path. Got {type(image)}")

    # Convert to grayscale if color (for MRI, should already be grayscale)
    if image.ndim == 3:
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    # Normalize to 0-255 if in 0-1 range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Calculate average brightness
    brightness = float(np.mean(image))

    if return_histogram:
        # Calculate histogram (256 bins for 0-255 range)
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        return brightness, histogram
    else:
        return brightness


def detect_corrupted_image(
    image_path: Union[str, Path],
    expected_min_size: Tuple[int, int] = (64, 64),
    expected_max_size: Tuple[int, int] = (4096, 4096)
) -> Tuple[bool, str]:
    """
    Check if an image file is corrupted or invalid.

    This function performs multiple checks to ensure image integrity:
    - File exists and is readable
    - Valid image format
    - Expected dimensions
    - Not truncated or corrupted

    Args:
        image_path (str or Path): Path to image file.
        expected_min_size (tuple): Minimum expected (width, height) in pixels.
            Default: (64, 64). Images smaller than this are likely corrupted.
        expected_max_size (tuple): Maximum expected (width, height) in pixels.
            Default: (4096, 4096). Images larger might be unprocessed raw data.

    Returns:
        tuple: (is_corrupted: bool, reason: str)
            - (False, "OK") if image is valid
            - (True, reason) if image is corrupted/invalid with explanation

    Example:
        >>> is_bad, reason = detect_corrupted_image("scan.jpg")
        >>> if is_bad:
        >>>     print(f"Image rejected: {reason}")
        >>> else:
        >>>     print("Image OK")
        >>>
        >>> # Check batch of images
        >>> for img_path in image_paths:
        >>>     is_corrupted, reason = detect_corrupted_image(img_path)
        >>>     if is_corrupted:
        >>>         logging.warning(f"{img_path}: {reason}")

    Production Use:
        In deployment, use this to filter out bad images before inference:
        - Log corrupted images for investigation
        - Alert if corruption rate exceeds threshold (e.g., >1%)
        - Notify data provider if systematic corruption detected
    """
    image_path = Path(image_path)

    # Check 1: File exists
    if not image_path.exists():
        return True, f"File does not exist: {image_path}"

    # Check 2: File is readable
    if not image_path.is_file():
        return True, f"Not a file: {image_path}"

    # Check 3: File size is reasonable (not empty, not too large)
    file_size = image_path.stat().st_size
    if file_size == 0:
        return True, "File is empty (0 bytes)"
    if file_size > 100_000_000:  # 100 MB is unusually large for MRI slice
        return True, f"File suspiciously large: {file_size / 1_000_000:.1f} MB"

    # Check 4: Can be opened as image
    try:
        with Image.open(image_path) as img:
            # Check 5: Has valid dimensions
            width, height = img.size

            if width < expected_min_size[0] or height < expected_min_size[1]:
                return True, f"Image too small: {width}x{height} (expected >= {expected_min_size})"

            if width > expected_max_size[0] or height > expected_max_size[1]:
                return True, f"Image too large: {width}x{height} (expected <= {expected_max_size})"

            # Check 6: Try to load pixel data (catches truncated images)
            try:
                img.load()
            except Exception as e:
                return True, f"Image data corrupted: {e}"

            # Check 7: Verify image mode is sensible for MRI
            # MRI scans are typically grayscale ('L') or RGB ('RGB')
            if img.mode not in ['L', 'RGB', 'RGBA', 'LA']:
                warnings.warn(f"Unusual image mode '{img.mode}' for MRI image")

    except Exception as e:
        return True, f"Cannot open as image: {e}"

    # All checks passed
    return False, "OK"


def analyze_dataset_quality(
    image_paths: List[Union[str, Path]],
    sample_size: Optional[int] = None
) -> Dict[str, any]:
    """
    Analyze quality metrics across an entire dataset.

    This function provides a comprehensive quality report for a collection of images,
    useful for:
    - Validating new datasets before training
    - Comparing training vs. production data distributions
    - Detecting data drift over time

    Args:
        image_paths (list): List of paths to image files.
        sample_size (int, optional): If provided, randomly sample this many images
            for analysis (useful for large datasets). If None, analyze all images.

    Returns:
        dict: Quality report containing:
            - 'total_images': Number of images analyzed
            - 'corrupted_count': Number of corrupted images
            - 'corrupted_percentage': Percentage of corrupted images
            - 'corrupted_images': List of (path, reason) for corrupted images
            - 'brightness_mean': Average brightness across dataset
            - 'brightness_std': Standard deviation of brightness
            - 'brightness_min': Minimum brightness
            - 'brightness_max': Maximum brightness
            - 'brightness_values': List of all brightness values
            - 'dimensions': List of (width, height) for all valid images

    Example:
        >>> from pathlib import Path
        >>> image_paths = list(Path("data/train").glob("*.jpg"))
        >>>
        >>> # Analyze all images
        >>> report = analyze_dataset_quality(image_paths)
        >>> print(f"Dataset size: {report['total_images']}")
        >>> print(f"Corrupted: {report['corrupted_percentage']:.1f}%")
        >>> print(f"Avg brightness: {report['brightness_mean']:.1f} ± {report['brightness_std']:.1f}")
        >>>
        >>> # Analyze sample (faster for large datasets)
        >>> report = analyze_dataset_quality(image_paths, sample_size=1000)

    Production Monitoring:
        Run this periodically on production data and compare to baseline:
        - Alert if corruption rate increases
        - Alert if brightness distribution shifts significantly
        - Log reports for audit trail
    """
    # Sample if requested
    if sample_size is not None and sample_size < len(image_paths):
        import random
        image_paths = random.sample(image_paths, sample_size)

    # Initialize tracking
    brightness_values = []
    corrupted_images = []
    dimensions = []
    total_images = len(image_paths)

    print(f"[Data Quality] Analyzing {total_images} images...")

    # Analyze each image
    for i, img_path in enumerate(image_paths):
        # Progress indicator for large datasets
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{total_images}")

        # Check for corruption
        is_corrupted, reason = detect_corrupted_image(img_path)

        if is_corrupted:
            corrupted_images.append((str(img_path), reason))
            continue  # Skip brightness analysis for corrupted images

        # Brightness analysis for valid images
        try:
            brightness = check_image_brightness(img_path)
            brightness_values.append(brightness)

            # Get dimensions
            with Image.open(img_path) as img:
                dimensions.append(img.size)

        except Exception as e:
            # Shouldn't happen if detect_corrupted_image passed, but be safe
            corrupted_images.append((str(img_path), f"Failed brightness analysis: {e}"))

    # Compile statistics
    brightness_array = np.array(brightness_values) if brightness_values else np.array([])

    report = {
        'total_images': total_images,
        'corrupted_count': len(corrupted_images),
        'corrupted_percentage': (len(corrupted_images) / total_images * 100) if total_images > 0 else 0,
        'corrupted_images': corrupted_images,
        'valid_images': total_images - len(corrupted_images),
    }

    # Brightness statistics (only for valid images)
    if len(brightness_array) > 0:
        report.update({
            'brightness_mean': float(np.mean(brightness_array)),
            'brightness_std': float(np.std(brightness_array)),
            'brightness_min': float(np.min(brightness_array)),
            'brightness_max': float(np.max(brightness_array)),
            'brightness_median': float(np.median(brightness_array)),
            'brightness_values': brightness_values,
        })
    else:
        report.update({
            'brightness_mean': None,
            'brightness_std': None,
            'brightness_min': None,
            'brightness_max': None,
            'brightness_median': None,
            'brightness_values': [],
        })

    # Dimension statistics
    if dimensions:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        report.update({
            'dimension_mean': (np.mean(widths), np.mean(heights)),
            'dimension_min': (min(widths), min(heights)),
            'dimension_max': (max(widths), max(heights)),
            'unique_dimensions': len(set(dimensions)),
        })

    print(f"[Data Quality] Analysis complete: {report['valid_images']} valid, "
          f"{report['corrupted_count']} corrupted")

    return report


def print_quality_report(report: Dict[str, any]) -> None:
    """
    Pretty-print a quality report from analyze_dataset_quality().

    Args:
        report (dict): Report from analyze_dataset_quality().

    Example:
        >>> report = analyze_dataset_quality(image_paths)
        >>> print_quality_report(report)
    """
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    # Overall statistics
    print(f"\nDATASET OVERVIEW:")
    print(f"  Total images:    {report['total_images']:>6}")
    print(f"  Valid images:    {report['valid_images']:>6}")
    print(f"  Corrupted:       {report['corrupted_count']:>6} ({report['corrupted_percentage']:>5.2f}%)")

    # Quality assessment
    if report['corrupted_percentage'] > 5:
        print(f"  ⚠ WARNING: High corruption rate! Investigate data pipeline.")
    elif report['corrupted_percentage'] > 0:
        print(f"  ⚠ Some corrupted images found. Check logs for details.")
    else:
        print(f"  ✓ No corrupted images detected.")

    # Brightness statistics
    if report['brightness_mean'] is not None:
        print(f"\nBRIGHTNESS STATISTICS:")
        print(f"  Mean:   {report['brightness_mean']:>6.1f} ± {report['brightness_std']:.1f}")
        print(f"  Median: {report['brightness_median']:>6.1f}")
        print(f"  Range:  {report['brightness_min']:>6.1f} to {report['brightness_max']:.1f}")

        # Brightness distribution assessment
        mean = report['brightness_mean']
        std = report['brightness_std']
        if std > 50:
            print(f"  ⚠ High brightness variance - images may need normalization")
        else:
            print(f"  ✓ Consistent brightness distribution")

    # Dimension statistics
    if 'dimension_mean' in report:
        print(f"\nIMAGE DIMENSIONS:")
        print(f"  Mean size:    {report['dimension_mean'][0]:.0f} x {report['dimension_mean'][1]:.0f}")
        print(f"  Min size:     {report['dimension_min'][0]} x {report['dimension_min'][1]}")
        print(f"  Max size:     {report['dimension_max'][0]} x {report['dimension_max'][1]}")
        print(f"  Unique sizes: {report['unique_dimensions']}")

        if report['unique_dimensions'] > 1:
            print(f"  ⚠ Multiple image sizes detected - ensure consistent preprocessing")

    # List corrupted images if any
    if report['corrupted_images']:
        print(f"\nCORRUPTED IMAGES (first 10):")
        for path, reason in report['corrupted_images'][:10]:
            print(f"  {Path(path).name}: {reason}")
        if len(report['corrupted_images']) > 10:
            print(f"  ... and {len(report['corrupted_images']) - 10} more")

    print("=" * 70 + "\n")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_dataset_distributions(
    report1: Dict[str, any],
    report2: Dict[str, any],
    name1: str = "Dataset 1",
    name2: str = "Dataset 2"
) -> Dict[str, any]:
    """
    Compare quality metrics between two datasets.

    Useful for detecting distribution shift between:
    - Training vs. validation data
    - Training vs. production data
    - Different time periods in production

    Args:
        report1 (dict): Quality report from first dataset.
        report2 (dict): Quality report from second dataset.
        name1 (str): Name for first dataset (for display).
        name2 (str): Name for second dataset (for display).

    Returns:
        dict: Comparison results with shift indicators.

    Example:
        >>> train_report = analyze_dataset_quality(train_images)
        >>> prod_report = analyze_dataset_quality(production_images)
        >>>
        >>> comparison = compare_dataset_distributions(
        >>>     train_report, prod_report,
        >>>     name1="Training", name2="Production"
        >>> )
        >>>
        >>> if comparison['significant_brightness_shift']:
        >>>     print("WARNING: Production data brightness differs from training!")
    """
    comparison = {
        'dataset1_name': name1,
        'dataset2_name': name2,
    }

    # Brightness comparison
    if report1['brightness_mean'] is not None and report2['brightness_mean'] is not None:
        brightness_diff = abs(report1['brightness_mean'] - report2['brightness_mean'])
        brightness_diff_std = brightness_diff / max(report1['brightness_std'], report2['brightness_std'])

        comparison['brightness_mean_diff'] = brightness_diff
        comparison['brightness_diff_in_std'] = brightness_diff_std

        # Flag if brightness differs by more than 1 standard deviation
        comparison['significant_brightness_shift'] = brightness_diff_std > 1.0

    # Corruption rate comparison
    corruption_diff = abs(report1['corrupted_percentage'] - report2['corrupted_percentage'])
    comparison['corruption_rate_diff'] = corruption_diff
    comparison['significant_corruption_change'] = corruption_diff > 2.0  # >2% difference

    return comparison


# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
USAGE EXAMPLES FOR TEAM:
=========================

Example 1: Check single image quality
--------------------------------------
from observability import data_quality

# Check if image is corrupted
is_bad, reason = data_quality.detect_corrupted_image("scan.jpg")
if is_bad:
    print(f"Image rejected: {reason}")

# Check brightness
brightness = data_quality.check_image_brightness("scan.jpg")
print(f"Brightness: {brightness:.1f}")


Example 2: Validate entire dataset
-----------------------------------
from pathlib import Path
from observability import data_quality

# Get all images
image_paths = list(Path("data/train").glob("**/*.jpg"))

# Analyze quality
report = data_quality.analyze_dataset_quality(image_paths)
data_quality.print_quality_report(report)

# Save report for documentation
import json
with open("data_quality_report.json", "w") as f:
    json.dump(report, f, indent=2)


Example 3: Monitor production data
-----------------------------------
from observability import data_quality, alerts

# Analyze production batch
prod_report = data_quality.analyze_dataset_quality(production_images)

# Compare to training baseline
baseline_report = load_baseline_report()
comparison = data_quality.compare_dataset_distributions(
    baseline_report, prod_report,
    name1="Training", name2="Production"
)

# Alert if significant shift
if comparison['significant_brightness_shift']:
    alerts.send_alert("Data drift detected: brightness distribution shift")


Example 4: Pre-inference validation
------------------------------------
from observability import data_quality

def safe_predict(image_path, model):
    # Validate image before inference
    is_corrupted, reason = data_quality.detect_corrupted_image(image_path)
    if is_corrupted:
        raise ValueError(f"Invalid image: {reason}")

    # Check brightness is in expected range
    brightness = data_quality.check_image_brightness(image_path)
    if brightness < 20 or brightness > 240:
        warnings.warn(f"Unusual brightness: {brightness}")

    # Proceed with inference
    return model.predict(image_path)
"""
