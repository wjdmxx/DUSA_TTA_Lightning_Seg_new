"""Category definitions for ADE20K, Cityscapes, and ACDC."""

from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# ADE20K (150 classes)
# =============================================================================

ADE20K_CATEGORIES: Tuple[str, ...] = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
    'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk',
    'person', 'earth', 'door', 'table', 'mountain', 'plant',
    'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
    'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
    'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
    'screen door', 'stairway', 'river', 'bridge', 'bookcase',
    'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
    'bench', 'countertop', 'stove', 'palm', 'kitchen island',
    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
    'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth',
    'television receiver', 'airplane', 'dirt track', 'apparel',
    'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
    'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
    'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
    'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
    'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
    'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
    'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
    'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
    'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag'
)

ADE20K_CATEGORY_NAMES: List[str] = list(ADE20K_CATEGORIES)
ADE20K_NUM_CLASSES: int = 150
ADE20K_IGNORE_INDEX: int = 255

ADE20K_CORRUPTIONS: Tuple[str, ...] = (
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression',
)

ADE20K_SEVERITIES: Tuple[int, ...] = (1, 2, 3, 4, 5)


# =============================================================================
# Cityscapes (19 classes)
# =============================================================================

CITYSCAPES_CATEGORIES: Tuple[str, ...] = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle',
)

CITYSCAPES_CATEGORY_NAMES: List[str] = list(CITYSCAPES_CATEGORIES)
CITYSCAPES_NUM_CLASSES: int = 19
CITYSCAPES_IGNORE_INDEX: int = 255

# Cityscapes labelId -> trainId mapping
# All unlisted labelIds map to 255 (ignore)
CITYSCAPES_LABEL_ID_TO_TRAIN_ID: Dict[int, int] = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}


def create_cityscapes_label_mapping() -> np.ndarray:
    """Create a lookup table for Cityscapes labelId -> trainId.

    Returns:
        np.ndarray of shape [256] where mapping[labelId] = trainId.
        Unmapped labels map to 255 (ignore).
    """
    mapping = np.full(256, 255, dtype=np.uint8)
    for label_id, train_id in CITYSCAPES_LABEL_ID_TO_TRAIN_ID.items():
        mapping[label_id] = train_id
    return mapping


# Same 15 corruptions as ADE20K-C (standard benchmark set)
CITYSCAPES_CORRUPTIONS: Tuple[str, ...] = (
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression',
)


# =============================================================================
# ACDC (uses Cityscapes 19 classes)
# =============================================================================

ACDC_CONDITIONS: Tuple[str, ...] = (
    'fog', 'night', 'rain', 'snow',
)

ACDC_NUM_CLASSES: int = 19  # Same as Cityscapes
ACDC_IGNORE_INDEX: int = 255


# =============================================================================
# Helper functions
# =============================================================================

def get_category_name(category_id: int, dataset: str = "ade20k") -> str:
    """Get category name by ID (0-indexed).

    Args:
        category_id: Category index
        dataset: "ade20k" or "cityscapes"

    Returns:
        Category name string
    """
    categories = ADE20K_CATEGORIES if dataset == "ade20k" else CITYSCAPES_CATEGORIES
    if 0 <= category_id < len(categories):
        return categories[category_id]
    raise ValueError(f"Invalid category ID: {category_id} for {dataset}")


def get_category_id(category_name: str, dataset: str = "ade20k") -> int:
    """Get category ID by name.

    Args:
        category_name: Category name string
        dataset: "ade20k" or "cityscapes"

    Returns:
        Category index
    """
    categories = ADE20K_CATEGORIES if dataset == "ade20k" else CITYSCAPES_CATEGORIES
    try:
        return categories.index(category_name)
    except ValueError:
        raise ValueError(f"Unknown category name: {category_name} for {dataset}")


def get_category_mapping(dataset: str = "ade20k") -> Dict[int, str]:
    """Get mapping from category ID to name.

    Args:
        dataset: "ade20k" or "cityscapes"

    Returns:
        Dictionary mapping ID to category name
    """
    categories = ADE20K_CATEGORIES if dataset == "ade20k" else CITYSCAPES_CATEGORIES
    return {i: name for i, name in enumerate(categories)}

