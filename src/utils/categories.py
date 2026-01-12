"""ADE20K category definitions."""

from typing import Dict, List, Tuple

# ADE20K 150 categories
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

# Alias for compatibility
ADE20K_CATEGORY_NAMES: List[str] = list(ADE20K_CATEGORIES)

# Number of classes (excluding background/ignore)
ADE20K_NUM_CLASSES: int = 150

# Ignore index for segmentation (pixels to ignore during evaluation)
ADE20K_IGNORE_INDEX: int = 255

# 15 corruption types for ADE20K-C
ADE20K_CORRUPTIONS: Tuple[str, ...] = (
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
)

# Severity levels (1-5, where 5 is most severe)
ADE20K_SEVERITIES: Tuple[int, ...] = (1, 2, 3, 4, 5)


def get_category_name(category_id: int) -> str:
    """Get category name by ID (0-indexed).

    Args:
        category_id: Category index (0-149)

    Returns:
        Category name string
    """
    if 0 <= category_id < len(ADE20K_CATEGORIES):
        return ADE20K_CATEGORIES[category_id]
    raise ValueError(f"Invalid category ID: {category_id}")


def get_category_id(category_name: str) -> int:
    """Get category ID by name.

    Args:
        category_name: Category name string

    Returns:
        Category index (0-149)
    """
    try:
        return ADE20K_CATEGORIES.index(category_name)
    except ValueError:
        raise ValueError(f"Unknown category name: {category_name}")


def get_category_mapping() -> Dict[int, str]:
    """Get mapping from category ID to name.

    Returns:
        Dictionary mapping ID to category name
    """
    return {i: name for i, name in enumerate(ADE20K_CATEGORIES)}
