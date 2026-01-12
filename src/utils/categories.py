"""Category definitions for segmentation datasets."""

from typing import Tuple

# ADE20K 150 categories
ADE_CATEGORIES: Tuple[str, ...] = (
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

# Number of classes for ADE20K
ADE_NUM_CLASSES = 150

# Cityscapes categories (for potential future use)
CITYSCAPES_CATEGORIES: Tuple[str, ...] = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
)

CITYSCAPES_NUM_CLASSES = 19


def get_category_names(dataset_name: str) -> Tuple[str, ...]:
    """Get category names for a given dataset.
    
    Args:
        dataset_name: Name of the dataset ('ade20k' or 'cityscapes')
        
    Returns:
        Tuple of category names
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ('ade20k', 'ade'):
        return ADE_CATEGORIES
    elif dataset_name == 'cityscapes':
        return CITYSCAPES_CATEGORIES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for a given dataset.
    
    Args:
        dataset_name: Name of the dataset ('ade20k' or 'cityscapes')
        
    Returns:
        Number of classes
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ('ade20k', 'ade'):
        return ADE_NUM_CLASSES
    elif dataset_name == 'cityscapes':
        return CITYSCAPES_NUM_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
