"""Multi-GPU device utilities using accelerate dispatch_model."""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices.

    Returns:
        List of GPU indices (based on CUDA_VISIBLE_DEVICES)
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def create_balanced_device_map(
    model: nn.Module,
    no_split_classes: Optional[List[str]] = None,
    max_memory: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Create a balanced device map for multi-GPU distribution.

    Uses accelerate's infer_auto_device_map with balanced memory allocation.

    Args:
        model: PyTorch model to distribute
        no_split_classes: List of module class names that should not be split
        max_memory: Optional memory limits per device

    Returns:
        Device map dictionary
    """
    try:
        from accelerate import infer_auto_device_map
        from accelerate.utils import get_balanced_memory
    except ImportError:
        raise ImportError("accelerate is required for multi-GPU support")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPUs available, using CPU")
        return {"": "cpu"}

    if num_gpus == 1:
        logger.info("Single GPU detected, placing entire model on cuda:0")
        return {"": "cuda:0"}

    # Get balanced memory allocation
    if max_memory is None:
        max_memory = get_balanced_memory(
            model,
            no_split_module_classes=no_split_classes or [],
        )

    # Infer device map
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_classes or [],
    )

    logger.info(f"Created balanced device map across {num_gpus} GPUs")
    return device_map


def setup_transformer_dispatch(
    transformer: nn.Module,
    device_map: Union[str, Dict[str, str]] = "balanced",
    input_device: str = "cuda:0",
    output_device: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Setup multi-GPU dispatch for SD3 transformer.

    Uses accelerate's dispatch_model to distribute the transformer
    across multiple GPUs.

    Args:
        transformer: SD3Transformer2DModel to distribute
        device_map: Either "balanced", "auto", or explicit mapping dict
        input_device: Device for input tensors
        output_device: Device for output tensors (defaults to input_device)
        gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Distributed model wrapper
    """
    try:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory
    except ImportError:
        raise ImportError("accelerate is required for multi-GPU support")

    num_gpus = torch.cuda.device_count()
    if output_device is None:
        output_device = input_device

    # Handle single GPU case
    if num_gpus <= 1:
        logger.info("Single GPU mode - no dispatch needed")
        transformer = transformer.to(input_device)
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
        return transformer

    # SD3 transformer has JointTransformerBlock that should not be split
    no_split_classes = ["JointTransformerBlock"]

    # Create device map
    if isinstance(device_map, str):
        if device_map in ["balanced", "auto"]:
            # Get balanced memory allocation
            max_memory = get_balanced_memory(
                transformer,
                no_split_module_classes=no_split_classes,
            )
            device_map = infer_auto_device_map(
                transformer,
                max_memory=max_memory,
                no_split_module_classes=no_split_classes,
            )
            logger.info(f"Auto device map created: {device_map}")
        else:
            raise ValueError(f"Unknown device_map string: {device_map}")

    # Enable gradient checkpointing before dispatch
    if gradient_checkpointing:
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")

    # Dispatch model
    transformer = dispatch_model(
        transformer,
        device_map=device_map,
    )

    logger.info(f"Transformer dispatched across {num_gpus} GPUs")
    return transformer


class DeviceAwareModule(nn.Module):
    """Wrapper for modules that handles device management.

    This wrapper ensures that inputs are moved to the correct device
    before forward pass, and outputs are moved to the specified output device.
    """

    def __init__(
        self,
        module: nn.Module,
        input_device: str = "cuda:0",
        output_device: Optional[str] = None,
        output_dtype: torch.dtype = torch.float32,
    ):
        """Initialize the wrapper.

        Args:
            module: The model to wrap
            input_device: Device to move inputs to
            output_device: Device to move outputs to
            output_dtype: Data type for outputs
        """
        super().__init__()
        self.module = module
        self.input_device = input_device
        self.output_device = output_device or input_device
        self.output_dtype = output_dtype

    def forward(self, *args, **kwargs):
        """Forward with automatic device handling."""
        # Move inputs to input device
        args = tuple(self._to_device(a, self.input_device) for a in args)
        kwargs = {k: self._to_device(v, self.input_device) for k, v in kwargs.items()}

        # Forward through module
        output = self.module(*args, **kwargs)

        # Move outputs to output device and dtype
        return self._cast_output(output)

    def _to_device(self, tensor: Any, device: str) -> Any:
        """Move tensor to device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device)
        elif isinstance(tensor, (list, tuple)):
            return type(tensor)(self._to_device(t, device) for t in tensor)
        elif isinstance(tensor, dict):
            return {k: self._to_device(v, device) for k, v in tensor.items()}
        return tensor

    def _cast_output(self, output: Any) -> Any:
        """Cast output to target device and dtype."""
        if isinstance(output, torch.Tensor):
            return output.to(device=self.output_device, dtype=self.output_dtype)
        elif isinstance(output, (list, tuple)):
            return type(output)(self._cast_output(o) for o in output)
        elif isinstance(output, dict):
            return {k: self._cast_output(v) for k, v in output.items()}
        return output

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def ensure_same_device(*tensors: torch.Tensor) -> str:
    """Ensure all tensors are on the same device, return that device.

    Args:
        *tensors: Tensors to check

    Returns:
        Device string

    Raises:
        ValueError: If tensors are on different devices
    """
    devices = set()
    for t in tensors:
        if isinstance(t, torch.Tensor):
            devices.add(str(t.device))

    if len(devices) > 1:
        raise ValueError(f"Tensors are on different devices: {devices}")

    return devices.pop() if devices else "cpu"


def move_to_device(
    data: Any,
    device: Union[str, torch.device],
) -> Any:
    """Recursively move data to device.

    Args:
        data: Tensor, list, tuple, or dict to move
        device: Target device

    Returns:
        Data on target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(d, device) for d in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(d, device) for d in data)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    return data
