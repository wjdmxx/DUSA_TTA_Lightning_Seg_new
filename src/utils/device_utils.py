"""
Device utilities for multi-GPU model distribution.
Handles model parallelism for large models like SD3.
"""

from typing import Dict, Optional, Union, Any, List
import torch
import torch.nn as nn


class WrappedDeviceModel(nn.Module):
    """
    Wraps a model and fixes it to a specific device.
    Automatically moves inputs to the target device before forward pass.
    """
    
    def __init__(self, model: nn.Module, device: Union[str, torch.device]):
        """
        Args:
            model: The model to wrap
            device: Target device for the model
        """
        super().__init__()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
    def forward(self, *args, **kwargs):
        """Forward pass with automatic data casting to target device."""
        args = self._cast_to_device(args)
        kwargs = self._cast_to_device(kwargs)
        return self.model(*args, **kwargs)
    
    def _cast_to_device(self, data: Any) -> Any:
        """Recursively move data to target device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._cast_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {k: self._cast_to_device(v) for k, v in data.items()}
        else:
            return data
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class WrappedPipelineModel(nn.Module):
    """
    Wraps a model with accelerate's device_map for pipeline parallelism.
    Useful for distributing large models across multiple GPUs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device_map: Union[str, Dict[str, str]],
        input_device: Union[str, torch.device] = "cuda:0",
        output_device: Union[str, torch.device] = "cuda:0",
    ):
        """
        Args:
            model: The model to distribute
            device_map: Either "auto", "balanced", or a dict mapping layer names to devices
            input_device: Device for input tensors
            output_device: Device for output tensors
        """
        super().__init__()
        self.input_device = torch.device(input_device)
        self.output_device = torch.device(output_device)
        
        # Use accelerate for model distribution
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
            
            if isinstance(device_map, str) and device_map in ["auto", "balanced"]:
                # Infer device map automatically
                max_memory = get_balanced_memory(model)
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=self._get_no_split_modules(model),
                )
            
            self.model = dispatch_model(model, device_map=device_map)
            self.device_map = device_map
            
        except ImportError:
            raise ImportError(
                "accelerate is required for pipeline parallelism. "
                "Install it with: pip install accelerate"
            )
    
    def _get_no_split_modules(self, model: nn.Module) -> List[str]:
        """Get module classes that should not be split across devices."""
        # Common transformer block names that shouldn't be split
        no_split = []
        if hasattr(model, "_no_split_modules"):
            no_split = model._no_split_modules
        return no_split
    
    def forward(self, *args, **kwargs):
        """Forward pass with device handling."""
        # Move inputs to input device
        args = self._cast_to_device(args, self.input_device)
        kwargs = self._cast_to_device(kwargs, self.input_device)
        
        output = self.model(*args, **kwargs)
        
        # Move output to output device
        return self._cast_to_device(output, self.output_device)
    
    def _cast_to_device(self, data: Any, device: torch.device) -> Any:
        """Recursively move data to target device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._cast_to_device(item, device) for item in data)
        elif isinstance(data, dict):
            return {k: self._cast_to_device(v, device) for k, v in data.items()}
        else:
            return data


def dispatch_model_to_devices(
    model: nn.Module,
    device_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Dispatch a model to specified device(s) based on configuration.
    
    Args:
        model: The model to dispatch
        device_config: Configuration dict with keys:
            - device: Single device string (e.g., "cuda:0")
            - device_map: Dict or string for multi-GPU distribution
            - input_device: Device for inputs (for pipeline parallelism)
            - output_device: Device for outputs (for pipeline parallelism)
            
    Returns:
        Wrapped model with device handling
    """
    if device_config is None:
        device_config = {"device": "cuda:0"}
    
    if "device_map" in device_config:
        # Use pipeline parallelism
        return WrappedPipelineModel(
            model,
            device_map=device_config["device_map"],
            input_device=device_config.get("input_device", "cuda:0"),
            output_device=device_config.get("output_device", "cuda:0"),
        )
    elif "device" in device_config:
        # Use single device
        return WrappedDeviceModel(model, device_config["device"])
    else:
        # Default to cuda:0
        return WrappedDeviceModel(model, "cuda:0")


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def create_balanced_device_map(
    num_layers: int,
    num_gpus: int,
    prefix: str = "transformer_blocks",
    input_layers: Optional[List[str]] = None,
    output_layers: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Create a balanced device map for transformer models.
    
    Args:
        num_layers: Number of transformer blocks
        num_gpus: Number of GPUs to use
        prefix: Prefix for layer names
        input_layers: Layer names to place on first GPU
        output_layers: Layer names to place on last GPU
        
    Returns:
        Device map dict
    """
    device_map = {}
    
    # Input layers on first GPU
    if input_layers:
        for layer in input_layers:
            device_map[layer] = "cuda:0"
    
    # Distribute transformer blocks evenly
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    current_layer = 0
    for gpu_idx in range(num_gpus):
        # Add extra layer to early GPUs if there's a remainder
        n_layers = layers_per_gpu + (1 if gpu_idx < remainder else 0)
        for _ in range(n_layers):
            device_map[f"{prefix}.{current_layer}"] = f"cuda:{gpu_idx}"
            current_layer += 1
    
    # Output layers on last GPU
    if output_layers:
        for layer in output_layers:
            device_map[layer] = f"cuda:{num_gpus - 1}"
    
    return device_map
