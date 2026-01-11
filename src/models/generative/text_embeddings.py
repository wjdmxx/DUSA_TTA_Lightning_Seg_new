"""
Text embeddings precomputation for SD3.
Pre-computes class embeddings to avoid redundant computation during TTA.
"""

from typing import List, Tuple, Optional
import torch
from tqdm import tqdm


@torch.no_grad()
def prepare_class_embeddings_sd3(
    pipe,
    class_names: Tuple[str, ...],
    prompt_template: str = "a photo of a {}",
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute class embeddings using SD3 pipeline's encode_prompt.
    
    Args:
        pipe: SD3 pipeline (StableDiffusion3Pipeline)
        class_names: Tuple of class names
        prompt_template: Template string with {} placeholder for class name
        device: Device to run encoding on
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of:
            - class_embeddings: (num_classes, seq_len, hidden_dim) prompt embeddings
            - pooled_embeddings: (num_classes, pooled_dim) pooled prompt embeddings
    """
    prompt_template = prompt_template.replace("_", " ")
    
    prompt_embeds_list = []
    pooled_embeds_list = []
    
    # Move pipe to device for encoding
    original_device = next(pipe.text_encoder.parameters()).device
    pipe.to(device)
    
    iterator = tqdm(class_names, desc="Encoding class prompts") if show_progress else class_names
    
    for class_name in iterator:
        prompt = prompt_template.format(class_name)
        
        # Use SD3's encode_prompt method
        # Returns: prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
        )
        
        prompt_embeds_list.append(prompt_embeds.cpu())
        pooled_embeds_list.append(pooled_prompt_embeds.cpu())
    
    # Move pipe back to original device to free GPU memory
    pipe.to("cpu")
    torch.cuda.empty_cache()
    
    # Stack embeddings: (num_classes, seq_len, hidden_dim)
    class_embeddings = torch.cat(prompt_embeds_list, dim=0)
    # Stack pooled embeddings: (num_classes, pooled_dim)
    pooled_embeddings = torch.cat(pooled_embeds_list, dim=0)
    
    return class_embeddings, pooled_embeddings


def save_embeddings(
    class_embeddings: torch.Tensor,
    pooled_embeddings: torch.Tensor,
    save_path: str,
) -> None:
    """
    Save precomputed embeddings to disk.
    
    Args:
        class_embeddings: Prompt embeddings tensor
        pooled_embeddings: Pooled embeddings tensor
        save_path: Path to save the embeddings
    """
    torch.save({
        "class_embeddings": class_embeddings,
        "pooled_embeddings": pooled_embeddings,
    }, save_path)


def load_embeddings(
    load_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load precomputed embeddings from disk.
    
    Args:
        load_path: Path to load the embeddings from
        
    Returns:
        Tuple of (class_embeddings, pooled_embeddings)
    """
    data = torch.load(load_path)
    return data["class_embeddings"], data["pooled_embeddings"]
