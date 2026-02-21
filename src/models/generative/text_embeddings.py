"""Text embedding management for SD3 with caching support."""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ...utils.categories import ADE20K_CATEGORIES, CITYSCAPES_CATEGORIES

logger = logging.getLogger(__name__)


class TextEmbeddingManager:
    """Manage pre-computed text embeddings for SD3.

    This class handles:
    - Computing text embeddings for all class names
    - Caching embeddings to disk
    - Loading cached embeddings
    """

    def __init__(
        self,
        config: DictConfig,
        pipe=None,
    ):
        """Initialize the embedding manager.

        Args:
            config: Configuration containing:
                - prompt_template: Template string with {} for class name
                - cache_dir: Directory for caching embeddings
            pipe: Optional StableDiffusion3Pipeline for computing embeddings
        """
        self.prompt_template = config.get("prompt_template", "a photo of a {}")
        self.cache_dir = Path(config.get("cache_dir", "./embeddings_cache"))

        # Determine dataset name for cache differentiation
        self.dataset_name = config.get("dataset", "ade20k").lower()

        # Determine class names: explicit list > dataset name > ADE20K default
        if "categories_list" in config:
            self.class_names = list(config.categories_list)
        elif self.dataset_name in ("cityscapes", "cityscapes-c", "acdc"):
            print(f"Using {self.dataset_name} categories for embedding generation")
            print(list(CITYSCAPES_CATEGORIES))
            self.class_names = list(CITYSCAPES_CATEGORIES)
        else:
            print("Using ADE20K categories for embedding generation")
            print(list(ADE20K_CATEGORIES))
            self.class_names = list(ADE20K_CATEGORIES)

        self.pipe = pipe

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Computed embeddings
        self._class_embeddings: Optional[torch.Tensor] = None
        self._pooled_embeddings: Optional[torch.Tensor] = None

    def _get_cache_path(self) -> Path:
        """Generate cache file path based on dataset and prompt template.

        Returns:
            Path to cache file
        """
        # Create hash of prompt template for unique filename
        hash_input = f"{self.dataset_name}_{self.prompt_template}"
        prompt_hash = hashlib.md5(
            hash_input.encode()
        ).hexdigest()[:8]
        # Use dataset-aware filename to avoid cache collisions between
        # ADE20K (150 classes) and Cityscapes/ACDC (19 classes)
        if self.dataset_name in ("cityscapes", "cityscapes-c", "acdc"):
            prefix = "cityscapes"
        else:
            prefix = "ade20k"
        return self.cache_dir / f"{prefix}_embeddings_{prompt_hash}.pt"

    def get_embeddings(
        self,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get class embeddings, loading from cache or computing.

        Args:
            device: Target device for embeddings

        Returns:
            Tuple of (class_embeddings, pooled_embeddings)
            - class_embeddings: [num_classes, seq_len, hidden_dim]
            - pooled_embeddings: [num_classes, hidden_dim]
        """
        if self._class_embeddings is not None:
            return (
                self._class_embeddings.to(device),
                self._pooled_embeddings.to(device),
            )

        cache_path = self._get_cache_path()

        if cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self._class_embeddings = cached["class_embeddings"]
            self._pooled_embeddings = cached["pooled_embeddings"]
        else:
            if self.pipe is None:
                raise ValueError(
                    "No cached embeddings found and no pipeline provided. "
                    f"Expected cache at: {cache_path}"
                )
            logger.info("Computing embeddings (this may take a while)...")
            self._compute_and_cache_embeddings(cache_path)

        return (
            self._class_embeddings.to(device),
            self._pooled_embeddings.to(device),
        )

    def _compute_and_cache_embeddings(self, cache_path: Path) -> None:
        """Compute embeddings for all classes and save to cache.

        Args:
            cache_path: Path to save embeddings
        """
        if self.pipe is None:
            raise ValueError("Pipeline required for computing embeddings")

        # Temporarily move pipe to GPU for encoding
        original_device = next(self.pipe.text_encoder.parameters()).device
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Move encoders to compute device
            self.pipe.text_encoder.to(compute_device)
            self.pipe.text_encoder_2.to(compute_device)
            self.pipe.text_encoder_3.to(compute_device)

            class_embeddings_list = []
            pooled_embeddings_list = []

            for class_name in tqdm(self.class_names, desc="Computing embeddings"):
                # Format prompt
                prompt = self.prompt_template.format(class_name.replace("_", " "))
                
                # Encode prompt using SD3 pipeline
                with torch.no_grad():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = self.pipe.encode_prompt(
                        prompt=prompt,
                        prompt_2=None,
                        prompt_3=None,
                        device=compute_device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )

                class_embeddings_list.append(prompt_embeds.cpu())
                pooled_embeddings_list.append(pooled_prompt_embeds.cpu())

            # Stack embeddings
            self._class_embeddings = torch.cat(class_embeddings_list, dim=0)
            self._pooled_embeddings = torch.cat(pooled_embeddings_list, dim=0)

            # Save to cache
            torch.save(
                {
                    "class_embeddings": self._class_embeddings,
                    "pooled_embeddings": self._pooled_embeddings,
                    "prompt_template": self.prompt_template,
                    "class_names": self.class_names,
                },
                cache_path,
            )
            logger.info(f"Embeddings cached to {cache_path}")

        finally:
            # Move encoders back
            self.pipe.text_encoder.to(original_device)
            self.pipe.text_encoder_2.to(original_device)
            self.pipe.text_encoder_3.to(original_device)
            torch.cuda.empty_cache()

    def get_embeddings_for_classes(
        self,
        class_indices: torch.Tensor,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for specific class indices.

        Args:
            class_indices: Tensor of class indices to retrieve
            device: Target device

        Returns:
            Tuple of (selected_class_embeddings, selected_pooled_embeddings)
        """
        class_emb, pooled_emb = self.get_embeddings(device="cpu")

        # Select embeddings for given indices
        selected_class = torch.index_select(class_emb, 0, class_indices.cpu())
        selected_pooled = torch.index_select(pooled_emb, 0, class_indices.cpu())

        return selected_class.to(device), selected_pooled.to(device)
