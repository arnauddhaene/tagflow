import torch
from torchvision import transforms


def _preprocess_image(mu: float = 0.456, sigma: float = 0.224) -> transforms.Compose:
    """Preprocess image

    Args:
        mu (float, optional): average for normalization layer. Defaults to 0.456.
        sigma (float, optional): standard deviation for normalization layer. Defaults to 0.224.

    Returns:
        transforms.Compose: transformation callback function
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma),
        transforms.Resize((256, 256))
    ])


def _postprocess_mask(initial_size: torch.Size) -> transforms.Compose:
    """Preprocess image

    Args:
        mu (float, optional): average for normalization layer. Defaults to 0.456.
        sigma (float, optional): standard deviation for normalization layer. Defaults to 0.224.

    Returns:
        transforms.Compose: transformation callback function
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(initial_size)
    ])
