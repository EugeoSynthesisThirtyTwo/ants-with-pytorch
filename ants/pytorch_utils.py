import torch as th
from torch import Tensor


def gaussian_blur_kernel(kernel_size: int, sigma: float, dtype=th.float32):
    # Create a 1D Gaussian kernel
    kernel = th.exp(-th.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype) ** 2 / (2 * sigma**2))
    kernel /= kernel.sum()

    # Convert the 1D kernel to a 2D kernel
    kernel = th.outer(kernel, kernel)

    # Reshape the kernel tensor to the required shape (1, 1, kernel_size, kernel_size)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    return kernel


def random_disk(n_points: int, device="cpu") -> Tensor:
    params = th.rand((2, n_points), device=device)  # radius, angle
    radius = th.sqrt(params[0])
    angle = (2 * th.pi) * params[1]
    return th.stack([radius * th.cos(angle), radius * th.sin(angle)], dim=1)
