"""Example images for docs."""

from pathlib import Path

import numpy as np

__all__ = [
    "gray_image_stack",
    "color_image_stack",
    "example_mask_stack",
]


def gray_image_stack() -> np.ndarray:
    """
    Get the example image stack in grayscale.

    Returns
    -------
    img : np.ndarray
        With shape (5, 512, 512, 3)

    See Also
    --------
    color_image_stack

    Notes
    -----
    See color_image_stack for documentation on how the images were generated.
    """
    return color_image_stack().mean(axis=-1) % 255


def color_image_stack() -> np.ndarray:
    """
    Get the example image as a color image.

    Returns
    -------
    img : np.ndarray
        With shape (5, 512, 512, 3)

    See Also
    --------
    color_image_stack

    Notes
    -----
    Image was generated using skimage version 0.20.0 with
    the following code snippet:

    ```python
    import numpy as np
    from skimage.draw import random_shapes
    shapes = np.zeros([5,512,512,3], dtype="uint8")
    for i in range(5):
        shapes[i] = random_shapes((512, 512), 25, random_seed=20230321+i)[0]
    np.savez_compressed("example_img_stack.npz", shapes)
    ```
    """
    return np.load(Path(__file__).parent / "example_img_stack.npz")["arr_0"]


def example_mask_stack() -> np.ndarray:
    """
    Get the premade example mask stack.

    Returns
    -------
    img : np.ndarray
        With shape (5, 512, 512)
    """
    return np.load(Path(__file__).parent / "mask.npz")["arr_0"]
