import re

import numpy as np
import pytest
from mpl_image_segmenter import ImageSegmenter


def test_rbga_image():
    img_rgb = np.zeros([128, 128, 3])
    img_rgba = np.zeros([128, 128, 4])
    ImageSegmenter(img_rgb, classes=["a", "b", "c"], color_image=True)
    ImageSegmenter(img_rgba, classes=["a", "b", "c"], color_image=True)


@pytest.mark.parametrize(
    ["shape", "color"],
    [
        ((3, 128, 128), False),
        ((4, 128, 128, 3), True),
        ((5, 128, 128, 4), True),
    ],
)
def test_image_stacks(shape: tuple, color: bool):
    top = 25
    left = 25
    right = 100
    bottom = 100
    imgs = np.zeros(shape)
    # imgs_rgb = np.zeros([4, 128, 128, 3])
    # imgs_rgba = np.zeros([5, 128, 128, 4])
    seg = ImageSegmenter(imgs, classes=["a", "b", "c"], color_image=color)
    seg.current_class = "a"
    seg._onselect([(left, top), (left, bottom), (right, bottom), (right, top)])
    seg.image_index = 1
    assert seg.mask[0].sum() == 5550
    seg._onselect([(left, top + 40), (left, bottom), (right, bottom), (right, top)])
    assert seg.mask[1].sum() == 4105
    assert seg.mask[2].sum() == 0

    with pytest.raises(ValueError):
        seg.image_index = 5


def test_current_class():
    img = np.zeros([128, 128])
    seg = ImageSegmenter(img, classes=["a", "b", "c"])

    top = 25
    left = 25
    right = 100
    bottom = 100
    seg.current_class = "a"
    seg._onselect([(left, top), (left, bottom), (right, bottom), (right, top)])
    seg.current_class = "b"
    top = 30
    seg._onselect([(left, top), (left, bottom), (right, bottom), (right, top)])
    seg.current_class = "c"
    top = 40
    seg._onselect([(left, top), (left, bottom), (right, bottom), (right, top)])

    assert seg.mask.sum() == 15170

    with pytest.raises(
        ValueError, match=re.escape("d is not one of the classes: ['a', 'b', 'c']")
    ):
        seg.current_class = "d"
    assert seg._cur_class_idx == 3

    seg.current_class = 1
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Current class must be bewteen 1 and 3."
            " It cannot be 0 as 0 is the background."
        ),
    ):
        seg.current_class = 5
