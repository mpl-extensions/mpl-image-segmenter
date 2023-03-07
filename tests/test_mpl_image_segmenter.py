import re

import numpy as np
import pytest
from mpl_image_segmenter import ImageSegmenter


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
