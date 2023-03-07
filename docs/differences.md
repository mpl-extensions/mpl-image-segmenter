# Differences from mpl-interactions


`mpl-image-segmenter` originally lived in the [mpl-interactions](mpl-interactions.readthedocs.io/) package.

On moving to it's own package a few changes to API were made.

## Object name

The class is now named `ImageSegmenter`
**mpl-image-segmenter**
```python
from mpl-image-segmenter import ImageSegmenter

segmenter = ImageSegmenter(img)
```

**mpl-interactions**
```python
from mpl-interactions import image_segmenter
segmenter = image_segmenter(img)
```

## Arguments

The `lineprops` argument has been dropped in favor `props` to be consistent with
Matplotlib 3.7+.
