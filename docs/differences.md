# Differences from mpl-interactions


`mpl-image-segmenter` originally lived in the [mpl-interactions](mpl-interactions.readthedocs.io/) package.

On moving to it's own package a few changes to API were made.

## Object name

The class is now named `ImageSegmenter`
**mpl-image-segmenter**
```python
from mpl_image_segmenter import ImageSegmenter

segmenter = ImageSegmenter(img)
```

**mpl-interactions**
```python
from mpl_interactions import image_segmenter
segmenter = image_segmenter(img)
```

## Arguments

The `lineprops` argument has been dropped in favor `props` to be consistent with
Matplotlib 3.7+.

The `color_image` argument was introduced in order to allow for the new behavior of allowing a stack of images.

## Properties


`segmenter.ph` has been replaced by `segmenter.panmanager`


`segmenter.verts` which was never fully functional has been removed and replaced by `segmenter.get_paths()`

## Stacking
You can now pass a stack of images, and control which is currently displayed using the `image_index` property.
