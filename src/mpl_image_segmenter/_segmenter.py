from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import __version__ as mpl_version
from matplotlib import get_backend
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS, to_rgba_array
from matplotlib.path import Path
from matplotlib.pyplot import ioff, subplots
from matplotlib.widgets import LassoSelector
from mpl_pan_zoom import PanManager, zoom_factory

if TYPE_CHECKING:
    from typing import Any


class ImageSegmenter:
    """Manually segment an image with the lasso selector."""

    def __init__(  # type: ignore
        self,
        imgs,
        classes=1,
        color_image=False,
        mask=None,
        mask_colors=None,
        mask_alpha=0.75,
        props=None,
        lasso_mousebutton="left",
        pan_mousebutton="middle",
        ax=None,
        figsize=(10, 10),
        **kwargs,
    ):
        """
        Manually segment an image.

        Parameters
        ----------
        imgs : array_like
            A single image, or a stack of images shape (N, Y, X)
        classes : int, iterable[string], default 1
            If a number How many classes to have in the mask.
        color_image : bool, default False
            If True treat the final dimension of `imgs` as the RGB(A) axis.
            Allows for shapes like ([N], Y, X, [3,4])
        mask : arraylike, optional
            If you want to pre-seed the mask
        mask_colors : None, color, or array of colors, optional
            the colors to use for each class. Unselected regions will always be
            totally transparent
        mask_alpha : float, default .75
            The alpha values to use for selected regions. This will always override
            the alpha values in mask_colors if any were passed
        props : dict, default: None
            props passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        lasso_mousebutton : str, or int, default: "left"
            The mouse button to use for drawing the selecting lasso.
        pan_mousebutton : str, or int, default: "middle"
            The button to use for `~mpl_interactions.generic.panhandler`. One of
            'left', 'middle', 'right', or 1, 2, 3 respectively.
        ax : `matplotlib.axes.Axes`, optional
            The axis on which to plot. If *None* a new figure will be created.
        figsize : (float, float), optional
            passed to plt.figure. Ignored if *ax* is given.
        **kwargs
            All other kwargs will passed to the imshow command for the image
        """
        # ensure mask colors is iterable and the same length as the number of classes
        # choose colors from default color cycle?

        self.mask_alpha = mask_alpha

        if isinstance(classes, Integral):
            self._classes: list[str | int] = list(range(classes))
        else:
            self._classes = classes
        self._n_classes = len(self._classes)
        if mask_colors is None:
            if self._n_classes <= 10:
                # There are only 10 tableau colors
                self.mask_colors = to_rgba_array(
                    list(TABLEAU_COLORS)[: self._n_classes]
                )
            else:
                # up to 949 classes. Hopefully that is always enough....
                self.mask_colors = to_rgba_array(list(XKCD_COLORS)[: self._n_classes])
        else:
            self.mask_colors = to_rgba_array(np.atleast_1d(mask_colors))
            # should probably check the shape here
        self.mask_colors[:, -1] = self.mask_alpha

        imgs = np.asanyarray(imgs)
        self._imgs = self._pad_to_stack(imgs, "imgs", color_image)
        self._color_image = color_image

        self._image_index = 0

        self._overlay = np.zeros((*self._imgs.shape[1:3], 4))
        if mask is None:
            self.mask = np.zeros(self._imgs.shape[:3])
        else:
            self.mask = self._pad_to_stack(mask, "mask", False)
            self._refresh_overlay_values()

        if ax is not None:
            self.ax = ax
            self.fig = self.ax.figure
        else:
            with ioff():
                self.fig, self.ax = subplots(figsize=figsize)
        self._displayed = self.ax.imshow(self._imgs[self._image_index], **kwargs)
        self._mask_im = self.ax.imshow(self._overlay)

        default_props = {"color": "black", "linewidth": 1, "alpha": 0.8}
        if props is None:
            props = default_props

        useblit = False if "ipympl" in get_backend().lower() else True
        button_dict = {"left": 1, "middle": 2, "right": 3}
        if isinstance(pan_mousebutton, str):
            pan_mousebutton = button_dict[pan_mousebutton.lower()]
        if isinstance(lasso_mousebutton, str):
            lasso_mousebutton = button_dict[lasso_mousebutton.lower()]

        if mpl_version < "3.7":
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                lineprops=props,
                useblit=useblit,
                button=lasso_mousebutton,
            )
        else:
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                props=props,
                useblit=useblit,
                button=lasso_mousebutton,
            )
        self.lasso.set_visible(True)

        # offset shape by 1 because we always pad into
        # being a stack (N, Y, X, [3,4])
        pix_x = np.arange(self._imgs.shape[1])
        pix_y = np.arange(self._imgs.shape[2])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self._pm = PanManager(self.fig, button=pan_mousebutton)
        self.disconnect_zoom = zoom_factory(self.ax)
        self.current_class = 1
        self._erasing = False
        self._paths: dict[str, list[Path]] = {"adding": [], "erasing": []}

    def _refresh_overlay_values(self) -> None:
        # leave the actual updating of image to other code
        # in order to easily manage what gets updated and when
        # the drawing happens
        for i in range(self._n_classes + 1):
            idx = self._mask[self._image_index] == i
            if i == 0:
                self._overlay[idx] = [0, 0, 0, 0]
            else:
                self._overlay[idx] = self.mask_colors[i - 1]

    @staticmethod
    def _pad_to_stack(arr: np.ndarray, name: str, color_image: bool) -> np.ndarray:
        if color_image and arr.ndim < 3:
            raise ValueError(
                f"{name} must be at least 3 dimensional when *color_image* is True"
                f" but it is {arr.ndim}D"
            )
        if arr.ndim == (2 + color_image):
            # make shape (1, M, N)
            #  or (1, M, N, [3, 4])
            return arr[None, ...]
        elif arr.ndim == (3 + color_image):
            return arr
        else:
            raise ValueError(
                f"{name} must be either 2 or 3 dimensional. Did"
                " you mean to set *color_image* to True?"
            )

    @property
    def mask(self) -> np.ndarray:
        if self._mask.shape[0] == 1:
            # don't complicate things in the simple case of
            # one image
            return self._mask[0]
        else:
            return self._mask

    @mask.setter
    def mask(self, val: np.ndarray) -> None:
        val = self._pad_to_stack(np.asanyarray(val), "mask", False)
        if self._color_image:
            compare_shape = self._imgs.shape[:-1]
        else:
            compare_shape = self._imgs.shape
        if val.shape != compare_shape:
            raise ValueError("Mask must have the same shape as imgs")
        self._mask = val

    @property
    def image_index(self) -> int:
        return self._image_index

    @image_index.setter
    def image_index(self, val: int) -> None:
        if not isinstance(val, Integral):
            raise ValueError("image_index must be an integer")
        if val >= self._imgs.shape[0]:
            raise ValueError(
                f"Too large - This segmenter only has {self._imgs.shape[0]} images."
            )
        self._image_index = val
        self._refresh_overlay_values()
        self._displayed.set_data(self._imgs[val])
        self._mask_im.set_data(self._overlay)
        self.fig.canvas.draw_idle()

    @property
    def panmanager(self) -> PanManager:
        return self._pm

    @property
    def erasing(self) -> bool:
        return self._erasing

    @erasing.setter
    def erasing(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise TypeError(f"Erasing must be a bool - got type {type(val)}")
        self._erasing = val

    @property
    def current_class(self) -> int | str:
        return self._classes[self._cur_class_idx - 1]

    @current_class.setter
    def current_class(self, val: int | str) -> None:
        if isinstance(val, str):
            if val not in self._classes:
                raise ValueError(f"{val} is not one of the classes: {self._classes}")
            # offset by one for the background
            self._cur_class_idx = self._classes.index(val) + 1
        elif isinstance(val, Integral):
            if 0 < val < self._n_classes + 1:
                self._cur_class_idx = val
            else:
                raise ValueError(
                    f"Current class must be bewteen 1 and {self._n_classes}."
                    " It cannot be 0 as 0 is the background."
                )

    def get_paths(self) -> dict[str, list[Path]]:
        """
        Get a dictionary of all the paths used to create the mask.

        Returns
        -------
        dict :
            With keys *adding* and *erasing* each containing a list of paths.
        """
        return self._paths

    def _onselect(self, verts: Any) -> None:
        p = Path(verts)
        indices = p.contains_points(self.pix, radius=0).reshape(self._mask.shape[1:3])
        if self._erasing:
            self._mask[self._image_index][indices] = 0
            self._overlay[indices] = [0, 0, 0, 0]
            self._paths["erasing"].append(p)
        else:
            self._mask[self._image_index][indices] = self._cur_class_idx
            self._overlay[indices] = self.mask_colors[self._cur_class_idx - 1]
            self._paths["adding"].append(p)

        self._mask_im.set_data(self._overlay)
        self.fig.canvas.draw_idle()

    def _ipython_display_(self) -> None:
        display(self.fig.canvas)  # type: ignore # noqa: F821
