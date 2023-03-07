from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import __version__ as mpl_version
from matplotlib import get_backend
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS, to_rgba_array
from matplotlib.path import Path
from matplotlib.pyplot import ioff, subplots
from matplotlib.widgets import LassoSelector
from mpl_interactions import zoom_factory
from mpl_interactions.generic import panhandler

if TYPE_CHECKING:
    from typing import Any


class image_segmenter:
    """Manually segment an image with the lasso selector."""

    def __init__(  # type: ignore
        self,
        img,
        nclasses=1,
        mask=None,
        mask_colors=None,
        mask_alpha=0.75,
        lineprops=None,
        props=None,
        lasso_mousebutton="left",
        pan_mousebutton="middle",
        ax=None,
        figsize=(10, 10),
        **kwargs,
    ):
        """
        Manually segmenting an image.

        Parameters
        ----------
        img : array_like
            A valid argument to imshow
        nclasses : int, default 1
            How many classes to have in the mask.
        mask : arraylike, optional
            If you want to pre-seed the mask
        mask_colors : None, color, or array of colors, optional
            the colors to use for each class. Unselected regions will always be
            totally transparent
        mask_alpha : float, default .75
            The alpha values to use for selected regions. This will always override
            the alpha values in mask_colors if any were passed
        lineprops : dict, default: None
            DEPRECATED - use props instead.
            lineprops passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
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

        if mask_colors is None:
            # this will break if there are more than 10 classes
            if nclasses <= 10:
                self.mask_colors = to_rgba_array(list(TABLEAU_COLORS)[:nclasses])
            else:
                # up to 949 classes. Hopefully that is always enough....
                self.mask_colors = to_rgba_array(list(XKCD_COLORS)[:nclasses])
        else:
            self.mask_colors = to_rgba_array(np.atleast_1d(mask_colors))
            # should probably check the shape here
        self.mask_colors[:, -1] = self.mask_alpha

        self._img = np.asarray(img)

        if mask is None:
            self.mask = np.zeros(self._img.shape[:2])
            """See :doc:`/examples/image-segmentation`."""
        else:
            self.mask = mask

        self._overlay = np.zeros((*self._img.shape[:2], 4))
        self.nclasses = nclasses
        for i in range(nclasses + 1):
            idx = self.mask == i
            if i == 0:
                self._overlay[idx] = [0, 0, 0, 0]
            else:
                self._overlay[idx] = self.mask_colors[i - 1]
        if ax is not None:
            self.ax = ax
            self.fig = self.ax.figure
        else:
            with ioff():
                self.fig, self.ax = subplots(figsize=figsize)
        self.displayed = self.ax.imshow(self._img, **kwargs)
        self._mask = self.ax.imshow(self._overlay)

        default_props = {"color": "black", "linewidth": 1, "alpha": 0.8}
        if (props is None) and (lineprops is None):
            props = default_props
        elif (lineprops is not None) and (mpl_version >= "3.7"):
            print("*lineprops* is deprecated - please use props")
            props = {"color": "black", "linewidth": 1, "alpha": 0.8}

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

        pix_x = np.arange(self._img.shape[0])
        pix_y = np.arange(self._img.shape[1])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self.ph = panhandler(self.fig, button=pan_mousebutton)
        self.disconnect_zoom = zoom_factory(self.ax)
        self.current_class = 1
        self.erasing = False

    def _onselect(self, verts: Any) -> None:
        self.verts = verts
        p = Path(verts)
        self.indices = p.contains_points(self.pix, radius=0).reshape(self.mask.shape)
        if self.erasing:
            self.mask[self.indices] = 0
            self._overlay[self.indices] = [0, 0, 0, 0]
        else:
            self.mask[self.indices] = self.current_class
            self._overlay[self.indices] = self.mask_colors[self.current_class - 1]

        self._mask.set_data(self._overlay)
        self.fig.canvas.draw_idle()

    def _ipython_display_(self) -> None:
        display(self.fig.canvas)  # type: ignore # noqa: F821
