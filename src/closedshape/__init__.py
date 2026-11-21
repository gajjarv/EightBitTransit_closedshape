"""closedshape: closed-shape shadow imaging of transit light curves (EightBitTransit-style)."""
from .limbdark import make_intensity
from .forward import Orbit, blocked_flux, grid_coords, pixel_design, supersample_times, rebin
from .pixel import clsq, closed_pixel, largest_blob, gradient_matrix
from .shapes import ShapeModel, fit_sphere, fit_ellipse, fit_fourier, f_to_e, e_to_f
from .utils import pixelize_ellipse, image_metrics, load_lightcurve
from .image import TransitingImage, invertLC

__version__ = "0.1.0"
