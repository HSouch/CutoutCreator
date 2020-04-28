# CutoutCreator
A simple, effective tool for generating cutouts of objects in astronomical imaging. 

## Quick Start

For a single image:

```python CutoutCreator.py path_to_image.fits object_catalog.fits```

For a directory of images:

``` python CutoutCreator.py path_to_directory/ object_catalog.fits -b```

For a directory of images using multiple threads:

``` python CutoutCreator.py path_to_directory/ object_catalog.fits -b --multi```

## Optional flags

```-s --cutout_size INT``` Define cutout size (integer). Default cutout size (width and height) is 61

```-b``` Run in batch mode. (The input is treated as a directory instead of a filename)

```--multi``` Run using multithreading. Will only run if batch mode is selected.

```-m --mask``` Mask object cutouts. This will try to remove the light from companion objects from the cutout.

```-O --output_subdir STR``` Generate (if needed) a defined subdirectory to save cutouts. 

```--ra --dec STR``` Define catalog column names for right ascension and declination. Defaults are RA and DEC.

```--snr FLOAT``` Define a required signal to noise ratio cutoff (float) that objects must be above in order to save the cutout. This is most effective when masking cutouts.

```-i --isolated FLOAT``` Select a percentage threshold of allowed masked pixels (between 0 and 1) in order to save images. This is useful if one is looking for isolated objects. 

```--maskparams STR "(FLOAT, FLOAT, INT)"``` Define the parameters for masking. These are the values ```nsigma, gauss_width, npixels``` which define the snr required to be a detected object, the width of the gaussian convolution kernel used in masking, and the minimum number of pixels required for a cluster of pixels to be a detection. These can be adjusted to increase or decrease the severity of the masking procedure.
