"""
Quick, automated cutout generation for astronomical imaging. (Requires an input image (or directory) and an object
catalogue.
"""

try:
    import os
    import argparse
    import time
    import warnings
    import math
    import logging
    from pathlib import Path

    warnings.filterwarnings("ignore")

    import skimage

    import multiprocessing as mp

    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import wcs
    from astropy.nddata.utils import Cutout2D, NoOverlapError
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel

    from photutils import detect_threshold, detect_sources, deblend_sources
    from numpy import copy, floor, ndarray, median, sqrt, mean, array, asarray, max, round, isnan, sort, flip, isinf


except ImportError:
    print("Issue with importing necessary packages. Please check you have the correct packages installed.")
    exit(1)

########################################################################################################################
# ANALYSIS METHODS BLOCK ###############################################################################################
########################################################################################################################


def get_tract_and_patch(filename):
    """
    Returns the tract and patch (as strings) for a given image filename.
    :param filename: In the format      *_TRACT_PATCH.fits        with no other underscores. Will automatically
                     convert the example patch of 5c5 to 5,5 in string format.
    :return:
    """
    # todo add complexity to this argument (return just tract if required)
    clip_1 = filename.split(".")[0]
    clip_2 = clip_1.split("_")
    tract, patch = clip_2[1], clip_2[2].replace("c", ",")
    return tract, patch


def objects_in_tract(catalogue, tract: int, patch: str):
    """
    Returns the indices of all galaxies that reside within the tract.
    """

    tract_column = asarray(catalogue["TRACT"])
    patch_column = asarray(catalogue["PATCH"], dtype=str)

    tract = int(tract)

    indices = []
    for n in range(0, len(tract_column)):
        if tract_column[n] == tract and patch_column[n] == patch:
            indices.append(n)
    return array(indices)


def get_wcs(fits_filename):
    """ Finds and returns the WCS for an image. If Primary Header WCS no good, searches each index until a good one
        is found. If none found, raises a ValueError
    """
    # Try just opening the initial header
    wcs_init = wcs.WCS(fits_filename)
    ra, dec = wcs_init.axis_type_names
    if ra.upper() == "RA" and dec.upper() == "DEC":
        return wcs_init

    else:
        hdu_list = fits.open(fits_filename)
        for n in hdu_list:
            try:
                wcs_slice = wcs.WCS(n.header)
                ra, dec = wcs_slice.axis_type_names
                if ra.upper() == "RA" and dec.upper() == "DEC":
                    return wcs_slice
            except:
                continue
        hdu_list.close()

    raise ValueError


def get_arc_conv(w: wcs.WCS):
    """ Gets pixels to arc-seconds conversion scale (Number of arcseconds per pixel) """
    pix_x, pix_y = 1, 1
    ra_1, dec_1 = w.wcs_pix2world(pix_x, pix_y, 0)
    ra_2, dec_2 = w.wcs_pix2world(pix_x + 1, pix_y + 1, 0)
    diff_1 = abs(ra_1 - ra_2) * 3600
    diff_2 = abs(dec_1 - dec_2) * 3600
    return (diff_1 + diff_2) / 2


def generate_cutout(image, position, img_wcs=None, size=91, world_coords=True):
    """
    Generates a cutout for a given image. Uses world coordinates by default, but can be configured to take
    in a position corresponding to the actual array indices.
    :param image:
    :param position:
    :param wcs:
    :param size:
    :param world_coords:
    :return:
    """

    if world_coords:
        coord = img_wcs.wcs_world2pix(position[0], position[1], 0, ra_dec_order=True)
        pix_x, pix_y = coord[0], coord[1]
    else:
        pix_x, pix_y = position[0], position[1]

    # Send an index error iff the position does not lie in the image.
    if 0 < pix_x < image.shape[0] and 0 < pix_y < image.shape[1]:
        try:
            cut = Cutout2D(image, (pix_x, pix_y), size)
            return cut.data
        except NoOverlapError:
            raise IndexError
    else:
        raise IndexError


def mask_cutout(cutout, nsigma=1., gauss_width=2.0, npixels=5):
    """ Masks a cutout using segmentation and deblending using watershed"""
    mask_data = {}

    # Generate a copy of the cutout just to prevent any weirdness with numpy pointers
    cutout_copy = copy(cutout)

    sigma = gauss_width * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma)
    kernel.normalize()

    # Find threshold for cutout, and make segmentation map
    threshold = detect_threshold(cutout, snr=nsigma)
    segments = detect_sources(cutout, threshold, npixels=npixels,
                              filter_kernel=kernel)

    # Attempt to deblend. Return original segments upon failure.
    try:
        deb_segments = deblend_sources(cutout, segments, npixels=5,
                                       filter_kernel=kernel)
    except ImportError:
        print("Skimage not working!")
        deb_segments = segments
    except:
        # Don't do anything if it doesn't work
        deb_segments = segments

    segment_array = deb_segments.data

    # Center pixel values. (Assume that the central segment is the image, which is should be)
    c_x, c_y = floor(segment_array.shape[0] / 2), floor(segment_array.shape[1] / 2)
    central = segment_array[int(c_x)][int(c_y)]

    # Estimate background with severe cutout
    bg_method = 1
    bg_est, bg_rms = estimate_background(cutout_copy)

    mask_data["BG_EST"] = bg_est
    mask_data["BG_RMS"] = bg_rms
    mask_data["N_OBJS"] = segments.nlabels
    mask_data["BGMETHOD"] = bg_method

    # Use alternative method to try and get a estimate or rms value if first method fails
    if isnan(bg_est) or isnan(bg_rms):
        bg_pixel_array = []
        for x in range(0, segment_array.shape[0]):
            for y in range(0, segment_array.shape[1]):
                if segment_array[x][y] == 0:
                    bg_pixel_array.append(cutout_copy[x][y])

        bg_est = median(bg_pixel_array)
        bg_rms = sqrt((mean(bg_pixel_array) - bg_est) ** 2)

        bg_method = 2

    # Return input image if no need to mask
    if segments.nlabels == 1:
        mask_data["N_MASKED"] = 0
        return cutout_copy, mask_data

    num_masked = 0
    # Mask pixels
    for x in range(0, segment_array.shape[0]):
        for y in range(0, segment_array.shape[1]):
            if segment_array[x][y] not in (0, central):
                cutout_copy[x][y] = bg_est
                num_masked += 1

    mask_data["N_MASKED"] = num_masked

    return cutout_copy, mask_data


def estimate_background(cutout, nsigma=1, gauss_width=1, npixels=3):
    """ Simple background detecting using super-pixel method"""
    sigma = gauss_width * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma)
    kernel.normalize()

    # Find threshold for cutout, and make segmentation map
    threshold = detect_threshold(cutout, snr=nsigma)
    segments = detect_sources(cutout, threshold, npixels=npixels,
                              filter_kernel=kernel)

    segment_array = segments.data

    x_step = int(cutout.shape[0] / 10)
    y_step = int(cutout.shape[1] / 10)

    super_pixel_medians, super_pixel_rms_vals = [], []

    for x in range(0, cutout.shape[0] - x_step, x_step):
        for y in range(0, cutout.shape[1] - y_step, y_step):
            super_pixel = cutout[y: y + y_step, x: x + x_step]

            super_pixel_contents = []
            for m in range(0, super_pixel.shape[0]):
                for n in range(0, super_pixel.shape[1]):
                    if segment_array[m][n] == 0:
                        super_pixel_contents.append(super_pixel[m][n])

            super_pixel_medians.append(median(super_pixel_contents))
            super_pixel_rms_vals.append(sqrt((mean(super_pixel_contents) - median(super_pixel_contents)) ** 2))

    return median(super_pixel_medians), median(super_pixel_rms_vals)

########################################################################################################################
# IMAGE PROCESSING METHODS BLOCK #######################################################################################
########################################################################################################################

global_verbose = False
global_batch = False
global_multithread = False
global_mask = False
cutout_size = 61
config_filename = ""
output_subdirectory = ""

check_snr, snr_threshold = False, 0.
check_isloated, masked_threshold = False, 0.

global_nsigma, global_gauss_width, global_npixels = 1., 2.0, 5


ra_key, dec_key = "RA", "DEC"


# Get command line arguments
def process_arguments():
    """ Gather all user arguments, and set global variables if necessary. """
    global global_verbose
    global global_batch
    global global_multithread
    global global_mask
    global has_tract_info
    global output_subdirectory
    global ra_key
    global dec_key
    global cutout_size
    global check_snr, snr_threshold
    global check_isloated, masked_threshold
    global global_nsigma, global_gauss_width, global_npixels

    parser = argparse.ArgumentParser(description="Automated cutout creation for astronomical imaging. (Version 2.1)",
                                     epilog="Please email hsouchereau@outlook.com for help inquiries or bug reports.")

    parser.add_argument("image_filename", type=str,
                        help="Filename of the image to be processed.")

    parser.add_argument("cat_filename", type=str, help="Filename for the catalogue.")

    parser.add_argument("-v", "--verbose",
                        help="Run in Verbose mode", action="store_true")

    parser.add_argument("-b", "--batch",
                        help="Run in batch mode. Will assume image_filename is a path containing many " +
                        "images.",
                        action="store_true")

    parser.add_argument("--multi",
                        help="Run in multithread mode.",
                        action="store_true")

    parser.add_argument("-m", "--mask",
                        help="Apply mask to non-central objects. Will also estimate background.",
                        action="store_true")

    parser.add_argument("-s", "--cutout_size", type=int,
                        help="Defines the size of the cutout")

    parser.add_argument("-O", "--output_subdir", type=str,
                        help="Make a custom subdirectory to save output in.")

    parser.add_argument("--ra", type=str,
                        help="Right Ascension Column Name in Table")

    parser.add_argument("--dec", type=str,
                        help="Declination Column Name in Table")

    parser.add_argument("--snr", type=float,
                        help="Desired signal to noise ratio to save cutout.")

    parser.add_argument("-i", "--isolated", type=float,
                        help="Percentage of masked pixels to allow (For isolated objects).")

    parser.add_argument("--maskparams", type=str,
                        help="Mask parameters, in the form (nsigma, gauss_width, npixels).\n" +
                        "Example Usage: --maskparams \"(2.0, 1.0, 7)\"")

    # Get all input arguments
    args = parser.parse_args()

    # Main Introduction Message
    if args.verbose:
        print("\n\tCutout Creator -- Written by Harrison Souchereau -- hsouchereau@outlook.com \n")
        global_verbose = True

    # Check tract, batch, and multi-threading flags
    if args.batch:
        global_batch = True
    if args.multi:
        global_multithread = True
    if args.output_subdir is not None:
        # Add "/" to output subdirectory if it doesn't exist.
        output_subdirectory = args.output_subdir
        output_subdirectory += "/" if output_subdirectory[len(output_subdirectory) - 1] is not "/" else ""
    if args.mask is not None:
        global_mask = args.mask
    if args.ra is not None:
        ra_key = args.ra
    if args.dec is not None:
        dec_key = args.dec
    if args.cutout_size is not None:
        cutout_size = args.cutout_size
    if args.snr is not None:
        check_snr = True
        snr_threshold = args.snr
    if args.isolated is not None:
        check_isloated = True
        masked_threshold = args.isolated

    if args.maskparams is not None:
        splits = str(args.maskparams).split("(")[1].split(")")[0].split(",")

        global_nsigma = float(splits[0])
        global_gauss_width = float(splits[1])
        global_npixels = int(splits[2])

    return args


def process(img_filename, cat_filename):
    """ Full pipeline processing. (Basically a wrapper to format before process_image()) """
    # Open catalog and images
    try:
        if global_verbose:
            print("Reading Catalog:\t", cat_filename)
        cat = Table.read(cat_filename, format="fits")
        if ra_key not in cat.colnames or dec_key not in cat.colnames:
            print("Column names not found. Exiting")
            exit(1)
    except (ValueError, FileNotFoundError):
        print("Error while opening catalog. Cannot find file from input, or the filename is an empty directory.")
        return
    except UnicodeDecodeError:
        print("Error when opening catalog. Check to ensure table is in FITS table format.")
        return

    # Gather all images to process (make an array of 1 object if not in batch mode)
    if global_batch:
        images = []
        images_raw = Path(img_filename).rglob('*.fits')
        for image in images_raw:
            images.append(str(image))
    else:
        images = [img_filename]

    if global_multithread:
        print("Multiprocessing Selected:", mp.cpu_count(), "threads available.")
        pool = mp.Pool(processes=len(images))
        results = [pool.apply_async(process_image, (cat, image)) for image in images]
        [res.get() for res in results]
    else:
        for image in images:
            process_image(cat, image)

    return None


def process_image(catalogue, img_filename):
    """ Run fitting routine on an image. """

    # Get the filename of the image without any path information
    img_filename_no_path = img_filename.split("/")[len(img_filename.split("/")) - 1]
    print("Processing:\t", img_filename_no_path, "\t", len(catalogue), "potential objects.")
    # Check for tract info, and gather objects that will be considered for fitting ###################################
    valid_indices = range(0, len(catalogue))

    # Load image #####################################################################################################
    img, w = None, None
    try:
        hdu_list = fits.open(img_filename)
        # Iterate through HDU_List until first image format is found
        for n in range(0, len(hdu_list)):
            img = hdu_list[n].data
            if type(img) == ndarray:
                break
        try:
            w = get_wcs(img_filename)
        except ValueError:
            logging.critical("No WCS Found for " + img_filename)
            return
        hdu_list.close()
    except:
        logging.critical("Image not found. Check path to images and filename.")
        return

    # Prepare wrapper FITS files if saving all output (isotables and cutouts) to one file

    cutout_fits = fits.HDUList()

    # Process Objects ################################################################################################
    for index in valid_indices:
        extra_params = {}   # Container dict for all local paramters (to add to output header)
        row = catalogue[index]

        extra_params["IMAGE"] = img_filename_no_path
        extra_params["CATALOG"] = catalogue_filename

        # Make cutout
        try:
            cutout = generate_cutout(img, (row[ra_key], row[dec_key]),
                                     size=cutout_size, img_wcs=w)
            if cutout.shape[0] != cutout.shape[1]:
                continue
        except (IndexError, ValueError):
            # Move on to the next objcet if outside of the image boundaries
            continue

        # Mask cutout
        if global_mask:
            try:
                masked_cutout, mask_data = mask_cutout(cutout,
                                                       nsigma=global_nsigma,
                                                       gauss_width=global_gauss_width,
                                                       npixels=global_npixels)

                cutout = masked_cutout
                extra_params.update(mask_data)

                bg_est, bg_rms = mask_data["BG_EST"], mask_data["BG_RMS"]
                if isnan(bg_est) or isnan(bg_rms):
                    continue
                # Do an isolation check if requested
                if check_isloated:

                    if extra_params["N_MASKED"] == 0:
                        continue

                    masked_percent = extra_params["N_MASKED"] / (cutout.shape[0] * cutout.shape[1])

                    extra_params["P_MASKED"] = masked_percent
                    if masked_percent > masked_threshold:
                        continue

            except ValueError:
                if global_verbose:
                    print("Error processing galaxy", index, "due to masking failure.")
                continue
        else:
            # If not masking, simply run the background estimation. Set to bad flag if nan check
            bg_est, bg_rms = estimate_background(cutout)
            if isnan(bg_est):
                bg_est = -999
            if isnan(bg_rms):
                bg_rms = 999

            extra_params["BG_EST"] = bg_est
            extra_params["BG_RMS"] = bg_rms

        snr = flip(sort(cutout, axis=None))[5] / bg_rms
        if isinf(snr):
            continue

        extra_params["SNR"] = snr

        if check_snr:
            if snr < snr_threshold:
                continue

        # print(round(snr, 3), round(masked_percent, 3))

        # Add information to file
        hdr = fits.Header()
        for column in row.colnames:
            if str(row[column]) not in ("nan", "inf"):
                hdr[column] = row[column]
        for key in extra_params:
            hdr[key] = extra_params[key]

        for hdr_key in hdr:
            if str(hdr[hdr_key]) in ("nan", "inf"):
                hdr[hdr_key] = -999
        cutout_fits.append(fits.ImageHDU(cutout, header=hdr))

    print(len(cutout_fits), "objects to be saved.")
    # Save cutouts and isotables #######################################################################################

    outfile_prefix = str(default_output_dir) + output_subdirectory + img_filename_no_path.split(".")[0]

    n = 1
    if not os.path.isfile(outfile_prefix + "_cutouts.fits"):
        cutout_fits.writeto(outfile_prefix + "_cutouts.fits")
    else:
        while os.path.isfile(outfile_prefix + "(" + str(n) + ")" + "_cutouts.fits"):
            n += 1
        cutout_fits.writeto(outfile_prefix + "(" + str(n) + ")" + "_cutouts.fits")

    if global_verbose:
        print("Finished fitting", img_filename_no_path, ":", len(valid_indices), "cutout extractions.")
    return None


########################################################################################################################
# ACTIVE CODE BLOCK ####################################################################################################
########################################################################################################################

start_time = time.time()

# Gather all command line arguments.
cl_args = process_arguments()


image_filename, catalogue_filename = cl_args.image_filename, cl_args.cat_filename


# Make output directory if it isn't made
default_output_dir = "output/"
try:
    if not os.path.isdir(default_output_dir):
        os.mkdir(default_output_dir)
    if not os.path.isdir(default_output_dir + output_subdirectory):
        os.mkdir(default_output_dir + output_subdirectory)
except FileExistsError:
    pass

# Process image(s)
warnings.filterwarnings("ignore")
process(image_filename, catalogue_filename)

# Print optional output message
time_seconds = time.time() - start_time
time_minutes = time_seconds / 60
print("Time Elapsed:", str(time_seconds)[:6], "seconds (", str(time_minutes)[:6], " minutes).")
