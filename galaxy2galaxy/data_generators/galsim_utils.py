"""HST/ACS Cosmos generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import astroimage_utils

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

import numpy as np
import tensorflow as tf
import os
import sys
import galsim
import argparse
from galsim.bounds import _BoundsI

class GalsimProblem(astroimage_utils.AstroImageProblem):
  """Base class for image problems generated with GalSim.

  Subclasses need only implement the `galsim_generator` function used to draw
  postage stamps with GalSim.
  """

  # START: Subclass interface
  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.187
    p.img_len = 64

  @property
  def num_bands(self):
    """Number of bands."""
    return 1

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """
    Function to implement to generate galaxy postage stamps,
    use draw_and_encode_stamp to yield actual images from
    galsim objects
    """
    raise NotImplementedError

  def prepare_to_generate(self, data_dir, tmp_dir):
    """Prepare to generate data in parallel on different processes.

    This function is called if multiprocess_generate is True.

    Some things that might need to be done once are downloading the data
    if it is not yet downloaded, and building the vocabulary.

    Args:
      data_dir: a string
      tmp_dir: a string
    """
    pass
  # END: Subclass interface

  def example_reading_spec(self):
    """Define how data is serialized to file and read back.

    Returns:
      data_fields: A dictionary mapping data names to its feature type.
      data_items_to_decoders: A dictionary mapping data names to TF Example
         decoders, to be used when reading back TF examples from disk.
    """
    p = self.get_hparams()

    data_fields = {

        "psf_cfht/encoded": tf.FixedLenFeature((), tf.string),
        "psf_cfht/format": tf.FixedLenFeature((), tf.string),

        "ps_cfht/encoded": tf.FixedLenFeature((), tf.string),
        "ps_cfht/format": tf.FixedLenFeature((), tf.string),
        
        "image_hst/encoded" : tf.FixedLenFeature((), tf.string),
        "image_hst/format" : tf.FixedLenFeature((), tf.string),
        
    }
    
    

    # Adds additional attributes to be decoded as specified in the configuration
    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_fields['attrs/'+k] = tf.FixedLenFeature([], tf.float32, -1)

    data_items_to_decoders = {
        
        "psf_cfht": tf.contrib.slim.tfexample_decoder.Image(
                image_key="psf_cfht/encoded",
                format_key="psf_cfht/format",
                channels=self.num_bands,
                # The factor 2 here is to account for x2 interpolation
                shape=[2*p.img_len, 2*p.img_len // 2 + 1, self.num_bands],
                dtype=tf.float32),

        "ps_cfht": tf.contrib.slim.tfexample_decoder.Image(
                image_key="ps_cfht/encoded",
                format_key="ps_cfht/format",
                channels=self.num_bands,
                shape=[p.img_len, p.img_len // 2 + 1],
                dtype=tf.float32),
        
        "targets": tf.contrib.slim.tfexample_decoder.Image(
                image_key="image_hst/encoded",
                format_key="image_hst/format",
                channels=self.num_bands,
                shape=[p.img_len, p.img_len, self.num_bands],
                dtype=tf.float32),
    }

    if hasattr(p, 'attributes'):
        for k in p.attributes:
            data_items_to_decoders[k] = tf.contrib.slim.tfexample_decoder.Tensor('attrs/'+k)

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.RMSE]
    return eval_metrics

  @property
  def decode_hooks(self):
    return [image_utils.convert_predictions_to_image_summaries]

  @property
  def multiprocess_generate(self):
    """Whether to generate the data in multiple parallel processes."""
    return True

  @property
  def num_generate_tasks(self):
    """Needed if multiprocess_generate is True."""
    return self.num_train_shards + self.num_dev_shards

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def draw_and_encode_stamp(gal, psf, stamp_size, pixel_scale, attributes=None, fwhm_sampler=None):
    """
    Draws the galaxy, psf and noise power spectrum on a postage stamp and
    encodes it to be exported in a TFRecord.
    """

    # Apply the PSF
    
    psf_hst = psf
    gal_hst = galsim.Convolve(gal, psf_hst)
    
    # Generate a CFHT-like PSF
    fwhm_cfht = fwhm_sampler.get(1)[0]
    psf = galsim.Kolmogorov(fwhm=fwhm_cfht,flux=1.0)#, scale_unit=galsim.arcsec) ommit units and let be handled withinin wcs
    
    g_sigma = 0.01 # standard deviation of the shear distribution
    def get_g(n_g):
        g = np.random.normal(0., g_sigma, n_g)
        while np.linalg.norm(g) > 1:
            g = np.random.normal(0., g_sigma, n_g)
        return g

    e1, e2 = get_g(2) #+[cst1, cst2] because the mean of the PSF ellipticity can be different from zero
    psf_cfht = psf.shear(g1=e1, g2=e2)
    gal_cfht = galsim.Convolve(gal, psf_cfht)

    # Draw a kimage of the galaxy, just to figure out what mask is, there might
    # be more efficient ways to do this though...
    bounds = _BoundsI(0, stamp_size//2, -stamp_size//2, stamp_size//2-1)
    imG_cfht = gal_cfht.drawKImage(bounds=bounds,
                         scale=2.*np.pi/(stamp_size * pixel_scale),
                         recenter=False)
    mask_cfht = ~(np.fft.fftshift(imG_cfht.array, axes=0) == 0)

    # We draw the pixel image of the convolved image
    im_cfht = gal_cfht.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale,
                       method='no_pixel', use_true_center=False).array.astype('float32')
    # Draw the Fourier domain image of the galaxy, using x1 zero padding,
    # and x2 subsampling
    interp_factor=2
    padding_factor=1
    Nk = stamp_size*interp_factor*padding_factor
    bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
    imCp_cfht = psf.drawKImage(bounds=bounds,
                         scale=2.*np.pi/(Nk * pixel_scale / interp_factor),
                         recenter=False)

    # Transform the psf array into proper format, remove the phase
    im_psf_cfht = np.abs(np.fft.fftshift(imCp_cfht.array, axes=0)).astype('float32')
    
    # Generate an empty noise power spectrum
    
    ps_cfht = np.zeros((stamp_size, stamp_size//2+1), dtype='float32')
    
    
    # Draw a kimage of the galaxy, just to figure out what mask is, there might
    # be more efficient ways to do this though...
    bounds = _BoundsI(0, stamp_size//2, -stamp_size//2, stamp_size//2-1)
    imG_hst = gal_hst.drawKImage(bounds=bounds,
                         scale=2.*np.pi/(stamp_size * pixel_scale),
                         recenter=False)
    mask_hst = ~(np.fft.fftshift(imG_hst.array, axes=0) == 0)

    # We draw the pixel image of the convolved image
    im_hst = gal_hst.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale,
                       method='no_pixel', use_true_center=False).array.astype('float32')

    # Draw the Fourier domain image of the galaxy, using x1 zero padding,
    # and x2 subsampling
    bounds = _BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
    imCp_hst = psf_hst.drawKImage(bounds=bounds,
                         scale=2.*np.pi/(Nk * pixel_scale / interp_factor),
                         recenter=False)

    # Transform the psf array into proper format, remove the phase
    im_psf_hst = np.abs(np.fft.fftshift(imCp_hst.array, axes=0)).astype('float32')
    
    # Generate an empty noise power spectrum
    
    ps_hst = np.zeros((stamp_size, stamp_size//2+1), dtype='float32')

    
    
    serialized_output = {"psf_cfht/encoded": [im_psf_cfht.tostring()],
                         "psf_cfht/format": ["raw"],
                         "ps_cfht/encoded": [ps_cfht.tostring()],
                         "ps_cfht/format": ["raw"],
                         "image_hst/encoded" : [im_hst.tostring()],
                         "image_hst/format" : ["raw"],
                         }
    
    
    

    # Adding the parameters provided
    if attributes is not None:
        for k in attributes:
            serialized_output['attrs/'+k] = [attributes[k]]

    return serialized_output

def maybe_download_cosmos(target_dir, sample="25.2"):
    """
    Checks for already accessible cosmos data, downloads it somewhere otherwise
    """

    import logging
    logging_level = logging.INFO

    # Setup logging to go to sys.stdout or (if requested) to an output file
    logging.basicConfig(format="%(message)s", level=logging_level, stream=sys.stdout)
    logger = logging.getLogger('galaxy2galaxy:galsim')

    try:
        catalog = galsim.COSMOSCatalog(sample=sample)
        return
    except:
        try:
            catalog = galsim.COSMOSCatalog(sample=sample, dir=target_dir)
            return
        except:
            logger.info("Couldn't access dataset, re-downloading")

    url = "https://zenodo.org/record/3242143/files/COSMOS_%s_training_sample.tar.gz"%(sample)
    file_name = os.path.basename(url)
    target = os.path.join(target_dir, file_name)
    unpack_dir = target[:-len('.tar.gz')]
    args = argparse.Namespace()
    args.quiet = True
    args.force = False
    args.verbosity = 2
    args.save = True

    # Download the tarball
    new_download, target, meta = galsim.download_cosmos.download(url, target,
                                                                 unpack_dir,
                                                                 args, logger)
    # Usually we unpack if we downloaded the tarball
    do_unpack = new_download

    # If the unpack dir is missing, then need to unpack
    if not os.path.exists(unpack_dir):
        do_unpack = True

    # But of course if there is no tarball, we can't unpack it
    if not os.path.isfile(target):
        do_unpack = False

    # Unpack the tarball
    if do_unpack:
        galsim.download_cosmos.unpack(target, target_dir, unpack_dir, meta,
                                      args, logger)

    # Usually, we remove the tarball if we unpacked it and command line doesn't specify to save it.
    do_remove = do_unpack

    # Remove the tarball
    if do_remove:
        logger.info("Removing the tarball to save space")
        os.remove(target)
