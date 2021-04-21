"""HST/ACS Cosmos generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import galsim_utils
from . import astroimage_utils
from . import seeing_distribution_class

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

from galaxy2galaxy.utils import registry

from astropy.io import fits
import numpy as np
import tensorflow as tf
import galsim

from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from numpy.fft import rfft,fft,fftshift,rfft2,irfft2
from scipy import random
from . import MakePSF

import astropy.units as u
from astropy.time import Time, TimeDelta

from astropy.coordinates import (
    EarthLocation,
    Angle,
    AltAz,
    ICRS,
    Longitude,
    FK5,
    SkyCoord
)
from astropy.constants import c as lspeed




# Path to data files required for cosmos
_COSMOS_DATA_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

class seeing_distribution(object):
    """ Seeing distribution
    Provide a seeing following CFIS distribution. Seeing generated from
    scipy.stats.rv_histogram(np.histogram(obs_seeing)). Object already
    initialized and saved into a numpy file.
    Parameters
    ----------
    path_to_file: str
        Path to the numpy file containing the scipy object.
    seed: int
        Seed for the random generation. If None rely on default one.
    """
    def __init__(self, path_to_file, seed=None):
        self._file_path = path_to_file
        self._load_distribution()
        self._random_seed = None
        if seed != None:
            self._random_seed = np.random.RandomState(seed)
    def _load_distribution(self):
        """ Load distribution
        Load the distribution from numpy file.
        """
        self._distrib = np.load(self._file_path, allow_pickle=True).item()
    def get(self, size=None):
        """ Get
        Return a seeing value from the distribution.
        Parameters
        ----------
        size: int
            Number of seeing value required.
        Returns
        -------
        seeing: float (numpy.ndarray)
            Return the seeing value or a numpy.ndarray if size != None.
        """
        return self._distrib.rvs(size=size, random_state=self._random_seed)

@registry.register_problem
class Img2imgCosmos(galsim_utils.GalsimProblem):
  """
  Img2img problem on GalSim's COSMOS 25.2 sample, at native pixel resolution,
  on 64px postage stamps.
  """

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
       We are going to split the GalSim data into shards of 1000 galaxies each,
       with 80 shards for training, 2 shards for validation.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 80,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 2,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.img_len = 64
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf", None)
    p.add_hparam("rotation", False)

  @property
  def num_bands(self):
    """Number of bands."""
    return 1

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog()
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'/COSMOS_25.2_training_sample')

    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog.getNObjects()))

    # Extracts additional information about the galaxies
    cat_param = catalog.param_cat[catalog.orig_index]
    from numpy.lib.recfunctions import append_fields
    import numpy as np

    bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']
    # Parameters for a 2 component fit
    cat_param = append_fields(cat_param, 'bulge_q', bparams[:,11])
    cat_param = append_fields(cat_param, 'bulge_beta', bparams[:,15])
    cat_param = append_fields(cat_param, 'disk_q', bparams[:,3])
    cat_param = append_fields(cat_param, 'disk_beta', bparams[:,7])
    cat_param = append_fields(cat_param, 'bulge_hlr', cat_param['hlr'][:,1])
    cat_param = append_fields(cat_param, 'bulge_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,1]), np.zeros(len(cat_param) )))
    cat_param = append_fields(cat_param, 'disk_hlr', cat_param['hlr'][:,2])
    cat_param = append_fields(cat_param, 'disk_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,2]), np.log10(cat_param['flux'][:,0])))

    # Parameters for a single component fit
    cat_param = append_fields(cat_param, 'sersic_flux_log10', np.log10(sparams[:,0]))
    cat_param = append_fields(cat_param, 'sersic_q', sparams[:,3])
    cat_param = append_fields(cat_param, 'sersic_hlr', sparams[:,1])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:,2])
    cat_param = append_fields(cat_param, 'sersic_beta', sparams[:,7])

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale*2)

      # We apply the orginal psf if a different PSF is not requested
      if hasattr(p, "psf"):
        psf = p.psf
      else:
        psf = gal.original_psf

      # Apply random rotation if requested
      if hasattr(p, "rotation") and p.rotation:
        rotation_angle = galsim.Angle(-np.random.rand()* 2 * np.pi,
                                      galsim.radians)
        gal = gal.rotate(rotation_angle)
        psf = psf.rotate(rotation_angle)

      # We save the corresponding attributes for this galaxy
      if hasattr(p, 'attributes'):
        params = cat_param[ind]
        attributes = {k: params[k] for k in p.attributes}
      else:
        attributes = None

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale,
                                               attributes=attributes)

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image = example["inputs"]

    # Clip to 1 the values of the image
    # image = tf.clip_by_value(image, -1, 1)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image
    example["targets"] = image
    return example


@registry.register_problem
class Img2imgCosmosHSC(Img2imgCosmos):
  """ COSMOS dataset at HSC resolution and uniform PSF
  """

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.168
    p.img_len = 64
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.psf = galsim.InterpolatedKImage(galsim.ImageCD(fits.getdata(os.path.join(_COSMOS_DATA_DIR, 'hsc_hann_window.fits'))+0j, scale=0.2921868167401221))
    p.rotation = True

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image = example["inputs"]

    # Apply random augmentation
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image
    example["targets"] = image
    return example


@registry.register_problem
class Attrs2imgCosmos(Img2imgCosmos):
  """ Conditional image generation problem based on COSMOS sample.
  """

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.img_len = 64
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag_auto', 'flux_radius', 'zphot', 'bulge_q', 'bulge_beta' ,
                    'disk_q', 'disk_beta', 'bulge_hlr', 'disk_hlr']

@registry.register_problem
class Img2imgCosmos32(Img2imgCosmos):
  """ Smaller version of the Img2imgCosmos problem, at half the pixel
  resolution
  """

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.06
    p.img_len = 32
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}

@registry.register_problem
class Img2imgCosmos128(Img2imgCosmos):
  """ Smaller version of the Img2imgCosmos problem, at half the pixel
  resolution
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.img_len = 128
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}

@registry.register_problem
class Attrs2imgCosmos128(Img2imgCosmos128):
  """ Smaller version of the Img2imgCosmos problem, at half the pixel
  resolution
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.img_len = 128
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag_auto', 'flux_radius', 'zphot']

@registry.register_problem
class Attrs2imgCosmos128Euclid(Img2imgCosmos128):
  """
  """

  def eval_metrics(self):
    eval_metrics = [ ]
    return eval_metrics

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.03
    p.img_len = 128
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag_auto', 'flux_radius', 'sersic_n', 'sersic_q']

@registry.register_problem
class Attrs2imgCosmos32(Attrs2imgCosmos):
  """ Lower resolution equivalent of conditional generation problem.
  """

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.06
    p.img_len = 32
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "attributes":  modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "attributes": None,
                    "targets": None}
    p.attributes = ['mag_auto', 'flux_radius', 'zphot', 'bulge_q', 'bulge_beta' ,
                    'disk_q', 'disk_beta', 'bulge_hlr', 'disk_hlr']


@registry.register_problem
class Attrs2imgCosmosEuclid2hst(Img2imgCosmos):

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
       We are going to split the GalSim data into shards of 1000 galaxies each,
       with 80 shards for training, 2 shards for validation.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 80,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 2,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.1
    p.img_len = 64
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf_euclid", None)
    p.add_hparam("psf_hst", None)
    p.add_hparam("rotation", False)
    p.attributes = ['mag_auto', 'flux_radius']
  """ Conditional image generation problem based on COSMOS sample.
  """

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image_euclid = example["inputs"]
    image_hst = example["targets"]

    # Clip to 1 the values of the image
    # image = tf.clip_by_value(image, -1, 1)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image_euclid
    example["targets"] = image_hst
    return example

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog()
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'/COSMOS_25.2_training_sample')

    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog.getNObjects()))

    # Extracts additional information about the galaxies
    cat_param = catalog.param_cat[catalog.orig_index]
    from numpy.lib.recfunctions import append_fields
    import numpy as np

    bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']
    # Parameters for a 2 component fit
    cat_param = append_fields(cat_param, 'bulge_q', bparams[:,11])
    cat_param = append_fields(cat_param, 'bulge_beta', bparams[:,15])
    cat_param = append_fields(cat_param, 'disk_q', bparams[:,3])
    cat_param = append_fields(cat_param, 'disk_beta', bparams[:,7])
    cat_param = append_fields(cat_param, 'bulge_hlr', cat_param['hlr'][:,1])
    cat_param = append_fields(cat_param, 'bulge_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,1]), np.zeros(len(cat_param) )))
    cat_param = append_fields(cat_param, 'disk_hlr', cat_param['hlr'][:,2])
    cat_param = append_fields(cat_param, 'disk_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,2]), np.log10(cat_param['flux'][:,0])))

    # Parameters for a single component fit
    cat_param = append_fields(cat_param, 'sersic_flux_log10', np.log10(sparams[:,0]))
    cat_param = append_fields(cat_param, 'sersic_q', sparams[:,3])
    cat_param = append_fields(cat_param, 'sersic_hlr', sparams[:,1])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:,2])
    cat_param = append_fields(cat_param, 'sersic_beta', sparams[:,7])
    
    # Generate a Euclid-like PSF
    lam = 700  # nm
    diam = 1.3    # meters
    psf = galsim.OpticalPSF(lam=lam, diam=diam, scale_unit=galsim.arcsec)

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale*2)

      # Apply random rotation if requested
      if hasattr(p, "rotation") and p.rotation:
        rotation_angle = galsim.Angle(-np.random.rand()* 2 * np.pi,
                                      galsim.radians)
        gal = gal.rotate(rotation_angle)
        psf = psf.rotate(rotation_angle)

      # We save the corresponding attributes for this galaxy
      if hasattr(p, 'attributes'):
        params = cat_param[ind]
        attributes = {k: params[k] for k in p.attributes}
      else:
        attributes = None

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale,
                                               attributes=attributes)

@registry.register_problem
class Attrs2imgCosmosCfht2hst(Img2imgCosmos):

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
       We are going to split the GalSim data into shards of 1000 galaxies each,
       with 80 shards for training, 2 shards for validation.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 49,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 2,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.187
    p.img_len = 64
    p.seed = 1995
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf_cfht", None)
    p.add_hparam("psf_hst", None)
    p.add_hparam("rotation", False)
    p.attributes = ['mag_auto', 'flux_radius']
  """ Conditional image generation problem based on COSMOS sample.
  """

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image_cfht = example["inputs"]
    image_hst = example["targets"]

    # Clip to 1 the values of the image
    # image = tf.clip_by_value(image, -1, 1)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image_cfht
    example["targets"] = image_hst
    return example

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog()
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'COSMOS_23.5_training_sample',sample='23.5')

    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog.getNObjects()))

    # Extracts additional information about the galaxies
    cat_param = catalog.param_cat[catalog.orig_index]
    from numpy.lib.recfunctions import append_fields
    import numpy as np

    bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']
    # Parameters for a 2 component fit
    cat_param = append_fields(cat_param, 'bulge_q', bparams[:,11])
    cat_param = append_fields(cat_param, 'bulge_beta', bparams[:,15])
    cat_param = append_fields(cat_param, 'disk_q', bparams[:,3])
    cat_param = append_fields(cat_param, 'disk_beta', bparams[:,7])
    cat_param = append_fields(cat_param, 'bulge_hlr', cat_param['hlr'][:,1])
    cat_param = append_fields(cat_param, 'bulge_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,1]), np.zeros(len(cat_param) )))
    cat_param = append_fields(cat_param, 'disk_hlr', cat_param['hlr'][:,2])
    cat_param = append_fields(cat_param, 'disk_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,2]), np.log10(cat_param['flux'][:,0])))

    # Parameters for a single component fit
    cat_param = append_fields(cat_param, 'sersic_flux_log10', np.log10(sparams[:,0]))
    cat_param = append_fields(cat_param, 'sersic_q', sparams[:,3])
    cat_param = append_fields(cat_param, 'sersic_hlr', sparams[:,1])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:,2])
    cat_param = append_fields(cat_param, 'sersic_beta', sparams[:,7])
    
    # Setting seeds for random number generators
    np.random.seed(seed=p.seed)
    fwhm_sampler = seeing_distribution(os.path.join(_COSMOS_DATA_DIR,'seeing_distribution.npy'),seed=p.seed)
    
    # Compute flux scaling factor to go from HST to CFHT
    # The values below were taken from the following two links
    # https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html
    # https://github.com/LSSTDESC/WeakLensingDeblending/blob/9f851f79f6f820f815528d11acabf64083b6e111/descwl/survey.py#L288
    cfht_eff_area = 8.022 #m^2 #effective area
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    exp_time = 200 #seconds # exposure time #value corresponding to CFIS # provided by A. Guinot
    qe = 0.77 # Quantum Efficiency (converts photon number to electrons)
    gain = 1.62 #e-/ADU #converts electrons to ADU
    flux_scaling = (cfht_eff_area/hst_eff_area) * exp_time * qe / gain

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale*2)
      # Scale galaxy flux
      gal = gal * flux_scaling

      # Load the COSMOS isotropic PSF to be used for the inputs
    
      psf = galsim.InterpolatedKImage(galsim.ImageCD(fits.getdata(os.path.join(_COSMOS_DATA_DIR,'hst_cosmos_effective_psf.fits'))+0j, scale=2.*np.pi/(0.03*128)))
        
      # Normalize the PSF
      psf = psf.withFlux(1.0)

      # Apply random rotation if requested
      if hasattr(p, "rotation") and p.rotation:
        rotation_angle = galsim.Angle(-np.random.rand()* 2 * np.pi,
                                      galsim.radians)
        gal = gal.rotate(rotation_angle)
        psf = psf.rotate(rotation_angle)

      # We save the corresponding attributes for this galaxy
      if hasattr(p, 'attributes'):
        params = cat_param[ind]
        attributes = {k: params[k] for k in p.attributes}
      else:
        attributes = None

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale,
                                               attributes=attributes,
                                               fwhm_sampler=fwhm_sampler)

@registry.register_problem
class Attrs2imgCosmosParametricCfht2hst(Img2imgCosmos):

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
       We are going to split the GalSim data into shards of 1000 galaxies each,
       with 80 shards for training, 2 shards for validation.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 49,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 2,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 1.5  
    p.img_len = 64
    p.seed = 1995
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
    p.add_hparam("psf_cfht", None)
    p.add_hparam("psf_hst", None)
    p.add_hparam("rotation", False)
    p.attributes = ['mag_auto', 'flux_radius']
  """ Conditional image generation problem based on COSMOS sample.
  """

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image_cfht = example["inputs"]
    image_hst = example["targets"]

    # Clip to 1 the values of the image
    # image = tf.clip_by_value(image, -1, 1)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["inputs"] = image_cfht
    example["targets"] = image_hst
    return example

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog(sample='23.5', use_real=False)
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'COSMOS_23.5_training_sample',sample='23.5', use_real=False)

    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog.getNObjects()))

    # Extracts additional information about the galaxies
    cat_param = catalog.param_cat[catalog.orig_index]
    from numpy.lib.recfunctions import append_fields
    import numpy as np

    bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']
    # Parameters for a 2 component fit
    cat_param = append_fields(cat_param, 'bulge_q', bparams[:,11])
    cat_param = append_fields(cat_param, 'bulge_beta', bparams[:,15])
    cat_param = append_fields(cat_param, 'disk_q', bparams[:,3])
    cat_param = append_fields(cat_param, 'disk_beta', bparams[:,7])
    cat_param = append_fields(cat_param, 'bulge_hlr', cat_param['hlr'][:,1])
    cat_param = append_fields(cat_param, 'bulge_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,1]), np.zeros(len(cat_param) )))
    cat_param = append_fields(cat_param, 'disk_hlr', cat_param['hlr'][:,2])
    cat_param = append_fields(cat_param, 'disk_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,2]), np.log10(cat_param['flux'][:,0])))

    # Parameters for a single component fit
    cat_param = append_fields(cat_param, 'sersic_flux_log10', np.log10(sparams[:,0]))
    cat_param = append_fields(cat_param, 'sersic_q', sparams[:,3])
    cat_param = append_fields(cat_param, 'sersic_hlr', sparams[:,1])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:,2])
    cat_param = append_fields(cat_param, 'sersic_beta', sparams[:,7])
    
    # Setting seeds for random number generators
    np.random.seed(seed=p.seed)
    fwhm_sampler = seeing_distribution(os.path.join(_COSMOS_DATA_DIR,'seeing_distribution.npy'),seed=p.seed)
    
    # Compute flux scaling factor to go from HST to CFHT
    # The values below were taken from the following two links
    # https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html
    # https://github.com/LSSTDESC/WeakLensingDeblending/blob/9f851f79f6f820f815528d11acabf64083b6e111/descwl/survey.py#L288
    cfht_eff_area = 8.022 #m^2 #effective area
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    exp_time = 200 #seconds # exposure time #value corresponding to CFIS # provided by A. Guinot
    qe = 0.77 # Quantum Efficiency (converts photon number to electrons)
    gain = 1.62 #e-/ADU #converts electrons to ADU
    flux_scaling = (cfht_eff_area/hst_eff_area) * exp_time * qe / gain
    
    # allow the fft operation in galsim to occupy more memory
    gsp = galsim.GSParams(maximum_fft_size=10000)

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale*2, gsparams=gsp)
      # Scale galaxy flux
      gal = gal * flux_scaling

      # Load the COSMOS isotropic PSF to be used for the inputs
    
      psf = galsim.InterpolatedKImage(galsim.ImageCD(fits.getdata(os.path.join(_COSMOS_DATA_DIR,'hst_cosmos_effective_psf.fits'))+0j, scale=2.*np.pi/(0.03*128)))
        
      # Normalize the PSF
      psf = psf.withFlux(1.0)

      # Apply random rotation if requested
      if hasattr(p, "rotation") and p.rotation:
        rotation_angle = galsim.Angle(-np.random.rand()* 2 * np.pi,
                                      galsim.radians)
        gal = gal.rotate(rotation_angle)
        psf = psf.rotate(rotation_angle)

      # We save the corresponding attributes for this galaxy
      if hasattr(p, 'attributes'):
        params = cat_param[ind]
        attributes = {k: params[k] for k in p.attributes}
      else:
        attributes = None

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_parametric_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale,
                                               attributes=attributes,
                                               fwhm_sampler=fwhm_sampler)



@registry.register_problem
class meerkat(Img2imgCosmos):

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.
       Note that each shard will be produced in parallel.
       We are going to split the GalSim data into shards of 1000 galaxies each,
       with 80 shards for training, 2 shards for validation.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1, #49,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1, #2,
    }]

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 1.5  
    p.img_len = 64
    p.seed = 1995
    p.example_per_shard = 1000
   
    p.modality = {"targets": modalities.ModalityType.IDENTITY} 
    p.vocab_size = {"targets": None} 
    p.add_hparam("psf_cfht", None)
    p.add_hparam("rotation", False)
    p.attributes = ['mag_auto', 'flux_radius']
  """ Conditional image generation problem based on COSMOS sample.
  """

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    p = self.get_hparams()
    image_hst = example["targets"]

    # Clip to 1 the values of the image
    # image = tf.clip_by_value(image, -1, 1)

    # Aggregate the conditions
    if hasattr(p, 'attributes'):
      example['attributes'] = tf.stack([example[k] for k in p.attributes])

    example["targets"] = image_hst
    return example

  def generator(self, data_dir, tmp_dir, dataset_split, task_id=-1):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    p = self.get_hparams()
    """
    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog(use_real=False)
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'catalogue_SFGs_complete_wide1.fits', use_real=False)
    
   """

    try:
        # try to use default galsim path to the data
        catalog = galsim.COSMOSCatalog(sample='23.5', use_real=False)
    except:
        # If that fails, tries to use the specified tmp_dir
        catalog = galsim.COSMOSCatalog(dir=tmp_dir+'COSMOS_23.5_training_sample',sample='23.5', use_real=False)
        
        
        
    # Create a list of galaxy indices for this task, remember, there is a task
    # per shard, each shard is 1000 galaxies.
    assert(task_id > -1)
    index = range(task_id*p.example_per_shard,
                  min((task_id+1)*p.example_per_shard, catalog.getNObjects()))

    # Extracts additional information about the galaxies
    cat_param = catalog.param_cat[catalog.orig_index]
    from numpy.lib.recfunctions import append_fields
    import numpy as np

    bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']
    # Parameters for a 2 component fit
    cat_param = append_fields(cat_param, 'bulge_q', bparams[:,11])
    cat_param = append_fields(cat_param, 'bulge_beta', bparams[:,15])
    cat_param = append_fields(cat_param, 'disk_q', bparams[:,3])
    cat_param = append_fields(cat_param, 'disk_beta', bparams[:,7])
    cat_param = append_fields(cat_param, 'bulge_hlr', cat_param['hlr'][:,1])
    cat_param = append_fields(cat_param, 'bulge_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,1]), np.zeros(len(cat_param) )))
    cat_param = append_fields(cat_param, 'disk_hlr', cat_param['hlr'][:,2])
    cat_param = append_fields(cat_param, 'disk_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,2]), np.log10(cat_param['flux'][:,0])))

    # Parameters for a single component fit
    cat_param = append_fields(cat_param, 'sersic_flux_log10', np.log10(sparams[:,0]))
    cat_param = append_fields(cat_param, 'sersic_q', sparams[:,3])
    cat_param = append_fields(cat_param, 'sersic_hlr', sparams[:,1])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:,2])
    cat_param = append_fields(cat_param, 'sersic_beta', sparams[:,7])
    
    # Setting seeds for random number generators
    np.random.seed(seed=p.seed)
    fwhm_sampler = seeing_distribution(os.path.join(_COSMOS_DATA_DIR,'seeing_distribution.npy'),seed=p.seed)
    
    # Compute flux scaling factor to go from HST to CFHT
    # The values below were taken from the following two links
    # https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html
    # https://github.com/LSSTDESC/WeakLensingDeblending/blob/9f851f79f6f820f815528d11acabf64083b6e111/descwl/survey.py#L288
    cfht_eff_area = 8.022 #m^2 #effective area
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    exp_time = 200 #seconds # exposure time #value corresponding to CFIS # provided by A. Guinot
    qe = 0.77 # Quantum Efficiency (converts photon number to electrons)
    gain = 1.62 #e-/ADU #converts electrons to ADU
    flux_scaling = (cfht_eff_area/hst_eff_area) * exp_time * qe / gain
    
    # allow the fft operation in galsim to occupy more memory
    gsp = galsim.GSParams(maximum_fft_size=10000)

    
        
    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale*2, gsparams=gsp)
      # Scale galaxy flux
      gal = gal * flux_scaling

      # Load the COSMOS isotropic PSF to be used for the inputs
      """
      psf = galsim.InterpolatedKImage(galsim.ImageCD(fits.getdata(os.path.join(_COSMOS_DATA_DIR,'hst_cosmos_effective_psf.fits'))+0j, scale=2.*np.pi/(0.03*128)))"""
      
      PSF, sampling, tabDEC, tabHAstart = MakePSF.compute(1, p.img_len, p.pixel_scale, Obslength=2., Timedelta=300, Elevmin=0., F=1420)
        
      
      #mask = sampling[0] + np.conj(sampling[0]).T   # Make Hermitian
    
      #psf = galsim.InterpolatedKImage(galsim.ImageCD(mask, scale = 2.*np.pi / (p.pixel_scale * p.img_len)))
      
      psf = galsim.InterpolatedImage(galsim.ImageD(np.real(PSF[0]), scale = p.pixel_scale))
        
        
      # Normalize the PSF
      psf = psf.withFlux(1.0)

      # Apply random rotation if requested
      if hasattr(p, "rotation") and p.rotation:
        rotation_angle = galsim.Angle(-np.random.rand()* 2 * np.pi,
                                      galsim.radians)
        gal = gal.rotate(rotation_angle)
        psf = psf.rotate(rotation_angle)

      # We save the corresponding attributes for this galaxy
      if hasattr(p, 'attributes'):
        params = cat_param[ind]
        attributes = {k: params[k] for k in p.attributes}
      else:
        attributes = None

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale,
                                               attributes=attributes,
                                               fwhm_sampler=fwhm_sampler)
        
        
