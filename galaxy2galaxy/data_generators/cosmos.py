"""HST/ACS Cosmos generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import galsim_utils
from . import astroimage_utils

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

from galaxy2galaxy.utils import registry

import tensorflow as tf
import galsim


@registry.register_problem
class GalsimCosmos64(galsim_utils.GalsimProblem):
  """
  Subclass of GalSim problem implementing drawing galaxies from the COSMOS
  25.2 sample.
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

    for ind in index:
      # Draw a galaxy using GalSim, any kind of operation can be done here
      gal = catalog.makeGalaxy(ind, noise_pad_size=p.img_len * p.pixel_scale)

      # We apply the orginal psf
      psf = gal.original_psf

      # Utility function encodes the postage stamp for serialized features
      yield galsim_utils.draw_and_encode_stamp(gal, psf,
                                               stamp_size=p.img_len,
                                               pixel_scale=p.pixel_scale)

  def preprocess_example(self, example, unused_mode, unused_hparams):
    """ Preprocess the examples, can be used for further augmentation or
    image standardization.
    """
    image = example["inputs"]

    # Clip to 1 the values of the image
    image = tf.clip_by_value(image, -1, 1)

    example["inputs"] = image
    example["targets"] = image
    return example

@registry.register_problem
class GalsimCosmos32(GalsimCosmos64):

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.pixel_scale = 0.06
    p.img_len = 32
    p.example_per_shard = 1000
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": None,
                    "targets": None}
