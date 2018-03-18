import os
import sys
import logging
import tarfile
from six.moves import urllib
import json
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import config_util

flags = tf.app.flags
flags.DEFINE_string('architecture', '', 'Name of architecture')
flags.DEFINE_string('experiment_id', '', 'Id of current experiment output')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
					'Path to label map proto')
flags.DEFINE_string('data_dir', 'data/',
					'Path to label map proto')
flags.DEFINE_string('hparams', '',
					'Params in json')
FLAGS = flags.FLAGS

# Map of architecture to configs and urls
arch_map = {
	'ssd_mobilenet_v1_coco': {
		'config': 'ssd_mobilenet_v1_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
		'checkpoint': 'ssd_mobilenet_v1_coco_2017_11_17'
	},
	'ssd_inception_v2_coco': {
		'config': 'ssd_inception_v2_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz',
		'checkpoint': 'ssd_inception_v2_coco_2017_11_17'
	},
	'faster_rcnn_inception_v2_coco': {
		'config': 'faster_rcnn_inception_v2_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
		'checkpoint': 'faster_rcnn_inception_v2_coco_2018_01_28'
	},
	'faster_rcnn_resnet50_coco': {
		'config': 'faster_rcnn_resnet50_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
		'checkpoint': 'faster_rcnn_resnet50_coco_2018_01_28'
	},
	'rfcn_resnet101_coco': {
		'config': 'rfcn_resnet101_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz',
		'checkpoint': 'rfcn_resnet101_coco_2018_01_28'
	},
	'faster_rcnn_resnet101_coco': {
		'config': 'faster_rcnn_resnet101_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz',
		'checkpoint': 'faster_rcnn_resnet101_coco_2018_01_28'
	},
	'faster_rcnn_inception_resnet_v2_atrous_coco': {
		'config': 'faster_rcnn_inception_resnet_v2_atrous_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz',
		'checkpoint': 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
	},
	'faster_rcnn_nas': {
		'config': 'faster_rcnn_nas_coco.config',
		'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
		'checkpoint': 'faster_rcnn_nas_coco_2018_01_28'
	}
}

def maybe_download_and_extract(url, output_dir):
	"""Download and extract model tar file.

	If the pretrained model we're using doesn't already exist, this function
	downloads it from the TensorFlow.org website and unpacks it into a directory.
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	filename = url.split('/')[-1]
	filepath = os.path.join(output_dir, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' %
							 (filename,
							  float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()

		filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(output_dir)

def main(_):
  arch_details = arch_map[FLAGS.architecture]
  # check graph type, download graph
  graph_url = arch_details['url']
  graph_path = '/models/research/object_detection/data/'
  maybe_download_and_extract(graph_url, graph_path)
  # Open config file
  config_path = os.path.join('/models/research/object_detection/samples/configs', 
  	arch_details['config'])
  configs = config_util.get_configs_from_pipeline_file(config_path)
  # Update paths in config
  hparams = tf.contrib.training.HParams(label_map_path=FLAGS.label_map_path, 
  	train_input_path=os.path.join(FLAGS.data_dir, 'train.record'),
  	eval_input_path=os.path.join(FLAGS.data_dir, 'val.record'))

  if FLAGS.hparams:
  	for key, val in json.loads(FLAGS.hparams).iteritems():
	  hparams.add_hparam(key, val)
  
  config_util.merge_external_params_with_configs(configs, hparams)
  # Save config inside dataset

  configs["train_config"].fine_tune_checkpoint = os.path.join(graph_path, 
  	arch_details['checkpoint'], 'model.ckpt')

  config_proto = config_util.create_pipeline_proto_from_configs(configs)
  config_str = text_format.MessageToString(config_proto)
  
  experiment_path = os.path.join(FLAGS.data_dir, FLAGS.experiment_id)
  if not os.path.exists(experiment_path):
  	os.makedirs(experiment_path)
  with open(os.path.join(experiment_path, 'pipeline.config'), 'w') as config_file:
	config_file.write(config_str)

if __name__ == '__main__':
  tf.app.run()