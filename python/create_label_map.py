r"""Create label map from dataset


Example usage:
    ./create_label_map.py --data_dir=/home/data/ \
        --label_map_path=/home/data/label_map.pbtxt
"""

import os
import io
import logging
from collections import defaultdict

from lxml import etree
import PIL.Image

from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
from object_detection.protos import string_int_label_map_pb2
import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('label_map_path', 'data/label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def get_class_set(annotations_dir, image_dir):
	categories = defaultdict(int)
	for filename in os.listdir(annotations_dir):
		try:
			with tf.gfile.GFile(os.path.join(annotations_dir,  filename), 'r') as fid:
				xml_str = fid.read()
			logging.info("xml: ", xml_str)
			xml = etree.fromstring(xml_str)
			data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
			img_path = os.path.join(image_dir, os.path.basename(data['filename']))
			logging.info("image path: ", img_path)
			with tf.gfile.GFile(img_path, 'rb') as fid:
				encoded_jpg = fid.read()
			encoded_jpg_io = io.BytesIO(encoded_jpg)
			image = PIL.Image.open(encoded_jpg_io)
			if image.format != 'JPEG':
				raise ValueError('Image format not JPEG')
			for obj in data['object']:
				categories[obj['name']] += 1
		except Exception as e:
			logging.exception('Could not decode xml')
	return categories


def write_label_map(categories, label_map_path):
	label_map = string_int_label_map_pb2.StringIntLabelMap()
	label_map_items = []
	for i, category in enumerate(categories):
		idx = i + 1
		proto = string_int_label_map_pb2.StringIntLabelMapItem()
		proto.id = idx
		proto.name = category

		label_map_items.append(proto)

	label_map.item.extend(label_map_items)
	label_map_str = text_format.MessageToString(label_map)
	with open(label_map_path, 'w') as label_map_file:
		label_map_file.write(label_map_str)


def main(_):  
  data_dir = FLAGS.data_dir

  logging.info('Creating label map from dataset')
  annotations_dir = os.path.join(data_dir, 'annotations')
  image_dir = os.path.join(data_dir, 'images')

  categories = get_class_set(annotations_dir, image_dir)
  logging.info("No of objects per categories: ", categories)

  logging.info("Creating label map proto file")
  write_label_map(categories, FLAGS.label_map_path)

if __name__ == '__main__':
  tf.app.run()