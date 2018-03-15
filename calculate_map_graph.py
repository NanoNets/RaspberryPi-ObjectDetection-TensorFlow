"""
Script to calculate MAP with quantzed or frozen graph

"""
import logging

from object_detection import eval_util
from object_detection.utils import object_detection_evaluation

# Create evaluation for all test images

# Tensorflow op so cant use with our graph: convert to result dict with eval_util.result_dict_for_single_example
evaluator.add_single_ground_truth_image_info(
              image_id=batch, groundtruth_dict=result_dict)
evaluator.add_single_detected_image_info(
              image_id=batch, detections_dict=result_dict)
metrics = evaluator.evaluate()
evaluator.clear()