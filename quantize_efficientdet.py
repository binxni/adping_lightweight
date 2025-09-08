#!/usr/bin/env python3
"""Quantize EfficientDet-Lite0 to INT8 and evaluate mAP, FPS, and model size."""

import argparse
import json
import os
import tempfile
import time
from typing import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_coco_dataset(split: str = "validation", img_size: int = 320) -> tf.data.Dataset:
    """Loads and preprocesses the COCO 2017 dataset."""
    ds, _ = tfds.load("coco/2017", split=split, with_info=True)

    def _preprocess(sample):
        image = tf.image.resize(sample["image"], (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, sample["image/id"]

    return ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)


def representative_dataset(ds: tf.data.Dataset, num_samples: int = 100) -> Iterable[np.ndarray]:
    for image, _ in ds.take(num_samples):
        yield [tf.expand_dims(image, 0)]


def convert_to_tflite(saved_model_dir: str, rep_ds: Iterable[np.ndarray], tflite_path: str) -> float:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_ds
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    start = time.time()
    tflite_model = converter.convert()
    conv_time = time.time() - start

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return conv_time


def evaluate_map(tflite_path: str, ds: tf.data.Dataset, annotation_file: str, img_size: int = 320) -> float:
    coco = COCO(annotation_file)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    results = []
    for image, image_id in ds:
        interpreter.set_tensor(input_details["index"], tf.expand_dims(tf.cast(image * 255, tf.uint8), 0))
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        count = int(interpreter.get_tensor(output_details[3]["index"])[0])
        for i in range(count):
            y_min, x_min, y_max, x_max = boxes[i]
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(classes[i]) + 1,
                    "bbox": [x_min * img_size, y_min * img_size,
                             (x_max - x_min) * img_size,
                             (y_max - y_min) * img_size],
                    "score": float(scores[i]),
                }
            )

    fd, temp_json = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(temp_json)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map50 = coco_eval.stats[1]
    os.remove(temp_json)
    return map50


def measure_fps(tflite_path: str, num_runs: int = 50) -> float:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    dummy = np.random.randint(0, 255, size=input_details["shape"], dtype=np.uint8)
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()

    start = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details["index"], dummy)
        interpreter.invoke()
    total = time.time() - start
    return num_runs / total


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize EfficientDet-Lite0 and evaluate")
    parser.add_argument(
        "--annotation_file",
        help="Path to COCO instances_val2017.json (downloaded if omitted)",
    )
    parser.add_argument(
        "--output", default="efficientdet-lite0-int8.tflite", help="Output TFLite file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_coco_dataset()
    rep_ds = representative_dataset(dataset)

    model_handle = "https://tfhub.dev/tensorflow/efficientdet/lite0/1"
    saved_model_dir = tf.keras.utils.get_file("efficientdet_lite0", model_handle, untar=True)

    conv_time = convert_to_tflite(saved_model_dir, rep_ds, args.output)

    annotation_file = args.annotation_file
    if annotation_file is None:
        ann_zip = tf.keras.utils.get_file(
            "annotations_trainval2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            extract=True,
        )
        annotation_file = os.path.join(
            os.path.dirname(ann_zip), "annotations", "instances_val2017.json"
        )

    map50 = evaluate_map(args.output, dataset.take(100), annotation_file)
    fps = measure_fps(args.output)
    model_size = os.path.getsize(args.output) / (1024 * 1024)

    print(f"mAP@50: {map50:.3f}")
    print(f"FPS: {fps:.2f}")
    print(f"Conversion time: {conv_time:.2f}s")
    print(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":
    main()

