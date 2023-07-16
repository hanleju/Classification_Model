# Copyright (c) Facebook, Inc. and its affiliates.
from .coco import register_coco_instances  # noqa
from .coco_panoptic import register_coco_panoptic_separated  # noqa


register_coco_instances("baggage", {},"C:/Users/root/Desktop/pidray/pidray/annotations/xray_train.json", "C:/Users/root/Desktop/pidray/pidray/train")
register_coco_instances("baggage_test", {}, "C:/Users/root/Desktop/pidray/pidray/annotations/xray_test_easy.json", "C:/Users/root/Desktop/pidray/pidray/easy")