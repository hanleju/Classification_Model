from detectron2.data.datasets import register_coco_instances

register_coco_instances("baggage", {},"C:/Users/root/Desktop/pidray/pidray/annotations/xray_train.json", "C:/Users/root/Desktop/pidray/pidray/train")
register_coco_instances("baggage_test", {}, "C:/Users/root/Desktop/pidray/pidray/annotations/xray_test_easy.json", "C:/Users/root/Desktop/pidray/pidray/easy")