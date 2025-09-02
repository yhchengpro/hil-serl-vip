from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt") 
results = model("/home/kkk/workspace/hil-serl/photos/photo19.png")

for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
results[0].show()