img_norm_cfg = dict(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=True)
albu_val_transforms = [
    dict(type="LongestMaxSize", max_size=189),
    dict(type="PadIfNeeded", min_height=189, min_width=165, border_mode=0,
         value=list(img_norm_cfg["mean"])[::-1]),
]
test_pipeline = [
    dict(
        type="Albu",
        transforms=albu_val_transforms,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"])]