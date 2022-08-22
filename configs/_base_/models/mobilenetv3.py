model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="TIMMBackbone",
        pretrained=False,
        model_name="mobilenetv3_large_100_miil_in21k",
        init_cfg=None,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=4,
        in_channels=1280,
        topk=(1,),
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
    ),
)