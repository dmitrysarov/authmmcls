# model settings
num_classes = 5

model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="MobileNetV2",
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone',
    ),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=num_classes,
        in_channels=1280,
        topk=(1,),
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
    ),
)