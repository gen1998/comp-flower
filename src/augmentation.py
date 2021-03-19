from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize,ToGray
)

from albumentations.pytorch import ToTensorV2

def get_train_transforms(input_shape):
    return Compose([
            Resize(input_shape[0], input_shape[1]),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms(input_shape):
    return Compose([
                Resize(input_shape[0], input_shape[1]),
                Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
                ToTensorV2(p=1.0),
            ], p=1.)

def get_inference_transforms(input_shape):
    return Compose([
                Resize(input_shape[0], input_shape[1]),
                ShiftScaleRotate(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)