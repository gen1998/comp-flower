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
            ShiftScaleRotate(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms(input_shape):
    return Compose([
                Resize(input_shape[0], input_shape[1]),
                ToTensorV2(p=1.0),
            ], p=1.)

def get_inference_transforms(input_shape):
    return Compose([
                Resize(input_shape[0], input_shape[1]),
                ShiftScaleRotate(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)