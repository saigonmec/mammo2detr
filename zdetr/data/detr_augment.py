import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_m2_positive_augmentations(height, width, enable_rotate90=True):
    """
    Augmentations for POSITIVE samples (with bbox) - bbox-safe transforms
    Tránh các transform có thể làm mất bbox như CoarseDropout, ElasticTransform mạnh
    """
    crop_h = int(height * 0.8)
    crop_w = int(width * 0.8)

    aug_list = [
        # Bổ sung crop an toàn cho bbox
        A.RandomSizedBBoxSafeCrop(
            height=crop_h, width=crop_w, erosion_rate=0.0, p=0.35
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        # Geometric transforms - bbox-aware
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.1, 0.1),
            rotate=(-45, 45),
            shear=(-15, 15),
            p=0.5,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.4,
        ),
        # Color/Intensity transforms (bbox-safe)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        A.Equalize(p=0.2),
        # Blur/Sharpen (bbox-safe)
        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.5), p=1.0),
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ],
            p=0.3,
        ),
        A.UnsharpMask(
            blur_limit=(3, 7), sigma_limit=(0.0, 3.0), alpha=(0.2, 0.5), p=0.2
        ),
        # Noise (bbox-safe, nhẹ)
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),
        # Light distortion (bbox-safe, nhẹ hơn)
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.2),
        # Downscale (bbox-safe)
        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.3,
                ),
                A.Downscale(
                    scale_range=(0.85, 0.85),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LANCZOS4,
                    },
                    p=0.7,
                ),
            ],
            p=0.15,
        ),
        # Color jitter
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
        A.ChannelShuffle(p=0.1),
    ]

    if enable_rotate90:
        aug_list.insert(3, A.RandomRotate90(p=0.5))

    return aug_list


def get_m2_negative_augmentations(height, width, enable_rotate90=True):
    """
    Augmentations for NEGATIVE samples (no bbox) - can use aggressive transforms
    Tận dụng lại augmentation mạnh từ base (có thể dùng CoarseDropout, ElasticTransform...)
    """
    aug_list = [
        # Crop tối đa 10% mỗi cạnh (trái/phải/trên/dưới)
        A.CropAndPad(
            percent=(-0.20, 0.0),  # chỉ crop, không pad, tối đa 20% mỗi cạnh
            pad_mode=cv2.BORDER_CONSTANT,
            pad_cval=0,
            p=0.35,
        ),
        # Geometric transforms - có thể mạnh hơn
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.1, 0.1),
            rotate=(-45, 45),
            shear=(-15, 15),
            p=0.5,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.4,
        ),
        # Color/Intensity
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        A.Equalize(p=0.2),
        # Blur/Sharpen
        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.5), p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                # A.MedianBlur(blur_limit=5, p=1.0),
                # A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=0.3,
        ),
        A.UnsharpMask(
            blur_limit=(3, 7), sigma_limit=(0.0, 3.0), alpha=(0.2, 0.5), p=0.2
        ),
        # Noise
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.2),
        # Downscale
        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.3,
                ),
                A.Downscale(
                    scale_range=(0.85, 0.85),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LANCZOS4,
                    },
                    p=0.7,
                ),
            ],
            p=0.15,
        ),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
        ),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        A.ChannelShuffle(p=0.1),
        # Elastic transforms - ONLY for negative
        # A.ElasticTransform(alpha=1, sigma=30, p=0.15),
        # A.GridDistortion(num_steps=5, distort_limit=0.4, p=0.2),
        # A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.2),
        # A.ChannelShuffle(p=0.15),
        # Spatial dropout - ONLY for negative
        # A.CoarseDropout(
        #    num_holes_range=(1, 5),
        #    hole_height_range=(0.01, 0.1),
        #    hole_width_range=(0.01, 0.15),
        #    fill=0,
        #    p=0.2,
        # ),
        # Advanced color
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
        # A.ToGray(p=0.05),
        # A.ToSepia(p=0.05),
    ]

    if enable_rotate90:
        aug_list.insert(4, A.RandomRotate90(p=0.5))

    return aug_list


def get_m2_train_augmentation(height, width, enable_rotate90=True):
    """
    Training augmentation with bbox support
    Trả về 2 transforms: 1 cho positive (bbox-safe), 1 cho negative (aggressive)
    """

    big_h = int(height * 1.2)
    big_w = int(width * 1.2)

    # Positive transform (bbox-safe)
    positive_aug_list = [A.Resize(big_h, big_w)]
    positive_aug_list += get_m2_positive_augmentations(
        height, width, enable_rotate90=enable_rotate90
    )
    positive_aug_list += [
        A.Resize(height, width),
        A.Normalize([0.5] * 3, [0.5] * 3),
        ToTensorV2(),
    ]

    positive_transform = A.Compose(
        positive_aug_list,
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height] - giữ nguyên format gốc
            label_fields=["labels"],
            min_area=100.0,  # Tối thiểu 100 pixel^2
            min_visibility=0.3,  # Tối thiểu 30% visible
        ),
    )

    # Negative transform (aggressive, no bbox) - KHÔNG CẦN bbox_params
    negative_aug_list = [A.Resize(big_h, big_w)]
    negative_aug_list += get_m2_negative_augmentations(
        height, width, enable_rotate90=enable_rotate90
    )
    negative_aug_list += [
        A.Resize(height, width),
        A.Normalize([0.5] * 3, [0.5] * 3),
        ToTensorV2(),
    ]

    negative_transform = A.Compose(negative_aug_list)  # Không có bbox_params

    return positive_transform, negative_transform


def get_m2_test_augmentation(height, width):
    """
    Test augmentation with bbox support (minimal transforms)
    """
    aug_list = [
        A.Resize(height, width),
        A.Normalize([0.5] * 3, [0.5] * 3),
        ToTensorV2(),
    ]

    return A.Compose(
        aug_list,
        bbox_params=A.BboxParams(
            format="coco",  # [x, y, width, height]
            label_fields=["labels"],
            min_area=0.0,
            min_visibility=0.0,  # Test mode: giữ tất cả bbox
        ),
    )


def get_m2_test_augmentation_no_bbox(height, width):
    """
    Test augmentation WITHOUT bbox support (for negative samples in test mode)
    """
    aug_list = [
        A.Resize(height, width),
        A.Normalize([0.5] * 3, [0.5] * 3),
        ToTensorV2(),
    ]

    return A.Compose(aug_list)  # No bbox_params
