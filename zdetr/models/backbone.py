import torch.nn as nn
import torchvision.models as models
import timm.models as timm_models

# from fastervit import create_model
import torch.hub as hub
import torch.serialization
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def get_resnet_backbone(model_type="resnet50"):
    if model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnext50":
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError("Unsupported resnet backbone type")
    return model, feature_dim


def get_timm_backbone(model_type):
    if model_type == "resnest50":
        model = timm_models.create_model("resnest50d", pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnest101":
        model = timm_models.create_model("resnest101e", pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "resnest50s2":
        model = timm_models.create_model("resnest50d_4s2x40d", pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_type == "regnety":
        model = timm_models.create_model("regnety_080_tv", pretrained=True)
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "convnextv2base":
        model = timm_models.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "convnextv2_tiny":
        model = timm_models.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "efficientnetv2":
        # Default: efficientnetv2_rw_m.agc_in1k (214 MB)
        model = timm_models.create_model(
            "efficientnetv2_rw_m.agc_in1k", pretrained=True
        )
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif model_type == "efficientnetv2s":
        # efficientnetv2_rw_s.ra2_in1k (96 MB)
        model = timm_models.create_model(
            "efficientnetv2_rw_s.ra2_in1k", pretrained=True
        )
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif model_type == "maxvit_tiny":
        model = timm_models.create_model("maxvit_tiny_tf_224.in1k", pretrained=True)
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "maxvit_tiny512":
        model = timm_models.create_model(
            "timm/maxvit_tiny_tf_512.in1k", pretrained=True
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "maxvit_small":
        model = timm_models.create_model("maxvit_small_tf_224.in1k", pretrained=True)
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "maxvit_base":
        model = timm_models.create_model("maxvit_base_tf_224.in1k", pretrained=True)
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "eva02_small":
        model = timm_models.create_model(
            "eva02_small_patch14_224.mim_in22k",
            pretrained=True,
            dynamic_img_size=True,
        )
        # Eva02 model's head is usually called 'head'
        # Try to get feature_dim from common attributes
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif hasattr(model, "num_features"):
            feature_dim = model.num_features
            model.head = nn.Identity()
        else:
            raise ValueError(
                "Cannot determine feature_dim for eva02_small_patch14_224.mim_in22k"
            )
    elif model_type == "vit_small":
        model = timm_models.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            dynamic_img_size=True,
        )
        # Lấy feature_dim từ head nếu có, hoặc num_features
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif hasattr(model, "num_features"):
            feature_dim = model.num_features
            model.head = nn.Identity()
        else:
            raise ValueError(
                "Cannot determine feature_dim for vit_small_patch14_reg4_dinov2.lvd142m"
            )
    elif model_type == "eva02_base":
        model = timm_models.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in1k",
            pretrained=True,
            dynamic_img_size=True,
        )
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif hasattr(model, "num_features"):
            feature_dim = model.num_features
            model.head = nn.Identity()
        else:
            raise ValueError(
                "Cannot determine feature_dim for eva02_base_patch14_448.mim_in22k_ft_in1k"
            )
    elif model_type == "swinv2_small":
        model = timm_models.create_model(
            "swinv2_small_window8_256.ms_in1k",
            pretrained=True,
            dynamic_img_size=True,
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "swinv2_tiny":
        model = timm_models.create_model(
            "swinv2_tiny_window8_256.ms_in1k",
            pretrained=True,
            dynamic_img_size=True,
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    elif model_type == "swinv2_base":
        model = timm_models.create_model(
            "swinv2_base_window8_256.ms_in1k",
            pretrained=True,
            dynamic_img_size=True,
        )
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    # --- MambaVision-T support ---
    else:
        raise ValueError("Unsupported timm backbone type")
    return model, feature_dim


def get_mamba_backbone(model_type, num_classes=None):
    if model_type == "mamba_t":
        model_wrapper = MambaVisionLogitsWrapper(
            model_name="nvidia/MambaVision-T-1K",
            num_classes=num_classes,
        )
        feature_dim = model_wrapper.feature_dim
        model = model_wrapper
        return model, feature_dim
    elif model_type == "mamba_s":
        model_wrapper = MambaVisionLogitsWrapper(
            model_name="nvidia/MambaVision-S-1K",
            num_classes=num_classes,
        )
        feature_dim = model_wrapper.feature_dim
        model = model_wrapper
        return model, feature_dim
    else:
        raise ValueError("Unsupported mamba backbone type")


class MambaVisionLogitsWrapper(nn.Module):
    def __init__(self, model_name="nvidia/MambaVision-T-1K", num_classes=None):
        super().__init__()
        from transformers import AutoModelForImageClassification

        base = AutoModelForImageClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.feature_dim = base.model.head.in_features

        # GÁN ĐÚNG VỊ TRÍ
        if num_classes is not None:
            base.model.head = nn.Linear(self.feature_dim, num_classes)
        else:
            base.model.head = nn.Identity()

        self.base_model = base

    def forward(self, x):
        return self.base_model(x)["logits"]


def get_dino_backbone(model_type="dinov2_vitb14", weights=None):
    dino_models = {
        "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
        "dinov2_small_reg": "vit_small_patch14_reg4_dinov2.lvd142m",
        "dinov2_base_reg": "vit_base_patch14_reg4_dinov3.lvd1689m",
        "dinov3_convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
        "dinov3_convnext_small": "convnext_small.dinov3_lvd1689m",
        "dinov3_vit16small": "vit_small_patch16_dinov3_qkvb.lvd1689m",
        "dinov3_vit16smallplus": "vit_small_plus_patch16_dinov3.lvd1689m",
        "dinov3_convnext_base": "convnext_base.dinov3_lvd1689m",
        "dinov3_vit16base": "vit_base_patch16_dinov3.lvd1689m",
        # larger models:
        "dinov3_vit16large": "vit_large_patch16_dinov3.lvd1689m",
        "dinov3_convnext_large": "convnext_large.dinov3_lvd1689m",
    }

    # --- Add medino_vitb16 support ---
    if model_type == "medino_vitb16":
        from huggingface_hub import hf_hub_download
        import torch

        ckpt_path = hf_hub_download(
            "TommyNgx/meddino_vitb16", "meddino_vitb16_ct3m.pth"
        )
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Giả sử model giống vit_base_patch16_dinov3 về kiến trúc
        if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif hasattr(model, "num_features") and model.num_features is not None:
            feature_dim = model.num_features
        elif hasattr(model, "head") and hasattr(model.head, "in_features"):
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        else:
            raise ValueError("Cannot determine feature_dim for medino_vitb16")
        return model, feature_dim

    # --- Add dinov2uni_base support ---
    if model_type == "dinov2uni_base":
        import timm

        model = timm.create_model(
            "hf-hub:TommyNgx/UniViT",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        # Lấy feature_dim từ head nếu có, hoặc num_features
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif hasattr(model, "num_features"):
            feature_dim = model.num_features
            model.head = nn.Identity()
        else:
            raise ValueError("Cannot determine feature_dim for dinov2uni_base")
        return model, feature_dim

    if model_type not in dino_models:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose from {list(dino_models.keys()) + ['medino_vitb16', 'dinov2uni_base']}"
        )

    model_args = {
        "pretrained": True,
        "num_classes": 0,
    }

    if "dynamic_img_size" in timm_models.create_model.__code__.co_varnames:
        model_args["dynamic_img_size"] = True

    model = timm_models.create_model(dino_models[model_type], **model_args)
    if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "num_features") and model.num_features is not None:
        feature_dim = model.num_features
    elif hasattr(model, "head") and hasattr(model.head, "in_features"):
        feature_dim = model.head.in_features
        model.head = nn.Identity()
    # print("Feature dim detected:", model)
    # else:
    # feature_dim = model  # .num_features

    return model, feature_dim
