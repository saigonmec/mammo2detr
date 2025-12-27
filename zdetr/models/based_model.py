from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    # get_fastervit_backbone,
    get_dino_backbone,
    get_mamba_backbone,
)
from .head import get_linear_head
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def get_based_model(model_type="resnet50", num_classes=2):
    if model_type in ["resnet34", "resnet50", "resnet101", "resnext50", "resnet152"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        # Replace the head with a linear classifier
        model = backbone
        model.fc = get_linear_head(feature_dim, num_classes)
    elif model_type in ["mamba_t", "mamba_s"]:
        model, feature_dim = get_mamba_backbone(model_type, num_classes=num_classes)
        # model.model.head = nn.Linear(feature_dim, num_classes)
        # print("Using Mamba_T backbone with custom head.", model)
    elif model_type in [
        "resnest50",
        "resnest101",
        "resnest50s2",
        "regnety",
        "convnextv2base",
        "convnextv2_tiny",
        "efficientnetv2",
        "efficientnetv2s",
        "maxvit_tiny",
        "maxvit_small",
        "maxvit_base",  # thêm maxvit_base
        "eva02_small",  # sửa lại đúng tên model_type
        "eva02_base",  # thêm eva02_base
        "vit_small",  # thêm model mới
        "swinv2_tiny",
        "swinv2_base",
        "swinv2_small",
        "mambaout_tiny",
    ]:
        model, feature_dim = get_timm_backbone(model_type)
        # Replace the head with a linear classifier for all timm backbones
        if hasattr(model, "fc"):
            model.fc = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "head") and hasattr(model.head, "fc"):
            model.head.fc = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "head"):
            model.head = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "classifier"):
            model.classifier = get_linear_head(feature_dim, num_classes)
        else:
            raise ValueError("Unknown head structure for timm backbone")
        # elif model_type == "fastervit":
        #    model, feature_dim = get_fastervit_backbone()
        # model.head = get_linear_head(feature_dim, num_classes)
    elif model_type in [
        "dinov2_small",
        "dinov2_base",
        "dinov2_small_reg",
        "dinov2_base_reg",
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_vit16small",
        "dinov3_vit16smallplus",
        "dinov3_vit16base",
        "dinov3_vit16large",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
        "medino_vitb16",
        "dinov2uni_base",
    ]:
        transformer, feature_dim = get_dino_backbone(model_type)

        model = DinoVisionTransformerClassifier(transformer, feature_dim, num_classes)
        # Freeze DINO backbone - only train the classification head
        for param in model.transformer.parameters():
            param.requires_grad = False
    else:
        raise ValueError("Unsupported model_type for base model")
    return model


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        transformer,
        feature_dim,
        num_classes,
        hidden_dim=512,
        dropout_p=0.3,
    ):
        super().__init__()
        self.transformer = transformer
        self.feature_dim = feature_dim
        # self.classifier = get_linear_head(feature_dim, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x
