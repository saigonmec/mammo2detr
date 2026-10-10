# Phân tích code `zdetr` và thiết kế lại trong `src/`

Tài liệu này tổng hợp kết quả phân tích toàn bộ ~12k dòng của `zdetr/` (DETR đa nhiệm cho
mammography), giải thích **vì sao mô hình cũ hội tụ kém**, và mô tả kiến trúc mới trong `src/`.
Mọi lỗi dưới đây đều được kiểm chứng bằng cách chạy trực tiếp các hàm cũ (xem mục 5).

---

## 1. Tổng quan luồng cũ

```
detr_train.py ─► load_detr_metadata (detr_data_pre.py) ─► get_detr_dataloaders (detr_data.py)
             ─► get_detr_model (detr_model.py: backbone ─► proj 1x1 ─► MultiScaleSpatialAttention
                 ─► EfficientDETRDecoder 1 layer / 1 head / 3 query ─► bbox_head sigmoid + obj_head)
             ─► train_detr_model (detr_engines.py) với DETRCriterion (detr_train_utils.py)
```

Phần classification/patch/MIL (`train_based.py`, `train_patch.py`, `mil_py.py`, …) là code của
dự án mammoSGM được chép sang, import `src.…` nên **không chạy được** trong repo này
(`zdetr/trainer/train_based.py:9-14`, `zdetr/data/detr_data.py:154,159`, `zdetr/trainer/detr_train.py:97`).

---

## 2. Các lỗi làm mô hình không hội tụ (xếp theo mức độ nghiêm trọng)

### 2.1. Ba quy ước toạ độ box dùng lẫn lộn (lỗi nặng nhất)

| Nơi | Quy ước được *giả định* | Dòng |
|---|---|---|
| Dataset tạo target | COCO `[x, y, w, h]` chuẩn hoá | `zdetr/data/detr_data.py:125-130` |
| Hungarian matcher (`box_iou`) | `[x, y, w, h]` (đúng với data) | `zdetr/utils/detr_train_utils.py:33-35` |
| **GIoULoss** | `[x1, y1, x2, y2]` | `zdetr/utils/bbox_loss.py:40-57` |
| **Reference point của decoder** | `[x1, y1, x2, y2]` (tâm = `(b0+b2)/2`) | `zdetr/models/detr_model.py:136-138, 193-195` |
| **Metric mAP/recall (`box_iou_batch`)** | `[cx, cy, w, h]` | `zdetr/utils/detr_metrics.py:24-35` |
| `rescale_bbox` khi inference | `[cx, cy, w, h]` | `zdetr/trainer/detr_run.py:75-84` |
| `generalized_box_iou` (common utils) | công thức union sai | `zdetr/utils/detr_common_utils.py:118` |

Hậu quả:
* **GIoU loss tính sai hình học** → gradient sai cho nhánh box. Ví dụ đã chạy: box GT
  `[0.1,0.2,0.3,0.4]`, pred lệch nhẹ: GIoU thật = 0.790 nhưng `GIoULoss` cũ ngầm tính 0.810;
  với các cặp box khác kích thước thì sai lệch còn lớn hơn nhiều.
* Reference point sai tâm: GT `[0.1,0.2,0.3,0.4]` có tâm thật `(0.25, 0.40)` nhưng decoder dùng `(0.20, 0.30)`.
* **mAP / IoU / recall báo cáo sai**: với pred `[0.35,0.55,0.1,0.1]` và GT trên, IoU thật = 0.0196
  nhưng metric cũ trả về 0.0. Vì metric sai nên không thể tin đường cong huấn luyện để tinh chỉnh.
* Matcher dùng cost "giou" nhưng thực chất là `-IoU` (phẳng khi box không chồng lấn).

### 2.2. Decoder về mặt cấu trúc không thể định vị tổn thương

`EfficientDeformableAttention` (`zdetr/models/detr_model_attn.py:198-213`) tính offset lấy mẫu từ
query, nhưng query = `query_embed` (tĩnh) → self-attention giữa các query tĩnh → vẫn tĩnh.
**Vị trí lấy mẫu vì thế không phụ thuộc vào ảnh.** Đã kiểm chứng: với hai ảnh ngẫu nhiên khác
nhau, `sampling_locations` giống hệt nhau (max |diff| = 0.0). Với 1 layer, 1 head, 4 điểm, mỗi
query chỉ "nhìn" 4 vị trí cố định của feature map 14×14 rồi hồi quy box bằng `Sigmoid` tuyệt đối.
Không có refinement, không có aux loss (`aux_outputs` luôn rỗng vì chỉ có 1 layer,
`detr_model.py:225-228`).

### 2.3. Denoising queries rò rỉ thông tin và không được giám sát

* DN queries (tạo từ GT + nhiễu) được nối vào cùng chuỗi với matching queries
  (`detr_model.py:127-142`) và đi qua self-attention **không có attention mask**
  (`detr_model.py:151-153`) → lúc train, matching queries "nhìn thấy" box GT bị nhiễu, lúc test
  thì không → lệch phân phối train/test.
* `dn_outputs` được trả về nhưng `DETRCriterion` không bao giờ dùng (`detr_train_utils.py:70-118`),
  tức là tốn tính toán mà không có tác dụng của DN-DETR.
* Nhiễu `randn * 0.2` cộng thẳng vào `[x,y,w,h]` (không tỉ lệ với kích thước box).

### 2.4. Augmentation khác nhau cho ảnh dương/âm → rò rỉ nhãn

Ảnh có box đi qua `get_m2_positive_augmentations` còn ảnh không box đi qua
`get_m2_negative_augmentations` (`zdetr/data/detr_augment.py:9-95` và `:98-201`, chọn ở
`zdetr/data/detr_data.py:85-108`). Hai pipeline khác nhau ở: `CropAndPad` (chỉ negative),
`RandomSizedBBoxSafeCrop` (chỉ positive), biên độ `HueSaturationValue`/`RGBShift`/blur khác nhau.
Một CNN hoàn toàn có thể học "dấu vết augmentation" thay vì bệnh lý → accuracy train/val ảo,
không khái quát.

### 2.5. Các vấn đề huấn luyện/đánh giá khác

* **Chọn model trên tập test**: `test_df` được dùng làm validation để early-stop và chọn checkpoint
  (`detr_train.py:66-75`, `detr_engines.py:411-426`) → số liệu test lạc quan.
* Metric detection chỉ gom ảnh **có box** (`detr_test_utils.py:63-67`, `detr_engines.py:292-296`):
  false positive trên ảnh âm không bao giờ bị đếm → mAP bị thổi phồng, không có FROC.
* `recall@0.25` thực chất là recall ở ngưỡng thấp nhất (có 3 query thì gần như "có box nào trúng không").
* `precision/recall/F1` dùng `average="weighted"` (bị lớp đa số chi phối); specificity được
  **parse từ chuỗi text** của `classification_report` (`detr_test_utils.py:171-182`).
* Không gradient clipping (DETR chuẩn dùng 0.1); AMP dùng API `torch.cuda.amp` cũ; scheduler
  `plateau.step(val_loss / val_total)` chia hai lần (`detr_engines.py:407`).
* Train-IoU tính giữa query 0 và GT 0 không qua matching (`detr_engines.py:232-239`) → vô nghĩa.
* Lưu checkpoint bằng cách **parse metric từ tên file** (`detr_engines.py:429-525`), rất dễ vỡ.
* Chuẩn hoá ảnh `mean=std=0.5` (`detr_augment.py:220`) trong khi backbone pretrained ImageNet.
* Ảnh lỗi → trả về tensor 0 im lặng (`detr_data.py:63-71`).

### 2.6. Backbone wrappers sai với ViT có register tokens

* `TimmFeatureWrapper` (`detr_wrapper.py:55-74`) giả định không có CLS token và lưới vuông.
* `DinoFeatureWrapper` (`detr_wrapper.py:166-182`) bỏ đúng **1** token đầu; các model
  `*_reg4_dinov2` có 5 prefix tokens (CLS + 4 register) → reshape sai hoặc lỗi.
* `unfreeze_last_blocks` (`detr_model.py:402-469`) để `patch_embed/pos_embed/cls_token` vẫn
  trainable; với ConvNeXt lại mở khoá các block cuối của **mọi** stage.

### 2.7. Chất lượng mã

`_generate_eval_visualizations` (`detr_run.py:689-922`) dùng biến chưa định nghĩa (`img_path`,
`preprocess`, `out_dir`, …) – code chết; `get_arg_or_config` dùng `inspect` để coi chuỗi là tên biến
(`common.py:27-45`); 3 bản `unfreeze_last_blocks`, 2 bản `get_weighted_sampler_detr`; hằng
`[0.5]*3` lặp ở 5 chỗ; nhiều `[DEBUG] print` trong đường chạy chính; lưu cả object model bằng pickle
(`_full.pth`) nên phụ thuộc phiên bản code.

---

## 3. Thiết kế mới trong `src/`

### 3.1. Nguyên tắc
1. **Một quy ước box duy nhất**: model/target/loss dùng `cxcywh` chuẩn hoá; IoU/GIoU tính trên
   `xyxy`; CSV là pixel COCO `xywh` và chỉ được chuyển đổi **một lần** trong dataset
   (`src/utils/box_ops.py`, `src/data/dataset.py`). Các hàm IoU/GIoU được test khớp torchvision.
2. **Kiến trúc DETR hiện đại, đúng chuẩn** (`src/models/`):
   * Backbone bất kỳ từ `timm` qua `features_only` (CNN, ConvNeXt, EfficientNet, Swin, MaxViT…)
     hoặc `forward_intermediates` cho ViT thuần (DINOv2/v3, EVA02) có xử lý đúng CLS/register
     tokens và kích thước ảnh động; pyramid kiểu ViTDet cho ViT đa tỉ lệ (`backbones.py`).
   * Transformer `standard` (dense, 1 level) hoặc `deformable` (multi-scale, thuần PyTorch, không
     cần CUDA op) (`attention.py`, `transformer.py`).
   * Query = content embedding + **anchor box 4-D học được** (DAB-DETR), positional query tính từ
     sine(anchor); **iterative box refinement** qua từng layer; dự đoán box dạng
     `sigmoid(delta + inverse_sigmoid(ref))` thay vì sigmoid tuyệt đối.
   * **Cross-attention có điều kiện** (Conditional-DETR): tách phần nội dung và phần vị trí; hai
     phép chiếu vị trí được khởi tạo bằng cùng một ma trận hoán vị nên attention đã tập trung quanh
     anchor ngay từ bước đầu (xem mục 5.3).
   * **Contrastive denoising** (DN-DETR/DINO) với attention mask block-diagonal đúng, nhóm dương/âm,
     loss có assignment cố định (`denoising.py`, `losses/criterion.py`).
   * Đa nhiệm thật sự: head phân loại ảnh (`query` token / `pool` / `hybrid`) cho **nhiều target**
     (`data.image_labels`), head detection với số lớp tổn thương tuỳ ý (`heads.py`, `detr.py`).
3. **Loss chuẩn**: Hungarian (focal cost + L1 + GIoU), sigmoid focal cho detection, aux loss
   mọi layer, DN loss, CE/Focal/LDAM có class-weight cho image-level (`src/losses/`).
4. **Dữ liệu**: metadata một chỗ, mã hoá nhãn xác định, **tách val theo bệnh nhân** nếu CSV chưa có,
   một pipeline augmentation bbox-aware chung cho mọi ảnh, letterbox + padding mask hoặc stretch,
   chuẩn hoá ImageNet, sampler cân bằng (hỗ trợ DDP) (`src/data/`).
5. **Huấn luyện**: AMP (`torch.amp`), grad-clip 0.1, accumulate, EMA, cosine/step/plateau có
   warmup, param-group riêng cho backbone, resume, top-k checkpoint theo metric cấu hình, early
   stopping, log CSV/JSONL (+TensorBoard/W&B), DDP qua `torchrun` (`src/engine/trainer.py`).
6. **Đánh giá đúng cho mammography**: AP@IoU (COCO 101 điểm), mAP@[.5:.95], **FROC** (sensitivity
   tại FPPI 0.125…4, tính trên **cả ảnh âm**, IoU hoặc center-match), localization recall@k,
   AUC/sens/spec/Youden cho image-level, AUC từ max score detection (`src/engine/metrics_*.py`).
7. **Kiểm thử**: `src/tests/` (pytest) gồm box ops, matcher/criterion, DN mask, model forward
   (standard/deformable/ViT), metrics bằng tay, dataset + loader, và một bài end-to-end
   train → evaluate → predict → resume trên dữ liệu tổng hợp.

### 3.2. Ánh xạ cũ → mới

| Cũ (`zdetr`) | Mới (`src`) |
|---|---|
| `detr_train.py` / `detr_test.py` / `detr_run.py` | `python -m src.train` / `src.evaluate` / `src.predict` |
| `config/config.yaml` (phẳng) | `src/configs/*.yaml` (lồng, kiểm tra khoá nghiêm ngặt, override `--set a.b=c`) |
| `detr_data_pre.py` + `detr_data_utils.py` + `dataloader.py` | `src/data/metadata.py` |
| `detr_data.py` + `detr_augment.py` | `src/data/dataset.py`, `transforms.py`, `builder.py` |
| `backbone.py` + `detr_wrapper.py` | `src/models/backbones.py` (alias giữ nguyên tên cũ) |
| `detr_model.py` + `detr_model_attn.py` | `src/models/detr.py`, `transformer.py`, `attention.py`, `denoising.py`, `heads.py` |
| `detr_train_utils.py` + `bbox_loss.py` + `loss.py` | `src/losses/` |
| `detr_metrics.py` + `detr_test_utils.py` | `src/engine/metrics_det.py`, `metrics_cls.py` |
| `detr_engines.py` | `src/engine/trainer.py`, `evaluator.py`, `postprocess.py` |
| `detr_viz_utils.py` + `detr_gradcam_utils.py` | `src/viz/draw.py` + attention map trong `src.predict --attention` |

---

## 4. Đề xuất tiếp theo (chưa cài đặt, theo thứ tự ưu tiên)

1. **Độ phân giải / tỉ lệ**: mammogram 2–4k px, tổn thương nhỏ (vi vôi hoá) chiếm < 1% cạnh ảnh.
   Nên dùng `resize_mode: letterbox` với `img_size` 1024×768 trở lên và `deformable` 3 level
   (8/16/32); chạy `python -m src.tools.validate_data` để xem phân bố kích thước box.
2. **Two-stage / mixed query selection** (Deformable-DETR two-stage, DINO): lấy top-k proposal từ
   encoder làm anchor khởi tạo thay vì anchor học được, giúp hội tụ nhanh hơn trên lesion thưa.
3. **Khởi tạo backbone từ checkpoint phân loại đã có** (các model MIL/patch của bạn): thêm tuỳ chọn
   `model.backbone.init_from` nạp `state_dict` của backbone sau khi đổi tên khoá; backbone đã "biết"
   mammogram sẽ giảm đáng kể số epoch.
4. **Multi-view**: ghép CC/MLO cùng bên thành 2 ảnh một mẫu với cross-view attention ở encoder
   (hoặc đơn giản là late-fusion ở head phân loại).
5. **Test-time augmentation** (lật ngang) và **calibration** (temperature scaling trên val) cho
   image-level score; **SWA/EMA** đã có sẵn.
6. **Lịch huấn luyện**: 50–100 epoch, lr 1e-4 (backbone ×0.1), cosine, batch 8–16, EMA bật; theo dõi
   `det/froc_mean` và `cls/<target>/auc` trên val, chọn checkpoint theo metric mục tiêu lâm sàng.
7. **Mất cân bằng**: ảnh âm ≫ dương → `balanced_sampler: true`, `class_weights: balanced`
   (hoặc `focal`); với detection, focal loss + nhiều ảnh âm là tự nhiên (không cần bỏ ảnh âm).

---

## 5. Bằng chứng kiểm chứng (đã chạy trong môi trường này)

### 5.1. Lỗi quy ước box (hàm cũ, dữ liệu `xywh` chuẩn hoá)
```
TRUE   IoU=0.7964  GIoU=0.7905   (torchvision, xyxy)
GIoULoss cũ (coi xywh là xyxy)  -> implied GIoU = 0.8100           (sai)
generalized_box_iou cũ          -> 0.8710                           (sai công thức union)
decoder ref point (b0+b2)/2,(b1+b3)/2 = (0.200,0.300) ; tâm thật = (0.250,0.400)
Box khác kích thước: IoU thật = 0.0196 ; detr_metrics.box_iou_batch cũ = 0.0000
```

### 5.2. Decoder cũ lấy mẫu tại vị trí cố định
```
sampling locations identical across two different images: True | max abs diff: 0.0
sampling locations identical across batch items: True | num sampled locations per query: 4
```

### 5.3. Model mới hội tụ trên bài toán tổng hợp (CPU, resnet18 không pretrained, 96–128 px)
* Decoder deformable, 300 bước: L1 box chính 0.47 → 0.03, GIoU loss 1.18 → 0.11; DN box loss
  0.28 → 0.05 và giảm dần qua từng layer (0.099 / 0.067 / 0.054) – đúng hành vi iterative refinement.
* Decoder standard (cùng bài toán, stride 8): với cross-attention cộng tính kiểu DETR gốc hoặc
  T=10000, DN loss gần như đứng yên ở mức nhiễu (DN query không dùng được thông tin anchor vì
  phép chiếu vị trí chưa "khớp"). Sau khi thêm conditional cross-attention, khởi tạo hoán vị cho
  `q_pos/k_pos` và nhiệt độ sine T=20: L1 chính 0.40 → 0.053, DN 0.277 → 0.150 – học được nhưng
  vẫn chậm hơn deformable (0.030 / 0.054). **Kết luận**: mặc định dùng `deformable` + 3 level
  (8/16/32); nếu dùng `standard` thì chọn stride 16 hoặc 8 và `pos_temperature: 20`.
* Toàn bộ pipeline đã qua 30 test pytest (unit + end-to-end train → evaluate → predict → resume).

### 5.4. Huấn luyện end-to-end qua CLI trên dữ liệu tổng hợp (CPU, 40 epoch ≈ 480 bước, 112×96 px)
Dữ liệu: 160 ảnh (`src/tools/make_synthetic_dataset.py`), 50% dương có 1–3 "tổn thương"; model
`debug_cpu.yaml` + `deformable` 3 level (d_model 64, 1 encoder / 2 decoder layer, resnet18 không pretrained).

| epoch | val loss | val cls AUC | val FROC-mean | val img_recall@1 | val AP50 |
|---|---|---|---|---|---|
| 3  | 13.21 | 0.52 | 0.02 | 0.00 | 0.00 |
| 11 | 11.46 | 1.00 | 0.07 | 0.13 | 0.00 |
| 23 | 9.01  | 1.00 | 0.13 | 0.19 | 0.00 |
| 31 | 7.72  | 1.00 | 0.39 | 0.63 | 0.04 |
| 39 | 6.90  | 1.00 | 0.43 | 0.63 | 0.16 |

Test (checkpoint tốt nhất theo `det/froc_mean`): cls AUC 1.00, acc 0.97, AP25 0.38, FROC-mean 0.42,
img_recall@1 0.75, box_img_auc 1.00, 0.06 FP/ảnh ở ngưỡng 0.5. Ảnh âm nhận p(cancer) ≈ 0.007.
(AP50 thấp vì box ở ảnh 112 px chỉ vài pixel; mục đích ở đây là kiểm chứng *xu hướng* và tính đúng
của toàn bộ chuỗi data → loss → metric, không phải chất lượng tuyệt đối.)

Lưu ý thực tế rút ra: với run ngắn, EMA decay 0.9998 (cửa sổ ~5000 bước) làm model đánh giá tụt
hậu so với model đang train; `Trainer` nay tự hạ decay sao cho cửa sổ EMA không quá nửa run.
  Lưu ý khi override list trên shell (zsh/bash): đặt trong nháy, ví dụ `--set 'data.img_size=[640,512]'`.
* Image-level head đạt accuracy 1.0 trên bài overfit 16 ảnh sau 150 bước; max score detection trên
  ảnh âm 0.012 so với ≥ 0.68 trên ảnh dương.
