# MLX→C++ Port — Phase 3v: Vision Model (Later)

## 목표
VisionModel을 완성하여 VLM 전체 추론을 지원한다.
**Priority: Phase 3 (text-only) 완료 후 진행**

---

## 3v.1 VisionModel

`vision.py`는 `Qwen3VLVisionModel` pass-through.
- Vision 모델 자체는 qwen3-vl 기존 구현 (ViT-based)
- PatchEmbed → Transformer blocks → feature extraction
- **구현 방법**: py_cpp의 vision 구현을 참조하거나, MLX에서 개별 구현

---

## 3v.2 VLM Composite (qwen3_5.py)

**Python**: `qwen3_5.py` Model(Qwen3VLModel)

```python
class Model(Qwen3VLModel):
    def get_input_embeddings(self, input_ids, pixel_values, image_grid_thw, ...):
        # 1. text embedding
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        # 2. vision embedding (if pixel_values)
        if pixel_values is not None:
            image_features = self.visual(pixel_values, grid_thw=image_grid_thw)
            # 3. merge
            inputs_embeds = self.merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids)
        return inputs_embeds

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        # masked_scatter: image_token_id (151655) 위치에 image_features 삽입
        image_mask = (input_ids == self.config.image_token_id)
        # expand + scatter
        ...
```

### sanitize (가중치 key 변환)

```python
def sanitize(weights, config):
    # key remapping:
    #   visual.patch_embed.proj.weight → transpose if Conv2d
    #   *.norm*.weight += 1.0 (RMSNorm offset)
    #   *.conv1d.weight → transpose if needed
```

---

## 3v.3 MRoPE for Vision

**get_rope_index**: 3D position ID 계산 (time, height, width)
- 이미지: (1, H, W) grid → position IDs 생성
- 비디오: (T, H, W) grid → temporal + spatial 통합
- 텍스트: 단순 offset으로 이어붙임
- 출력: position_ids [3, B, total_S]

---

## 상태
- [ ] 3v.1 VisionModel 구현 (또는 py_cpp 재사용)
- [ ] 3v.2 VLM Composite (get_input_embeddings, merge)
- [ ] 3v.3 MRoPE 3D position (get_rope_index)
- [ ] 3v.T Vision + Language 통합 테스트
