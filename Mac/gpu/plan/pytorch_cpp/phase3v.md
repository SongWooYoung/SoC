# Phase 3v: qwen3_5.h — Vision Model + VLM

## 목표
Phase 3(텍스트 모델)이 완료된 후, `modeling_qwen3_5.py`의 **Vision 모델** 및 **VLM 결합** 부분을 C++/Metal로 구현한다.

## 전제 조건
- Phase 3 완료 (`Qwen3_5ForCausalLM` 텍스트 생성 동작 확인)
- VLM은 텍스트 backbone을 그대로 사용하므로, Phase 3의 코드 위에 Vision 부분을 추가하는 구조

## 원본
- `modeling_qwen3_5.py` 내 Vision 관련 클래스들 (이미 인라인되어 있음)

## 구현 순서 (bottom-up)

### 3v-a. Qwen3_5VisionRotaryEmbedding
- 2D rotary embedding (height, width)
- `theta=10000.0`, `dim = head_dim // 2`

### 3v-b. Qwen3_5VisionPatchEmbed
- `nn.Conv3d` (temporal_patch_size × patch_size × patch_size)
- 입력: raw pixel values → patch embeddings

### 3v-c. Qwen3_5VisionAttention
- QKV를 하나의 Linear으로 생성 후 split
- `apply_rotary_pos_emb_vision()` — 2D rotary
- Flash Attention 지원 (cu_seqlens 기반 variable-length)
- fallback: chunk별 개별 attention

### 3v-d. Qwen3_5VisionMLP
- `linear_fc1` → activation → `linear_fc2`
- bias=True (텍스트 MLP과 다름)

### 3v-e. Qwen3_5VisionBlock
- LayerNorm → VisionAttention → residual
- LayerNorm → VisionMLP → residual

### 3v-f. Qwen3_5VisionPatchMerger
- spatial merge (spatial_merge_size² 패치를 하나로)
- norm → linear_fc1 → GELU → linear_fc2

### 3v-g. Qwen3_5VisionModel
- PatchEmbed (3v-b)
- positional embedding (learned `nn.Embedding` + bilinear interpolation)
- rotary pos emb (3v-a)
- VisionBlock[] × depth (3v-e)
- PatchMerger (3v-f)
- `rot_pos_emb()` — 2D position 계산
- `fast_pos_embed_interpolate()` — bilinear interpolation

### 3v-h. Qwen3_5Model (VLM 결합)
- `Qwen3_5VisionModel` + `Qwen3_5TextModel` 결합
- `get_image_features()` — pixel_values → vision embedding
- `get_video_features()` — video pixel_values → vision embedding
- `get_placeholder_mask()` — image/video token 위치 식별
- `compute_3d_position_ids()` — MRoPE 3D position 계산
- `get_rope_index()` — vision token과 text token의 position 매핑
- `masked_scatter` — vision embedding을 text embedding에 삽입

### 3v-i. Qwen3_5ForConditionalGeneration
- `Qwen3_5Model` (3v-h) + `lm_head`
- image/video 입력 처리
- `prepare_inputs_for_generation()` — 첫 iteration에만 pixel_values 전달
- `_expand_inputs_for_generation()` — beam search 시 vision tensor 복제

## 결과물
- `models/qwen3_5/qwen3_5.h` 에 Vision 관련 클래스 추가

## 상태
- [ ] 3v-a. VisionRotaryEmbedding
- [ ] 3v-b. VisionPatchEmbed
- [ ] 3v-c. VisionAttention
- [ ] 3v-d. VisionMLP
- [ ] 3v-e. VisionBlock
- [ ] 3v-f. VisionPatchMerger
- [ ] 3v-g. VisionModel
- [ ] 3v-h. Qwen3_5Model (VLM 결합)
- [ ] 3v-i. ForConditionalGeneration
