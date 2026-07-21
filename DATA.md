# Data Preparation Scripts

## `scripts/prepare_mmsb.py`

Downloads all MM-SafetyBench policies and splits. Saves images as `<data_dir>/images/<policy>/<image_type>/<sample_id>.jpg`. Writes `<data_dir>/data.json` items in this format:

```json
{
  "id": 0,
  "jailbreak_query": "...",
  "redteam_query": "...",
  "policy": "...",
  "image_path": "images/<policy>/<image_type>/<sample_id>.jpg",
  "image_type": "SD"
}
```

## `scripts/prepare_figstep.py`

Downloads the FigStep SafeBench CSV and screenshots and writes `<data_dir>/data.json` items in this format:

```json
{
  "id": 0,
  "jailbreak_query": "...",
  "redteam_query": "...",
  "typography": "...",
  "policy": "...",
  "image_path": "images/query_<dataset>_<category_id>_<task_id>_6.png"
}
```

## `scripts/prepare_steer.py`

1. Downloads COCO training images (the image source for LLaVA-Instruct-150K) and samples 160 image-instruction pairs from LLaVA-Instruct-150K.
2. Samples 160 MM-SafetyBench examples evenly across policies, including their `SD`, `SD_TYPO`, and `TYPO` variants.
3. Generates a keyword for each sampled COCO image using GPT-4o.
4. Embeds each keyword into an image to create `TYPO` and `SD_TYPO` variants for the COCO images.
5. Generates image captions for the COCO and MM-SafetyBench images.
6. Writes `llava/data.json` and `mmsb/data.json`.
