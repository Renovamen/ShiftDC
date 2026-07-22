# Data Preparation Scripts

## `prepare_mmsb.py`

Downloads all [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench) policies and variants (`SD`, `TYPO`, and `SD_TYPO`). It saves images under `<data_dir>/images` and writes `<data_dir>/data.json` items in this format:

```json
{
  "id": 0,
  "redteam_query": "<original query>",
  "jailbreak_query": "<query adapted for the jailbreak image>",
  "policy": "<policy>",
  "image_path": "images/<policy>/<variant>/<sample_id>.jpg",
  "image_type": "<variant>"
}
```

## `prepare_figstep.py`

Downloads the [FigStep](https://github.com/CryptoAILab/FigStep) CSV and screenshots and writes `<data_dir>/data.json` items in this format:

```json
{
  "id": 0,
  "redteam_query": "<original query>",
  "jailbreak_query": "<query adapted for the jailbreak image>",
  "typography": "<typography embedded in the image>",
  "policy": "<policy>",
  "image_path": "images/<image_name>.png"
}
```

## `prepare_steer.py`

1. Downloads [COCO](https://cocodataset.org/#download) training images (the image source for [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)) and samples 160 image-instruction pairs from LLaVA-Instruct-80K (the subset of LLaVA-Instruct-150K).
2. Samples 160 MM-SafetyBench examples evenly across policies, including their `SD`, `SD_TYPO`, and `TYPO` variants.
3. Generates a keyword for each sampled COCO image using GPT-4o.
4. Embeds each keyword into images to create `TYPO` and `SD_TYPO` variants for the COCO images.
5. Generates image captions for the COCO and MM-SafetyBench images.
6. Writes `llava/data.json` and `mmsb/data.json`.

If any sample's `keyword` or `caption` is `123` in the output JSON files, it is a placeholder for an error (likely a connection issue), and the sample must be rerun.
