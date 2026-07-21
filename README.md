# Understanding and Rectifying Safety Perception Distortion in VLMs

This repository contains the code for the paper "Understanding and Rectifying Safety Perception Distortion in VLMs".

![shiftdc-idea](/assets/idea.png)

[[Paper](https://arxiv.org/abs/2502.13095)] [[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/118667.png)]

## Installation

We use [vLLM](https://docs.vllm.ai/en/latest/) to run some of the inference. Since using vLLM to extract hidden states and do steering is a bit tricky, we use Hugging Face as the backend for these tasks.

```bash
git clone https://github.com/Renovamen/ShiftDC.git
cd ShiftDC

conda create --name mm python=3.12
conda activate mm

uv pip install vllm==0.17.1 --torch-backend=auto
uv pip install -r requirements.txt
```

## Data

These scripts help prepare data for extracting activations and for evaluation on the jailbreak task. See [DATA.md](DATA.md) for details on what each script does.

First, rename the [`.env.example`](.env.example) file to `.env` and add your OpenAI API key there. Then, run:

```bash
# Download and prepare MM-SafetyBench
python scripts/prepare_mmsb.py --data_dir data/mmsb

# Download and prepare the data for extracting safety shift
python scripts/prepare_steer.py --data_dir data/steer --mmsb_data_dir data/mmsb

# Download and prepare FigStep
python scripts/prepare_figstep.py --data_dir data/figstep
```

## Run

Our pipeline consists of the following steps:

1. Extract the safety-relevant shift (Equation 4):

    ```bash
    python run_safety_shift.py \
        -m llava-hf/llava-1.5-7b-hf \
        --input_dir data/steer \
        --output_dir {step1_dir} \
        --batch_size 32
    ```

2. Generate captions for images:

    ```bash
    python run_caption.py -m "llava-hf/llava-1.5-7b-hf" --data_dir data/mmsb --output_dir {step2_dir}
    ```

3. Extract activations for text-only and vision–language inputs to compute the modality-induced activation shift (Equation 5):

    ```bash
    python run_activation.py --mode tt -m "llava-hf/llava-1.5-7b-hf" --caption_jsonl {step2_dir}/caption.jsonl --data_dir data/mmsb --batch_size 32

    python run_activation.py --mode vl -m "llava-hf/llava-1.5-7b-hf" --caption_jsonl {step2_dir}/caption.jsonl --data_dir data/mmsb --batch_size 16
    ```

    Here `{step2_dir}` is the directory that contains `caption.jsonl` from step 2. These two commands write the following files into the same directory:

    - `tt_activations.npy`
    - `tt_index.jsonl`
    - `tt_meta.json`
    - `vl_activations.npy`
    - `vl_index.jsonl`
    - `vl_meta.json`

4. Calibrate the activation shift (Equations 6–7):

    ```bash
    python run_shiftdc.py \
        -m llava-hf/llava-1.5-7b-hf \
        --input_dir {step2_dir} \
        --data_dir data/mmsb \
        --safety_shift_npy {step1_output_path} \
        --layer_start 10 \
        --layer_end 31
    ```


## Acknowledgement

Our code reuses components from [refusal_direction](https://github.com/andyrdt/refusal_direction) and [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench). We appreciate their work.


## Citing this work

```text
@article{zou2025understanding,
  title={Understanding and Rectifying Safety Perception Distortion in VLMs},
  author={Zou, Xiaohan and Kang, Jian and Kesidis, George and Lin, Lu},
  journal={arXiv preprint arXiv:2502.13095},
  year={2025}
}
```
