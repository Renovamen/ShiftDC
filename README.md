# Understanding and Rectifying Safety Perception Distortion in VLMs

This repository contains the code for the paper "Understanding and Rectifying Safety Perception Distortion in VLMs".

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
        --output_dir outputs/safety_shift \
        --batch_size 32
    ```

2. Generate captions for images:

    ```bash
    python run_caption.py -m "llava-hf/llava-1.5-7b-hf" --data_dir data/mmsb --output_dir outputs
    ```

3. Extract activations for text-only and vision–language inputs to compute the modality-induced activation shift (Equation 5):

    ```bash
    python run_tt_activation.py -m "llava-hf/llava-1.5-7b-hf" --caption_jsonl {step2_output_path} --data_dir data/mmsb --batch_size 32

    python run_vl_activation.py -m "llava-hf/llava-1.5-7b-hf" --caption_jsonl {step2_output_path} --data_dir data/mmsb --batch_size 16
    ```

4. Calibrate the activation shift (Equations 6–7):

    ```bash
    python run_shiftdc.py \
        -m llava-hf/llava-1.5-7b-hf \
        --caption_jsonl {step2_output_path} \
        --data_dir data/mmsb \
        --tt_activations_npy {step3_output_path} \
        --tt_index_jsonl {step3_output_path} \
        --vl_activations_npy {step3_output_path} \
        --vl_index_jsonl {step3_output_path} \
        --safety_shift_npy {step1_output_path} \
        --safety_shift_meta {step1_output_path} \
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
