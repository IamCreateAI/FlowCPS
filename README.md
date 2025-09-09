<h1 align="center"> Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2509.05952'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
</div>

## Changelog

2025/09/09: Please check [this commit](https://github.com/IamCreateAI/FlowCPS/commit/814c35979d9ee3563178373bccca0605ded6ffa9) for the implementation of Flow-CPS. We also provide the code to generate Figure 2 in [this notebook](https://github.com/IamCreateAI/FlowCPS/blob/main/scripts/analysis/sde_noise_level.ipynb)


## 🚀 Quick Started
### 1. Environment Set Up
Clone this repository and install packages.
```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```

### 2. Model Download
To avoid redundant downloads and potential storage waste during multi-GPU training, please pre-download the required models in advance.

**Models**
* **SD3.5**: `stabilityai/stable-diffusion-3.5-medium`
* **Flux**: `black-forest-labs/FLUX.1-dev`

**Reward Models**
* **PickScore**:
  * `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  * `yuvalkirstain/PickScore_v1`
* **CLIPScore**: `openai/clip-vit-large-patch14`
* **Aesthetic Score**: `openai/clip-vit-large-patch14`


### 3. Reward Preparation
The steps above only install the current repository. Since each reward model may rely on different versions, combining them in one Conda environment can cause version conflicts. To avoid this, we adopt a remote server setup inspired by ddpo-pytorch. You only need to install the specific reward model you plan to use.

#### GenEval
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### OCR
Please install paddle-ocr:
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```
Then, pre-download the model using the Python command line:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```

#### Pickscore
PickScore requires no additional installation.

#### DeQA
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### UnifiedReward
Since `sglang` may conflict with other environments, we recommend creating a new conda environment.
```bash
conda create -n sglang python=3.10.16
conda activate sglang
pip install "sglang[all]"
```
We use sglang to deploy the reward service. After installing sglang, please run the following command to launch UnifiedReward:
```bash
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
```
#### ImageReward
Please install imagereward:
```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

### 4. Start Training
If the GPU memory is insufficient, you can use DeepSpeed Zero2 or Zero3.

```bash
# zero2
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml
# zero3
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml
```

#### GRPO

- Single-node training:
```bash
# sd3
bash scripts/single_node/grpo.sh
# flux
bash scripts/single_node/grpo_flux.sh
```
- Multi-node training for SD3:
```bash
# Master node
bash scripts/multi_node/sd3/main.sh
# Other nodes
bash scripts/multi_node/sd3/main1.sh
bash scripts/multi_node/sd3/main2.sh
bash scripts/multi_node/sd3/main3.sh
```
- Multi-node training for FLUX.1-dev:
```bash
# Master node
bash scripts/multi_node/flux/main.sh
# Other nodes
bash scripts/multi_node/flux/main1.sh
bash scripts/multi_node/flux/main2.sh
bash scripts/multi_node/flux/main3.sh
```
- Multi-node training for FLUX.1-Kontext-dev:

Please first download [generated\_images.zip](https://huggingface.co/datasets/jieliu/counting_edit/blob/main/generated_images.zip) and extract it into the `counting_edit` directory. You can also use the scripts in the `counting_edit` directory to generate the data yourself.

Please install `diffusers` from the main branch to support `FLUX.1-Kontext-dev`:
```bash
pip install git+https://github.com/huggingface/diffusers.git
```
After upgrading Diffusers, some packages such as PEFT may also need to be upgraded. If you encounter any errors, please upgrade them according to the error messages.
Then, run the scripts:
```bash
# Master node
bash scripts/multi_node/flux_kontext/main.sh
# Other nodes
bash scripts/multi_node/flux_kontext/main1.sh
bash scripts/multi_node/flux_kontext/main2.sh
bash scripts/multi_node/flux_kontext/main3.sh
```


- Multi-node training for Qwen-Image:

In the implementation of Qwen-Image, we have unified Flow-GRPO and Flow-GRPO-Fast. You can control the size of the SDE window with `config.sample.sde_window_size`, and adjust the position of the window with `config.sample.sde_window_range`.

Please install `diffusers` from the main branch to support `Qwen-Image`:
```bash
pip install git+https://github.com/huggingface/diffusers.git
```
Then run the scripts:
```bash
# Master node
bash scripts/multi_node/qwenimage/main.sh 0
# Other nodes
bash scripts/multi_node/qwenimage/main.sh 1
bash scripts/multi_node/qwenimage/main.sh 2
bash scripts/multi_node/qwenimage/main.sh 3
```
Using the provided configuration, the resulting reward curve of Qwen-Image on the test set is shown below.

<p align="center">
  <img src="flow_grpo/assets/flow_grpo_fast_qwenimage.png" alt="Flow-GRPO-Fast Illustration" width=350"/>
</p>


- Multi-node training for Qwen-Image-Edit:

Same as Flux Kontext, please first download [generated\_images.zip](https://huggingface.co/datasets/jieliu/counting_edit/blob/main/generated_images.zip) and extract it into the `counting_edit` directory. You can also use the scripts in the `counting_edit` directory to generate the data yourself.

Please install `diffusers` from the main branch to support `Qwen-Image-Edit`:
```bash
pip install git+https://github.com/huggingface/diffusers.git
```
Then run the scripts:
```bash
# Master node
bash scripts/multi_node/qwenimage_edit/main.sh 0
# Other nodes
bash scripts/multi_node/qwenimage_edit/main.sh 1
bash scripts/multi_node/qwenimage_edit/main.sh 2
bash scripts/multi_node/qwenimage_edit/main.sh 3
```

Using the provided configuration, the resulting reward curve of Qwen-Image-Edit on the test set is shown below.

<p align="center">
  <img src="flow_grpo/assets/qwenimageedit_epoch.png" alt="Flow-GRPO-Fast Illustration" width="350"/>
  <img src="flow_grpo/assets/qwenimageedit_time.png" alt="Flow-GRPO-Fast Illustration" width="350"/> 
</p>


#### DPO / OnlineDPO / SFT / OnlineSFT
 Single-node training:
```bash
bash scripts/single_node/dpo.sh
bash scripts/single_node/sft.sh
```
Multi-node training:

Please update the entry Python script and config file names in the `scripts/multi_node` bash file.

## How to Support Other Models

To integrate a new model into this framework, please follow the steps below:

**1. Add the following files adapted for your model:**

* `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`:
  This file is adapted from [pipeline\_stable\_diffusion\_3.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py). You can refer to diffusers for your model.

* `scripts/train_sd3.py`:
  This script is based on [train\_dreambooth\_lora\_sd3.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sd3.py) from the DreamBooth examples.

* `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`:
  This file handles SDE sampling. In most cases, you don't need to modify it. However, if your definitions of `dt` or `velocity` differ in sign or convention, please adjust accordingly.

**2. Verify SDE sampling:**
Set `noise_level = 0` in [sde\_demo.py](https://github.com/yifan123/flow_grpo/tree/main/scripts/demo/sd3_sde_demo.py) to check whether the generated images look normal. This helps verify that your SDE implementation is correct.

**3. Ensure on-policy consistency:**
Set [`config.sample.num_batches_per_epoch = 1`](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py#L120) and [`config.train.gradient_accumulation_steps = 1`](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py#L125C5-L125C47) to enforce a purely on-policy setup, where the model collecting samples is identical to the one being trained.
Under this setting, the [ratio](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3.py#L886) should remain exactly 1. If it's not, please check whether the sampling and training code paths differ—for example, through use of `torch.compile` or other model wrappers—and make sure both share the same logic.

**4. Tune reward behavior:**
Start with `config.train.beta = 0` to observe if the reward increases during training. You may also need to adjust the noise level [here](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py#L47) based on your model. Other hyperparameters are generally model-agnostic and can be kept as default.


## 🏁 Multi Reward Training
For multi-reward settings, you can pass in a dictionary where each key is a reward name and the corresponding value is its weight.
For example:

```python
{
    "pickscore": 0.5,
    "ocr": 0.2,
    "aesthetic": 0.3
}
```

This means the final reward is a weighted sum of the individual rewards.

The following reward models are currently supported:
* **Geneval** evaluates T2I models on complex compositional prompts.
* **OCR** provides an OCR-based reward.
* **PickScore** is a general-purpose T2I reward model trained on human preferences.
* **[DeQA](https://github.com/zhiyuanyou/DeQA-Score)** is a multimodal LLM-based image quality assessment model that measures the impact of distortions and texture damage on perceived quality.
* **ImageReward** is a general-purpose T2I reward model capturing text-image alignment, visual fidelity, and safety.
* **QwenVL** is an experimental reward model using prompt engineering.
* **Aesthetic** is a CLIP-based linear regressor predicting image aesthetic scores.
* **JPEG\_Compressibility** measures image size as a proxy for quality.
* **UnifiedReward** is a state-of-the-art reward model for multimodal understanding and generation, topping the human preference leaderboard.

        
## ✨ Important Hyperparameters
You can adjust the parameters in `config/grpo.py` to tune different hyperparameters. An empirical finding is that `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48`, i.e., `group_number=48`, `group_size=24`.
Additionally, setting `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2`.

## 🤗 Acknowledgement
This repo is based on [flow-grpo](https://github.com/yifan123/flow_grpo), [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) and [diffusers](https://github.com/huggingface/diffusers). We thank the authors for their valuable contributions to the AIGC community. Special thanks to Kevin Black for the excellent *ddpo-pytorch* repo.

## ⭐Citation
If you find Flow-CPS useful for your research or projects, we would greatly appreciate it if you could cite the following paper:
```
@article{wang2025coefficients,
  title={Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching},
  author={Wang, Feng and Yu, Zihao},
  journal={arXiv preprint arXiv:2509.05952},
  year={2025}
}
```
```
