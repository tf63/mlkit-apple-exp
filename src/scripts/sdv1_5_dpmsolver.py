import src.patches.scheduling_dpmsolver_multistep  # noqa

import os
from typing import List

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from src.utils import ExperimentalContext


def inference(context: ExperimentalContext, prompts: List[str], num_inference_steps=50):
    # モデルの読み込み
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
    ).to(context.device)

    # スケジューラの読み込み
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    negative_prompts = ['low quality, bad quality' for _ in range(len(prompts))]

    # 推論
    images = pipeline(
        prompts,
        negative_prompt=negative_prompts,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, prompts[i].replace(' ', '_'), f'i{i}_n{num_inference_steps}')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps', root_dir=os.path.join('out', 'sdv1_5_dpmsolver'))

    prompts = [
        'a cat, fat, with brown fur, with short legs',
        'a cat, fat, with white fur, with short legs',
    ]

    inference(context=context, prompts=prompts, num_inference_steps=25)
