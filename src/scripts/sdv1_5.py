import os
from typing import List

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from src.utils import ExperimentalContext


def inference(context: ExperimentalContext, prompts: List[str], guidance_scale=0.0, num_inference_steps=50):
    # スケジューラの読み込み
    scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

    # モデルの読み込み
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(context.device)

    negative_prompts = ['low quality, bad quality' for _ in range(len(prompts))]

    # 推論
    images = pipeline(
        prompts,
        negative_prompt=negative_prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, prompts[i].replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps', root_dir=os.path.join('out', 'sdv1_5'))

    prompts = [
        'a cat, fat, with brown fur, with short legs',
        'a cat, fat, with white fur, with short legs',
    ]

    inference(context=context, prompts=prompts, num_inference_steps=50, guidance_scale=2.0)
