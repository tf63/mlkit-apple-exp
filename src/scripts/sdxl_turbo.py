import src.patches.scheduling_euler_ancestral_discrete  # noqa

import os
from typing import List

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from src.utils import ExperimentalContext


def inference(context: ExperimentalContext, prompts: List[str], guidance_scale=0.0, num_inference_steps=1):
    # モデルの読み込み
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
    ).to(context.device)

    negative_prompt = 'low quality, bad quality'

    # 推論
    images = pipeline(
        prompts,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, prompts[i].replace(' ', '_'), f'i{i}_n{num_inference_steps}_s{guidance_scale}')


def inference_loop(context: ExperimentalContext, guidance_scale=0.0, num_inference_steps=1):
    # モデルの読み込み
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
    ).to(context.device)

    batch_size = 1

    while True:
        prompt = input('Enter a prompt (type "q" to quit): ')
        if prompt == 'q':
            break

        prompts = [prompt for i in range(batch_size)]
        negative_prompt = 'low quality, bad quality'

        # 推論
        images = pipeline(
            prompts,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=context.generator,
        ).images

        # 画像の保存
        for i, image in enumerate(images):
            context.save_image(image, prompts[i].replace(' ', '_'), f'i{i}_n{num_inference_steps}_s{guidance_scale}')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps', root_dir=os.path.join('out', 'sdxl_turbo'))
    prompts = [
        'a cat, fat, with brown fur, with short legs',
        'a cat, fat, with white fur, with short legs',
    ]

    inference(context=context, prompts=prompts, num_inference_steps=4, guidance_scale=0.0)
    # inference_loop(context=context)
