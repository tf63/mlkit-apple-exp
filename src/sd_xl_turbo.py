import src.patches.scheduling_euler_ancestral_discrete  # noqa

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from src.utils import ExperimentalContext, save_images


def inference(context: ExperimentalContext, prompt: str):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
    ).to(context.device)

    batch_size = 1
    prompts = [prompt for i in range(batch_size)]

    images = pipeline(
        prompts,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=context.generator,
    ).images

    save_images(images, prompt.replace(' ', '_'))


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps')
    prompt = 'A cinematic shot of a baby racoon wearing an intricate italian priest robe'

    inference(context=context, prompt=prompt)
