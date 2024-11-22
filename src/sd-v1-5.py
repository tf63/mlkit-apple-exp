import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from src.utils import ExperimentalContext, save_images


def inference(context: ExperimentalContext, prompt: str):
    scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(context.device)

    batch_size = 2
    prompts = [prompt for i in range(batch_size)]

    images = pipeline(
        prompts,
        num_inference_steps=50,
        generator=context.generator,
    ).images

    save_images(images, prompt.replace(' ', '_'))


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps')
    prompt = 'a photo of an astronaut riding a horse on mars'

    inference(context=context, prompt=prompt)
