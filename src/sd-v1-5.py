from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from src.utils import prepare, save_images


def inference(prompt: str):
    scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        scheduler=scheduler,
    ).to('mps')

    batch_size = 2
    prompts = [prompt for i in range(batch_size)]

    images = pipe(prompts, num_inference_steps=5).images

    save_images(images, prompt.replace(' ', '_'))


if __name__ == '__main__':
    prepare()

    prompt = 'a photo of an astronaut riding a horse on mars'
    inference(prompt=prompt)
