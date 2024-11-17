from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from src.utils import prepare


def inference(prompt):
    scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        scheduler=scheduler,
    ).to('mps')

    image = pipe(prompt, batch_size=1, num_inference_steps=5).images[0]

    image.save('out/astronaut_rides_horse.png')


if __name__ == '__main__':
    prepare()

    prompt = 'a photo of an astronaut riding a horse on mars'
    inference(prompt=prompt)
