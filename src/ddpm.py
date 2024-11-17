# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
# pipe = pipe.to('mps')

# # pipe.enable_attention_slicing()

# prompt = 'a photo of an astronaut riding a horse on mars'

# image = pipe(prompt).images[0]

# image.save('output.jpg')

# from diffusers import DDPMPipeline
import os

from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from src.utlis import fix_seed

if __name__ == '__main__':
    fix_seed(0)
    os.makedirs('out', exist_ok=True)

    pipe = DDPMPipeline.from_pretrained('google/ddpm-cat-256').to('mps')
    pipe.enable_attention_slicing()

    images = pipe(batch_size=1, num_inference_steps=1000).images

    for i, image in enumerate(images):
        # save image
        image.save(f'out/ddpm_generated_image{i}.png')
