import src.patches.scheduling_euler_ancestral_discrete  # noqa

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)

from src.utils import ExperimentalContext, save_images
from diffusers.utils.loading_utils import load_image
from diffusers.utils.pil_utils import make_image_grid


def inference(context: ExperimentalContext, prompt: str):
    pipeline_text2img = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
    ).to(context.device)

    pipeline_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipeline_text2img).to(context.device)

    init_image = load_image(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png'
    )
    init_image = init_image.resize((512, 512))

    image = pipeline_img2img(prompt, image=init_image, strength=0.5, guidance_scale=0.0, num_inference_steps=2).images[
        0
    ]
    image = make_image_grid([init_image, image], rows=1, cols=2)

    save_images([image], 'test.png')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps')
    prompt = 'cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k'

    inference(context=context, prompt=prompt)
