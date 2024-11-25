# https://huggingface.co/kandinsky-community/kandinsky-2-1-inpaint
import os

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting

# from diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
from diffusers.utils.loading_utils import load_image
from diffusers.utils.pil_utils import make_image_grid

from src.utils import ExperimentalContext


def inference(context: ExperimentalContext, prompt: str, guidance_scale=0.0, num_inference_steps=2):
    # text2imgモデルの読み込み

    pipeline_inpaint = AutoPipelineForInpainting.from_pretrained(
        'kandinsky-community/kandinsky-2-1-inpaint', torch_dtype=torch.float16
    ).to(context.device)
    # pipeline_inpaint = KandinskyInpaintPipeline.from_pretrained(
    #     'kandinsky-community/kandinsky-2-1-inpaint', torch_dtype=torch.float16
    # ).to(context.device)

    negative_prompt = 'low quality, bad quality'

    # ソース画像･マスク画像の読み込み
    # https://huggingface.co/docs/diffusers/ja/tutorials/autopipeline
    img_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    mask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'

    init_image = load_image(img_url).convert('RGB')
    mask_image = load_image(mask_url).convert('RGB')

    # 推論
    image = pipeline_inpaint(
        prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        # strength=0.8,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images[0]

    # 画像の保存
    image_compare = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    context.save_image(image, prompt.replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}')
    context.save_image(image_compare, prompt.replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}_comp')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps', root_dir=os.path.join('out', 'kandinsky_inpaint_sample'))
    prompt = 'a bench'
    # prompt = 'A majestic tiger sitting on a bench'

    inference(context=context, prompt=prompt, num_inference_steps=25)