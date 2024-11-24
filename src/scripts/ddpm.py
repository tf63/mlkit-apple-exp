import os

import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from src.utils import ExperimentalContext


def inference(context: ExperimentalContext, batch_size, num_inference_steps=1000):
    # モデルの読み込み
    pipeline = DDPMPipeline.from_pretrained('google/ddpm-cat-256', torch_dtype=torch.float16).to(context.device)

    # 推論
    images = pipeline(
        batch_size=batch_size,
        generator=context.generator,
        num_inference_steps=num_inference_steps,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, 'uncond', f'i{i}_n{num_inference_steps}')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps', root_dir=os.path.join('out', 'ddpm_cat'))
    batch_size = 4

    inference(context=context, batch_size=batch_size, num_inference_steps=100)
