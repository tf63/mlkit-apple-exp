import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from src.utils import ExperimentalContext, save_images


def inference(context: ExperimentalContext):
    pipeline = DDPMPipeline.from_pretrained('google/ddpm-cat-256', torch_dtype=torch.float16).to(context.device)
    pipeline.enable_attention_slicing()

    batch_size = 1

    images = pipeline(
        batch_size=batch_size,
        generator=context.generator,
        num_inference_steps=1000,
    ).images

    save_images(images, 'ddpm_generated')


if __name__ == '__main__':
    context = ExperimentalContext(seed=42, device='mps')

    inference(context=context)
