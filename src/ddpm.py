from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from src.utils import prepare


def inference():
    pipe = DDPMPipeline.from_pretrained('google/ddpm-cat-256').to('mps')
    pipe.enable_attention_slicing()

    images = pipe(batch_size=1, num_inference_steps=1000).images

    for i, image in enumerate(images):
        # save image
        image.save(f'out/ddpm_generated_image{i}.png')


if __name__ == '__main__':
    prepare()

    inference()
