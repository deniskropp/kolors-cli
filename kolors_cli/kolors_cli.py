import argparse
import torch
import random
import numpy as np
from huggingface_hub import snapshot_download
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from kolors.pipelines import pipeline_stable_diffusion_xl_chatglm_256_ipadapter, pipeline_stable_diffusion_xl_chatglm_256
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models import unet_2d_condition
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors")
ckpt_IPA_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors-IP-Adapter-Plus")

text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder', torch_dtype=torch.float16).half().to(device)
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half().to(device)
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet_t2i = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to(device)
unet_i2i = unet_2d_condition.UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to(device)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'{ckpt_IPA_dir}/image_encoder', ignore_mismatched_sizes=True).to(dtype=torch.float16, device=device)
ip_img_size = 336
clip_image_processor = CLIPImageProcessor(size=ip_img_size, crop_size=ip_img_size)

pipe_t2i = pipeline_stable_diffusion_xl_chatglm_256.StableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder, 
    tokenizer=tokenizer, 
    unet=unet_t2i, 
    scheduler=scheduler, 
    force_zeros_for_empty_prompt=False
).to(device)

pipe_i2i = pipeline_stable_diffusion_xl_chatglm_256_ipadapter.StableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet_i2i,
    scheduler=scheduler,
    image_encoder=image_encoder,
    feature_extractor=clip_image_processor,
    force_zeros_for_empty_prompt=False
).to(device)

if hasattr(pipe_i2i.unet, 'encoder_hid_proj'):
    pipe_i2i.unet.text_encoder_hid_proj = pipe_i2i.unet.encoder_hid_proj
    
pipe_i2i.load_ip_adapter(f'{ckpt_IPA_dir}' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt, 
          ip_adapter_image=None, 
          ip_adapter_scale=0.5, 
          negative_prompt="", 
          seed=0, 
          randomize_seed=False, 
          width=1024, 
          height=1024, 
          guidance_scale=5.0, 
          num_inference_steps=25):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    if ip_adapter_image is None:
        pipe_t2i.to(device)
        image = pipe_t2i(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps, 
            width=width, 
            height=height,
            generator=generator
        ).images[0] 
        return image
    else:
        pipe_i2i.to(device)
        image_encoder.to(device)
        pipe_i2i.image_encoder = image_encoder
        pipe_i2i.set_ip_adapter_scale([ip_adapter_scale])
        image = pipe_i2i(
                prompt=prompt,
                ip_adapter_image=[ip_adapter_image],
                negative_prompt=negative_prompt, 
                height=height,
                width=width,
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator
            ).images[0]
        return image

def main():
    parser = argparse.ArgumentParser(description="Kolors CLI for generating images based on prompts.")
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt for image generation.")
    parser.add_argument("--ip_adapter_image", type=str, default=None, help="Path to the image prompt (optional).")
    parser.add_argument("--ip_adapter_scale", type=float, default=0.5, help="Image influence scale (default: 0.5).")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (default: '').")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation (default: 0).")
    parser.add_argument("--randomize_seed", action="store_true", help="Randomize the seed (default: False).")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image (default: 1024).")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image (default: 1024).")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale (default: 5.0).")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps (default: 25).")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path (default: output.png).")

    args = parser.parse_args()

    if args.ip_adapter_image:
        try:
            ip_adapter_image = Image.open(args.ip_adapter_image)
        except Exception as e:
            print(f"Error opening image file {args.ip_adapter_image}: {e}")
            return
    else:
        ip_adapter_image = None

    try:
        image = infer(
            prompt=args.prompt,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=args.ip_adapter_scale,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            randomize_seed=args.randomize_seed,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        )
        image.save(args.output)
        print(f"Image saved to {args.output}")
    except Exception as e:
        print(f"Error during image generation: {e}")

if __name__ == "__main__":
    main()