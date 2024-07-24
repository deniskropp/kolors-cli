import spaces
import random
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from kolors.pipelines import pipeline_stable_diffusion_xl_chatglm_256_ipadapter, pipeline_stable_diffusion_xl_chatglm_256
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from kolors.models import unet_2d_condition
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
import gradio as gr
import numpy as np

device = "cuda"
ckpt_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors")
ckpt_IPA_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors-IP-Adapter-Plus")

text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder', torch_dtype=torch.float16).half().to(device)
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half().to(device)
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet_t2i = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to(device)
unet_i2i = unet_2d_condition.UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to(device)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'{ckpt_IPA_dir}/image_encoder',ignore_mismatched_sizes=True).to(dtype=torch.float16, device=device)
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

@spaces.GPU
def infer(prompt, 
          ip_adapter_image = None, 
          ip_adapter_scale = 0.5, 
          negative_prompt = "", 
          seed = 0, 
          randomize_seed = True, 
          width = 1024, 
          height = 1024, 
          guidance_scale = 5.0, 
          num_inference_steps = 25
          ):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    if ip_adapter_image is None:
        pipe_t2i.to(device)
        image = pipe_t2i(
            prompt = prompt, 
            negative_prompt = negative_prompt,
            guidance_scale = guidance_scale, 
            num_inference_steps = num_inference_steps, 
            width = width, 
            height = height,
            generator = generator
        ).images[0] 
        return image
    else:
        pipe_i2i.to(device)
        image_encoder.to(device)
        pipe_i2i.image_encoder = image_encoder
        pipe_i2i.set_ip_adapter_scale([ip_adapter_scale])
        image = pipe_i2i(
                prompt=prompt ,
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

examples = [
    ["一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”", None, None],
    ["3D anime style, hyperrealistic oil painting, dolphin leaping out of the water", None, None],
    ["穿着黑色T恤衫，上面中文绿色大字写着“可图”", "image/test_ip.jpg", 0.5],
    ["A cute dog is running", "image/test_ip2.png", 0.5]
]

css="""
#col-left {
    margin: 0 auto;
    max-width: 600px;
}
#col-right {
    margin: 0 auto;
    max-width: 750px;
}
"""

def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

with gr.Blocks(css=css) as Kolors:
    gr.HTML(load_description("assets/title.md"))
    with gr.Row():
        with gr.Column(elem_id="col-left"):
            with gr.Row():
                prompt = gr.Textbox(
                    label="Prompt",
                    show_label=False,
                    placeholder="Enter your prompt",
                    container=False,
                    lines=2
                )
            with gr.Row():
                ip_adapter_image = gr.Image(label="Image Prompt (optional)", type="pil")
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    placeholder="Enter a negative prompt",
                    visible=True,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=5.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=25,
                    )
                with gr.Row():
                    ip_adapter_scale = gr.Slider(
                        label="Image influence scale",
                        info="Use 1 for creating variations",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                    )
            with gr.Row():
                run_button = gr.Button("Run", scale=0)
            
        with gr.Column(elem_id="col-right"):
            result = gr.Image(label="Result", show_label=False)
    
    with gr.Row():
        gr.Examples(
                fn = infer,
                examples = examples,
                inputs = [prompt, ip_adapter_image, ip_adapter_scale],
                outputs = [result]
            )

    run_button.click(
        fn = infer,
        inputs = [prompt, ip_adapter_image, ip_adapter_scale, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result]
    )

Kolors.queue().launch(debug=True)
