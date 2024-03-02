import os
import sys

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import argparse

os.environ["HF_TOKEN"] = "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"

# define class_name and output_dir as argparse arguments
parser = argparse.ArgumentParser(description="Prune Train")
parser.add_argument(
    "--class_name", type=str, help="class_name"
)
parser.add_argument("--output_dir", type=str, help="output_dir")
parser.add_argument("--images_per_class", type=int, default=10, help="images_per_class")

args = parser.parse_args()
class_name = args.class_name
output_dir = args.output_dir
images_per_class = args.images_per_class

prompt_templates = [
    "A black and white image of {class_name} highlighting dramatic contrasts.",
    "High-contrast black and white photo of {class_name} emphasizing dramatic lighting.",
    "Black and white {class_name} image with bold contrasts, showcasing its form.",
    "Black and white {class_name} photography capturing stark shadows and highlights.",
    "Dramatic black and white image of {class_name} with deep blacks and bright whites.",
    "A high-resolution photo of {class_name} capturing fine details.",
    "Close-up photo of {class_name} in high resolution, revealing intricate details.",
    "Detailed, high-resolution image of {class_name} showcasing its intricate features.",
    "Sharp, high-resolution photo of {class_name} capturing every minute detail.",
    "Crystal clear photo of {class_name} in high resolution, showing fine textures and patterns.",
    "A minimalist image of {class_name} using clean lines and muted colors.",
    "Simple and elegant photo of {class_name} featuring clean lines and muted tones.",
    "Minimalist composition of {class_name} with clean lines and a subdued color palette.",
    "Uncluttered image of {class_name} using clean lines and muted colors for a serene look.",
    "Photo of {class_name} with a minimalist aesthetic, featuring clean lines and soft colors.",
    "A photo of {class_name} in analogous colors.",
    "{class_name} photo featuring a harmonious blend of analogous colors.",
    "Image of {class_name} showcasing a cohesive color scheme using analogous colors.",
    "Photo of {class_name} where the colors naturally complement each other, following an analogous color scheme.",
    "{class_name} image with a pleasing color palette using analogous colors that sit next to each other on the color wheel.",
    "l Prompt: A photo of {class_name} in complementary colors.",
    "Image of {class_name} featuring contrasting yet balanced colors, using a complementary color scheme.",
    "Photo of {class_name} where the colors create a sense of visual tension through complementary colors.",
    "{class_name} image with a vibrant color palette using complementary colors that sit opposite each other on the color wheel.",
    "Photo of {class_name} showcasing a dynamic color combination with complementary hues.",
    "A photo of {class_name} in earth tones.",
    "Photo of {class_name} featuring natural, earthy colors like brown, green, and beige.",
    "Image of {class_name} with a warm and inviting color palette reminiscent of nature, using earth tones.",
    "{class_name} photo with a grounded and organic color scheme inspired by the natural world, using earth tones.",
    "Photo of {class_name} showcasing a calming and natural color palette with earth tones.",
    "A photo of {class_name} in neutral tones.",
    "Photo of {class_name} featuring a subtle and timeless color palette with neutral tones.",
    "Image of {class_name} with a classic and versatile color scheme using neutral tones.",
    "{class_name} photo with a sophisticated and understated color palette dominated by neutral tones.",
    "Photo of {class_name} where the focus falls on the subject itself through the use of neutral tones.",
    "A photo of {class_name}.",
    "Capture a stunning image of {class_name}.",
    "Showcase the beauty of {class_name} in a photo.",
    "Create a photo that portrays {class_name} accurately.",
    "Generate a visually appealing image of {class_name}.",
    "A realistic image of {class_name}.",
    "Create a lifelike image of {class_name} that captures its true essence.",
    "Generate a photo of {class_name} that appears true to life.",
    "Produce an image of {class_name} with realistic details and lighting.",
    "Depict {class_name} in a photo with a high degree of visual realism.",
    "A vintage photograph of {class_name} with a warm, faded aesthetic.",
    "Create a nostalgic photo of {class_name} with a warm, vintage tone and faded colors.",
    "Generate an image of {class_name} in the style of a vintage photograph, featuring a warm color palette and faded appearance.",
    "Produce a photo of {class_name} with a classic aesthetic, using a warm color scheme and a subtle vintage fade.",
    "Depict {class_name} in a photo reminiscent of old times, with a warm, faded aesthetic and a vintage feel.",
]

print(f"Loading Floyd model - stage 1")
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
)
# stage_1.enable_xformers_memory_efficient_attention() # remove line if torch.__version__ >= 2.0.0
stage_1.enable_model_cpu_offload()

print(f"Loading Floyd model - stage 2")
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
)
# stage_2.enable_xformers_memory_efficient_attention() # remove line if torch.__version__ >= 2.0.0
stage_2.enable_model_cpu_offload()

print(f"Loading Floyd model - stage 3")
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    **safety_modules,
    torch_dtype=torch.float16,
)
# stage_3.enable_xformers_memory_efficient_attention() # remove line if torch.__version__ >= 2.0.0
stage_3.enable_model_cpu_offload()

for count, prompt_template in enumerate(prompt_templates):
    prompt = prompt_template.format(class_name=class_name.lower())
    print(prompt)

    for i in tqdm(range(images_per_class)):
        with torch.inference_mode():
            prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
            image = stage_1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                output_type="pt",
            ).images
            image = stage_2(
                image=image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                output_type="pt",
            ).images
            image = stage_3(prompt=prompt, image=image, noise_level=100).images
        image = image[0]
        image.save(f"{output_dir}/{count}_{i}.png")
