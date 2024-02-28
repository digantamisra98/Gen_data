import os
import sys

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

class_name = sys.argv[1]
output_dir = sys.argv[2]
images_per_class = 50
batch_size = 4

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
    "A macro photography image of {class_name} showcasing unique textures.",
    "Extreme close-up photo of {class_name} revealing its unique textural details.",
    "Magnified image of {class_name} highlighting its fascinating textures and patterns.",
    "Macro photography of {class_name} showcasing the intricate details of its surface.",
    "Close-up view of {class_name} showcasing its unique textures through macro photography.",
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
    "A vintage photograph of {class_tname} with a warm, faded aesthetic.",
    "Create a nostalgic photo of {class_name} with a warm, vintage tone and faded colors.",
    "Generate an image of {class_name} in the style of a vintage photograph, featuring a warm color palette and faded appearance.",
    "Produce a photo of {class_name} with a classic aesthetic, using a warm color scheme and a subtle vintage fade.",
    "Depict {class_name} in a photo reminiscent of old times, with a warm, faded aesthetic and a vintage feel.",
]


pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")


for prompt_template in prompt_templates:
    prompt = prompt_template.format(class_name=class_name.lower())
    print(prompt)

    for i in tqdm(range(0, images_per_class, batch_size)):
        start, end = i, int(min(images_per_class, i + batch_size))
        inputs = {
            "prompt": [prompt] * (end - start),
            "generator": [
                torch.Generator("cuda").manual_seed(i) for i in range(start, end)
            ],
            "num_inference_steps": 40,
            "denoising": 0.8,
        }

        with torch.inference_mode():
            print("Base image generation ...")
            image = pipe(
                prompt=inputs["prompt"],
                generator=inputs["generator"],
                num_inference_steps=inputs["num_inference_steps"],
                denoising_end=inputs["denoising"],
                output_type="latent",
            ).images

            print("Refining base images ...")
            images = refiner(
                prompt=inputs["prompt"],
                generator=inputs["generator"],
                num_inference_steps=inputs["num_inference_steps"],
                denoising_start=inputs["denoising"],
                image=image,
            ).images

        print("Saving the generated images ...")
        for img in images:
            img.save(f"{output_dir}/{start}.png")
            start += 1

        del image
        torch.cuda.empty_cache()
