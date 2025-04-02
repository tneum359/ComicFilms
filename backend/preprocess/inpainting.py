from diffusers import StableDiffusionInpaintPipeline
import torch
import os
from PIL import Image
import glob
import numpy as np
import logging
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessor/logs/inpainting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_inpainting_pipeline():
    """Sets up the inpainting pipeline with appropriate device configuration"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False,
            requires_safety_checker=False
        ).to(device)
        return pipeline
    except Exception as e:
        logging.error(f"Failed to setup inpainting pipeline: {str(e)}")
        raise

def inpaint_panel(pipeline, panel_image, mask_image, original_panel_path, prompt="comic panel background, detailed, high quality, clean, no text, no speech bubbles"):
    """
    Inpaints a panel using an existing mask and blends with original image.
    Returns both the raw inpainting result and the final composited result.
    """
    try:
        # Convert mask to grayscale and check if it has any white pixels
        mask_array = np.array(mask_image.convert('L'))
        if np.max(mask_array) == 0:  # If mask is completely black
            logging.info("No areas to inpaint (mask is completely black). Using original image.")
            original = Image.open(original_panel_path).convert('RGB')
            return original, original  # Return same image for both results
        
        # Convert images for stable diffusion processing
        panel_image = panel_image.convert('RGB')
        mask_image = mask_image.convert('RGB')
        
        # Resize for stable diffusion if needed
        width, height = panel_image.size
        if width > 512 or height > 512:
            ratio = min(512/width, 512/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            panel_image = panel_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            mask_image = mask_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Run inpainting
        result = pipeline(
            prompt=prompt,
            image=panel_image,
            mask_image=mask_image,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        # Load original panel
        original_panel = Image.open(original_panel_path).convert('RGB')
        original_size = original_panel.size
        
        # Resize result and mask to match original panel size
        raw_result = result.resize(original_size, Image.Resampling.LANCZOS)
        blend_mask = mask_image.convert('L').resize(original_size, Image.Resampling.LANCZOS)
        
        # Blend original panel and inpainted result using the mask
        final_result = Image.composite(raw_result, original_panel, blend_mask)
        
        return raw_result, final_result
    except Exception as e:
        logging.error(f"Failed to inpaint panel: {str(e)}")
        raise

def process_all_pages():
    """Process all pages using existing masks"""
    try:
        # Setup pipeline
        pipeline = setup_inpainting_pipeline()
        
        # Get all mask directories and sort them properly
        mask_dirs = glob.glob(os.path.join("preprocessor/masks", "page_*"))
        mask_dirs = sorted(mask_dirs, key=lambda x: int(os.path.basename(x).split('_')[1]))
        mask_dirs = [d for d in mask_dirs if int(os.path.basename(d).split('_')[1]) >= 4]
        
        if not mask_dirs:
            logging.warning("No mask directories found")
            return
        
        logging.info(f"Found {len(mask_dirs)} pages to process, starting from page 4")
        
        for mask_dir in tqdm(mask_dirs, desc="Processing pages"):
            page_num = int(os.path.basename(mask_dir).split('_')[1])
            logging.info(f"Processing page {page_num}")
            
            # Setup input/output directories with zero-padded page number
            panel_dir = os.path.join("preprocessor/outputs/panels", f"page_{page_num:03d}")
            output_dir = os.path.join("preprocessor/outputs/inpainted", f"page_{page_num}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all masks for this page
            mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*_mask.png")))
            
            for mask_path in tqdm(mask_paths, desc=f"Processing panels for page {page_num}"):
                try:
                    # Extract panel number from mask filename (panel_X_mask.png)
                    panel_idx = int(os.path.basename(mask_path).split('_')[1])
                    # Construct panel path with zero-padded panel number
                    panel_path = os.path.join(panel_dir, f"panel_{panel_idx:03d}.png")
                    
                    if not os.path.exists(panel_path):
                        logging.warning(f"Panel image not found: {panel_path}")
                        continue
                    
                    # Load images
                    panel_image = Image.open(panel_path)
                    mask_image = Image.open(mask_path)
                    
                    # Process panel
                    raw_result, final_result = inpaint_panel(pipeline, panel_image, mask_image, panel_path)
                    
                    # Save both results
                    raw_output_path = os.path.join(output_dir, f"panel_{panel_idx}_sd_result.png")
                    final_output_path = os.path.join(output_dir, f"panel_{panel_idx}_inpainted.png")
                    
                    raw_result.save(raw_output_path)
                    final_result.save(final_output_path)
                    logging.info(f"Saved inpainting results to: {raw_output_path} and {final_output_path}")
                    
                except Exception as e:
                    logging.error(f"Failed to process panel with mask {mask_path}: {str(e)}")
                    continue
        
        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f"Failed to process pages: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        process_all_pages()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)
