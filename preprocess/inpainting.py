from diffusers import StableDiffusionInpaintPipeline
import torch
import os
from PIL import Image
import glob
import numpy as np
import logging
from tqdm import tqdm
import sys
from pathlib import Path
import re
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def dummy_safety_checker(images, clip_input, **kwargs):
    """Dummy safety checker that always passes"""
    return images, [False] * len(images)

def setup_inpainting_pipeline():
    """Sets up the inpainting pipeline with appropriate device configuration"""
    try:
        # Get absolute path of the project root
        import os
        
        # Define model directory path
        model_dir = Path(os.path.expanduser("~/Desktop/comic_new/models/stable-diffusion-inpainting"))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        # Check if model exists - looking for safetensors files
        has_model = False
        if model_dir.exists():
            # Look for model_index.json (proper way to detect diffusers model)
            if (model_dir / "model_index.json").exists():
                logging.info(f"Found model_index.json in {model_dir}")
                
                # Check specifically for safetensors files in unet directory
                unet_dir = model_dir / "unet"
                if unet_dir.exists():
                    safetensor_files = list(unet_dir.glob("*.safetensors"))
                    if safetensor_files:
                        logging.info(f"Found safetensors files in unet directory: {[f.name for f in safetensor_files]}")
                        has_model = True
        
        if has_model:
            logging.info(f"Loading inpainting model from local cache: {model_dir}")
            pipeline_path = str(model_dir)
            
            # Load model with safetensors explicitly enabled
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                pipeline_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,  # Explicitly use safetensors format
                requires_safety_checker=False
            ).to(device)
        else:
            logging.info(f"Model not found at {model_dir} or missing safetensors files. Downloading...")
            # Download from HF with default safetensors
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,  # Prefer safetensors when downloading
                requires_safety_checker=False
            ).to(device)
        
        # Disable safety checker
        pipeline.safety_checker = dummy_safety_checker
        return pipeline
    except Exception as e:
        logging.error(f"Failed to setup inpainting pipeline: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

def inpaint_panel(pipeline, panel_image, mask_image, original_panel_path, prompt="comic panel background, detailed, high quality, clean, no text, no speech bubbles", guidance_scale=8.5):
    """
    Inpaints a panel using an existing mask and blends with original image.
    Returns both the raw inpainting result and the final composited result.
    """
    try:
        # Extract panel name for better logging
        panel_name = Path(original_panel_path).stem
        logging.info(f"⏳ Processing panel: {panel_name}")
        
        # Convert mask to grayscale and check if it has any white pixels
        mask_array = np.array(mask_image.convert('L'))
        white_pixel_percent = np.sum(mask_array > 0) / mask_array.size * 100
        logging.info(f"Panel {panel_name}: Mask has {white_pixel_percent:.2f}% white pixels")
        
        if np.max(mask_array) == 0:  # If mask is completely black
            logging.info(f"Panel {panel_name}: No areas to inpaint (mask is completely black). Using original image.")
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
        start_time = time.time()
        result = pipeline(
            prompt=prompt,
            image=panel_image,
            mask_image=mask_image,
            num_inference_steps=50,
            guidance_scale=guidance_scale
        ).images[0]
        inpaint_time = time.time() - start_time
        logging.info(f"Panel {panel_name}: Inpainting completed in {inpaint_time:.2f} seconds")
        
        # Load original panel
        original_panel = Image.open(original_panel_path).convert('RGB')
        original_size = original_panel.size
        
        # Resize result and mask to match original panel size
        raw_result = result.resize(original_size, Image.Resampling.LANCZOS)
        blend_mask = mask_image.convert('L').resize(original_size, Image.Resampling.LANCZOS)
        
        # Blend original panel and inpainted result using the mask
        final_result = Image.composite(raw_result, original_panel, blend_mask)
        logging.info(f"✅ Panel {panel_name}: Processing complete")
        
        return raw_result, final_result
    except Exception as e:
        logging.error(f"Failed to inpaint panel {original_panel_path}: {str(e)}")
        raise

def process_all_panels(start_panel=44):
    """
    Process all panels using existing masks
    
    Args:
        start_panel (int): The panel number to start processing from (inclusive)
    """
    try:
        # Setup pipeline
        pipeline = setup_inpainting_pipeline()
        
        # Get all panel images
        panels_dir = Path("data/panels")
        masks_dir = Path("data/masks")
        output_dir = Path("data/inpainted")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get panel files and sort them properly using numerical order
        panel_paths = list(panels_dir.glob("*.png"))
        
        # Debug current order before sorting
        logging.info(f"Before sorting, first few panels: {[p.name for p in panel_paths[:5]]}")
        
        # Extract panel numbers and sort numerically
        def get_panel_number(path):
            # Extract numerical part from filenames like "panel_001.png"
            match = re.search(r'panel_(\d+)', path.name)
            if match:
                return int(match.group(1))
            return float('inf')  # Put panels without numbers at the end
        
        # Sort by the numerical panel number
        panel_paths.sort(key=get_panel_number)
        
        # Filter panels based on the starting panel number
        if start_panel > 0:
            original_count = len(panel_paths)
            panel_paths = [p for p in panel_paths if get_panel_number(p) >= start_panel]
            logging.info(f"Starting from panel {start_panel}, filtered from {original_count} to {len(panel_paths)} panels")
        
        # Debug the order after sorting
        logging.info(f"After sorting, first few panels: {[p.name for p in panel_paths[:5]]}")
        
        if not panel_paths:
            logging.warning("No panel images found")
            return
        
        total_panels = len(panel_paths)
        logging.info(f"Found {total_panels} panels to process")
        
        start_time_all = time.time()
        
        # Process each panel
        for i, panel_path in enumerate(panel_paths):
            panel_name = panel_path.stem
            panel_num = re.search(r'(\d+)', panel_name)
            panel_num = panel_num.group(1) if panel_num else "unknown"
            
            progress_percent = (i / total_panels) * 100
            logging.info(f"[{i+1}/{total_panels}] ({progress_percent:.1f}%) Processing panel {panel_name}")
            
            try:
                mask_path = masks_dir / f"{panel_name}_mask.png"
                
                if not mask_path.exists():
                    logging.warning(f"Mask not found for panel {panel_path}")
                    continue
                
                # Load images
                panel_start_time = time.time()
                logging.info(f"Loading panel image and mask for {panel_name}")
                panel_image = Image.open(panel_path)
                mask_image = Image.open(mask_path)
                
                # Process panel
                logging.info(f"Starting inpainting for panel {panel_name}")
                raw_result, final_result = inpaint_panel(pipeline, panel_image, mask_image, panel_path)
                
                # Save final result
                output_path = output_dir / f"{panel_name}_inpainted.png"
                final_result.save(output_path)
                
                panel_time = time.time() - panel_start_time
                logging.info(f"✓ Panel {panel_name} completed in {panel_time:.2f} seconds. Saved to: {output_path}")
                
                # Estimate remaining time
                elapsed_time = time.time() - start_time_all
                panels_left = total_panels - (i + 1)
                if i > 0:  # Avoid division by zero
                    avg_time_per_panel = elapsed_time / (i + 1)
                    est_remaining = avg_time_per_panel * panels_left
                    logging.info(f"Estimated time remaining: {est_remaining/60:.1f} minutes ({est_remaining:.0f} seconds)")
                
            except Exception as e:
                logging.error(f"Failed to process panel {panel_path}: {str(e)}")
                continue
        
        total_time = time.time() - start_time_all
        logging.info(f"All panels processed in {total_time/60:.2f} minutes")
        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f"Failed to process panels: {str(e)}")
        raise


# Add a function to run from the command line with arguments
def main():
    """Command-line interface for inpainting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inpaint panels to remove text bubbles")
    parser.add_argument("--start-panel", type=int, default=0, 
                        help="Panel number to start from (e.g., 10 to start from panel_010.png)")
    args = parser.parse_args()
    
    # Run process_all_panels with the specified starting panel
    process_all_panels(start_panel=args.start_panel)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)
