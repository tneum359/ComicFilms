from diffusers import StableDiffusionInpaintPipeline
import torch
import os
from PIL import Image, ImageDraw
import glob
from ultralytics import YOLO
import numpy as np
import logging
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inpainting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_inpainting_pipeline():
    """
    Sets up the inpainting pipeline with appropriate device configuration
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=False
        ).to(device)
        return pipeline
    except Exception as e:
        logging.error(f"Failed to setup inpainting pipeline: {str(e)}")
        raise

def setup_bubble_detector():
    """
    Sets up the speech bubble detection model using YOLOv8
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "models/yolov8m_seg-speech-bubble.pt"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            logging.info("Downloading speech bubble detector model...")
            import huggingface_hub
            huggingface_hub.hf_hub_download(
                repo_id="kitsumed/yolov8m_seg-speech-bubble",
                filename="model.pt",
                local_dir="models",
                local_dir_use_symlinks=False
            )
            os.rename("models/model.pt", model_path)
        
        logging.info(f"Loading speech bubble detector from {model_path}")
        model = YOLO(model_path)
        return model
    except Exception as e:
        logging.error(f"Failed to setup bubble detector: {str(e)}")
        raise

def create_text_bubble_mask(image, model):
    """
    Creates a binary mask for text bubbles in the image using YOLOv8 model
    """
    try:
        # Run inference
        results = model(image, conf=0.5)[0]
        
        # Create white mask (areas to be inpainted)
        mask = Image.new('RGB', image.size, (255, 255, 255))
        draw = ImageDraw.Draw(mask)
        
        # Draw detected bubbles as black regions on the mask (areas to be preserved)
        for box in results.boxes:
            if box.conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Add some padding around the detected bubble
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.size[0], x2 + padding)
                y2 = min(image.size[1], y2 + padding)
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        
        return mask
    except Exception as e:
        logging.error(f"Failed to create text bubble mask: {str(e)}")
        raise

def inpaint_text_bubbles(pipeline, page_image, text_bubble_mask, prompt="", page_number=None, panel_number=None):
    """
    Inpaints text bubbles in a manga/comic page
    """
    try:
        # Create output directory structure matching input
        output_dir = os.path.join("preprocessor/outputs/inpainted", f"page_{page_number}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure images are in RGB mode
        page_image = page_image.convert('RGB')
        text_bubble_mask = text_bubble_mask.convert('RGB')
        
        # Store original size
        original_size = page_image.size
        
        # Resize images if needed (SD typically expects 512x512 or similar)
        width, height = page_image.size
        if width > 512 or height > 512:
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(512/width, 512/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize both images
            page_image = page_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            text_bubble_mask = text_bubble_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Run inpainting with a more specific prompt for manga backgrounds
        prompt = prompt or "manga panel background, detailed, high quality, clean, no text, no speech bubbles"
        result = pipeline(
            prompt=prompt,
            image=page_image,
            mask_image=text_bubble_mask,
            num_inference_steps=50,  # Increased for better quality
            guidance_scale=7.5  # Balanced between creativity and prompt adherence
        )
        
        # Resize the inpainted result back to original size
        inpainted_image = result.images[0]
        if inpainted_image.size != original_size:
            inpainted_image = inpainted_image.resize(original_size, Image.Resampling.LANCZOS)
        
        # Save the mask
        mask_filename = f"panel_{panel_number}_mask.png"
        mask_path = os.path.join(output_dir, mask_filename)
        text_bubble_mask.save(mask_path)
        logging.info(f"Saved mask to: {mask_path}")
        
        # Save the inpainted result
        inpainted_filename = f"panel_{panel_number}_inpainted.png"
        inpainted_path = os.path.join(output_dir, inpainted_filename)
        inpainted_image.save(inpainted_path)
        
        logging.info(f"Saved inpainted image to: {inpainted_path}")
        return inpainted_image
    except Exception as e:
        logging.error(f"Failed to inpaint text bubbles: {str(e)}")
        raise

def process_all_pages():
    """
    Process all pages in the input directory starting from page 4
    """
    try:
        # Setup both pipelines
        logging.info("Setting up pipelines...")
        inpainting_pipeline = setup_inpainting_pipeline()
        bubble_model = setup_bubble_detector()
        
        base_dir = "preprocessor/outputs/panels"
        
        # Get all page directories
        page_dirs = sorted(glob.glob(os.path.join(base_dir, "page_*")))
        if not page_dirs:
            logging.warning(f"No page directories found in {base_dir}")
            return
        
        # Filter to start from page 4
        page_dirs = [d for d in page_dirs if int(os.path.basename(d).split('_')[1]) >= 4]
        logging.info(f"Found {len(page_dirs)} pages to process (starting from page 4)")
        
        # Process each page with progress bar
        for page_dir in tqdm(page_dirs, desc="Processing pages"):
            page_num = int(os.path.basename(page_dir).split('_')[1])
            logging.info(f"Processing page {page_num}")
            
            # Get all panel images in the current page
            panel_images = sorted(glob.glob(os.path.join(page_dir, "*.png")))
            
            # Process each panel with progress bar
            for panel_idx, panel_path in enumerate(tqdm(panel_images, desc=f"Processing panels for page {page_num}", leave=False)):
                try:
                    # Load panel image
                    panel_image = Image.open(panel_path)
                    
                    # Create mask for the panel using the bubble detector
                    mask = create_text_bubble_mask(panel_image, bubble_model)
                    
                    # Process the panel
                    inpainted_panel = inpaint_text_bubbles(
                        pipeline=inpainting_pipeline,
                        page_image=panel_image,
                        text_bubble_mask=mask,
                        prompt="manga panel background, detailed, high quality",
                        page_number=page_num,
                        panel_number=panel_idx
                    )
                    
                except Exception as e:
                    logging.error(f"Failed to process panel {panel_path}: {str(e)}")
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
