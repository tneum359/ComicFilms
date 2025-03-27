import torch
import os
import glob
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_gen.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_pipeline():
    """Setup the Stable Video Diffusion pipeline"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logging.info(f"Using device: {device} with dtype: {torch_dtype}")
        
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None
        )
        
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.unet.enable_forward_chunking()
        else:
            pipe = pipe.to(device)
            
        return pipe, device
    except Exception as e:
        logging.error(f"Failed to setup pipeline: {str(e)}")
        raise

def process_panel(pipe, device, image_path, output_path):
    """Process a single panel and generate video"""
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        
        # Calculate new dimensions maintaining aspect ratio
        target_height = 576
        aspect_ratio = image.width / image.height
        target_width = int(target_height * aspect_ratio)
        target_width = min(1024, target_width)  # Cap width at 1024
        
        # Resize image
        image = image.resize((target_width, target_height))
        
        # Configure generation parameters based on device
        if device == "cuda":
            frames = pipe(
                image,
                decode_chunk_size=2,
                generator=torch.Generator(device=device).manual_seed(42),
                num_frames=14,  # 2 seconds at 7fps
                motion_bucket_id=127,  # High motion
                noise_aug_strength=0.1  # Reduced noise
            ).frames[0]
        else:
            frames = pipe(
                image,
                num_inference_steps=30,  # Reduced steps for CPU
                generator=torch.Generator(device=device).manual_seed(42),
                num_frames=14,  # 2 seconds at 7fps
                motion_bucket_id=127,  # High motion
                noise_aug_strength=0.1  # Reduced noise
            ).frames[0]
        
        # Export to video
        export_to_video(frames, output_path, fps=7)
        logging.info(f"Generated video saved to: {output_path}")
        
        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"Failed to process panel {image_path}: {str(e)}")
        raise

def process_all_panels():
    """Process all inpainted panels and generate videos"""
    try:
        # Setup pipeline
        pipe, device = setup_pipeline()
        
        # Create output directory
        output_base = "preprocessor/outputs/animations"
        os.makedirs(output_base, exist_ok=True)
        
        # Find all inpainted panels
        inpainted_pattern = "preprocessor/outputs/inpainted/page_*/panel_*_inpainted.png"
        panel_paths = sorted(glob.glob(inpainted_pattern))
        
        if not panel_paths:
            logging.warning("No inpainted panels found!")
            return
        
        logging.info(f"Found {len(panel_paths)} panels to process")
        
        # Process each panel
        for panel_path in panel_paths:
            # Extract page and panel numbers from the path
            path_parts = panel_path.split('/')
            page_dir = path_parts[-2]  # e.g., "page_4"
            panel_file = path_parts[-1]  # e.g., "panel_0_inpainted.png"
            
            page_num = page_dir.split('_')[1]
            panel_num = panel_file.split('_')[1]
            
            # Create output directory for this page
            page_output_dir = os.path.join(output_base, f"page_{page_num}")
            os.makedirs(page_output_dir, exist_ok=True)
            
            # Generate output video path
            output_path = os.path.join(page_output_dir, f"panel_{panel_num}_animation.mp4")
            
            logging.info(f"Processing {panel_path}")
            process_panel(pipe, device, panel_path, output_path)
            
        logging.info("All panels processed successfully")
        
    except Exception as e:
        logging.error(f"Failed to process panels: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        process_all_panels()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)