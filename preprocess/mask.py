import os
from PIL import Image
import glob
from ultralytics import YOLO
import numpy as np
import logging
from tqdm import tqdm
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_bubble_detector():
    """Sets up the speech bubble detection model using YOLOv8"""
    try:
        model_path = "models/yolov8m_seg-speech-bubble.pt"
        os.makedirs("models", exist_ok=True)
        
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
        return YOLO(model_path)
    except Exception as e:
        logging.error(f"Failed to setup bubble detector: {str(e)}")
        raise

def create_text_bubble_mask(image, model):
    """Creates binary mask for text bubbles"""
    try:
        results = model(image, conf=0.5)[0]
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        
        for box in results.boxes:
            if box.conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.size[0], x2 + padding)
                y2 = min(image.size[1], y2 + padding)
                mask[y1:y2, x1:x2] = 1.0
        
        return mask
    except Exception as e:
        logging.error(f"Failed to create text bubble mask: {str(e)}")
        raise

def process_all_panels():
    """Process all panels and create masks"""
    try:
        bubble_model = setup_bubble_detector()
        panels_dir = Path("data/panels")
        masks_dir = Path("data/masks")
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all panel images
        panel_images = sorted(glob.glob(str(panels_dir / "*.png")))
        
        logging.info(f"Found {len(panel_images)} panels to process")
        
        for panel_path in tqdm(panel_images, desc="Processing panels"):
            try:
                panel_image = Image.open(panel_path).convert('RGB')
                mask = create_text_bubble_mask(panel_image, bubble_model)
                
                # Save mask with same name as panel but with _mask suffix
                mask_path = masks_dir / f"{Path(panel_path).stem}_mask.png"
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(mask_path)
                logging.info(f"Saved mask to {mask_path}")
                
            except Exception as e:
                logging.error(f"Failed to process panel {panel_path}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        process_all_panels()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1)