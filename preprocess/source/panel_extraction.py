import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def read_image(path_to_image):
    """Read and convert image to RGB format."""
    with open(path_to_image, "rb") as file:
        image = Image.open(file).convert("RGB")
        image = np.array(image)
    return image

def crop_panel(color_image, bbox, panel_num):
    """Crop a panel from the color image using the bounding box coordinates."""
    # Get image dimensions
    height, width = color_image.shape[:2]
    
    # Convert coordinates to integers and clamp to image boundaries
    x1 = max(0, min(int(bbox[0]), width))
    y1 = max(0, min(int(bbox[1]), height))
    x2 = max(0, min(int(bbox[2]), width))
    y2 = max(0, min(int(bbox[3]), height))
    
    # Check if the bounding box is valid
    if x2 <= x1 or y2 <= y1:
        logging.warning(f"Invalid bounding box for panel {panel_num}")
        return None
    
    # Crop the panel
    panel = color_image[y1:y2, x1:x2]
    panel_image = Image.fromarray(panel)
    
    # Save the panel directly in the panels directory
    panels_dir = Path("data/panels")
    panels_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panels_dir / f"panel_{panel_num:03d}.png"
    panel_image.save(panel_path)
    return panel_path

def load_or_download_model():
    """Load the model from local models directory or download if not available."""
    from transformers import AutoModel
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = models_dir / "magiv2"
    model_exists = model_dir.exists() and list(model_dir.glob("*.bin"))
    
    if model_exists:
        logging.info("Loading MAGI model from local cache...")
        try:
            model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception as e:
            logging.error(f"Error loading from local cache: {e}")
            logging.info("Attempting to download the model instead...")
            model_exists = False
    
    if not model_exists:
        logging.info("Downloading MAGI model...")
        try:
            # Download model
            model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
            
            # Save model to local directory
            model.save_pretrained(str(model_dir))
            logging.info(f"Model saved to {model_dir}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            raise
    
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    logging.info("Model loaded successfully")
    return model

def extract_panels(input_dir, output_dir):
    """Extract panels from comic pages and save them."""
    try:
        # Find all available image files
        image_paths = []
        input_path = Path(input_dir)
        for img_path in sorted(input_path.glob("*.png")):
            image_paths.append(img_path)
        
        if not image_paths:
            logging.error(f"No image files found in {input_dir}")
            return None
        
        logging.info(f"Found {len(image_paths)} images to process")
        
        # Load comic pages
        chapter_pages = [read_image(str(x)) for x in image_paths]  # grayscale for model
        color_pages = [np.array(Image.open(str(x))) for x in image_paths]  # color for cropping
        
        # Import and load the model
        logging.info("Loading MAGI model...")
        try:
            model = load_or_download_model()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
        
        logging.info(f"Processing {len(chapter_pages)} comic pages...")
        panel_paths = []
        
        # Process in batches to avoid memory issues
        batch_size = 1
        global_panel_count = 0  # Global counter for panels across all pages
        
        for i in range(0, len(chapter_pages), batch_size):
            batch_gray_images = chapter_pages[i:i+batch_size]
            batch_color_images = color_pages[i:i+batch_size]
            batch_paths = image_paths[i:i+batch_size]
            
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(chapter_pages) + batch_size - 1)//batch_size}")
            
            try:
                # Use predict_detections_and_associations
                results = model.predict_detections_and_associations(batch_gray_images)
                
                # Save results for each image in batch
                for j, (color_image, result, img_path) in enumerate(zip(batch_color_images, results, batch_paths)):
                    # Extract and save panels using color image
                    if "panels" in result:
                        logging.info(f"Extracting panels from {img_path.name}...")
                        for panel_bbox in result["panels"]:
                            # Save the panel image with global panel count
                            panel_path = crop_panel(color_image, panel_bbox, global_panel_count)
                            if panel_path:  # Only proceed if panel was successfully cropped
                                panel_paths.append(panel_path)
                                logging.info(f"Saved panel {global_panel_count:03d}")
                                global_panel_count += 1  # Increment global counter
                    
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                continue
        
        logging.info(f"Successfully extracted {len(panel_paths)} panels")
        return panel_paths
        
    except Exception as e:
        logging.error(f"Script failed: {e}")
        return None

if __name__ == "__main__":
    try:
        panel_paths = extract_panels("data/pages", "data/panels")
        if panel_paths:
            logging.info(f"Successfully extracted {len(panel_paths)} panels")
        else:
            logging.error("Failed to extract panels")
    except Exception as e:
        logging.error(f"Script failed: {e}")