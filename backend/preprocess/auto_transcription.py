import os
import numpy as np
from PIL import Image
import torch
import json
from pathlib import Path
from transformers import AutoModelForVision2Seq


# Define input and output paths
input_dir = Path("inputs/images")
output_dir = Path("outputs/transcripts")
panels_dir = Path("outputs/panels")
output_dir.mkdir(parents=True, exist_ok=True)
panels_dir.mkdir(parents=True, exist_ok=True)

# Define custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# Monkey patch json.dumps to use our custom encoder
original_dumps = json.dumps
json.dumps = lambda *args, **kwargs: original_dumps(*args, cls=CustomJSONEncoder, **kwargs)

def read_image(path_to_image):
    with open(path_to_image, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

def crop_panel(color_image, bbox, page_num, panel_num):
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
        print(f"Warning: Invalid bounding box for panel {panel_num} on page {page_num}")
        return None
    
    # Crop the panel
    panel = color_image[y1:y2, x1:x2]
    panel_image = Image.fromarray(panel)
    
    # Create page-specific directory for panels
    page_panels_dir = panels_dir / f"page_{page_num:03d}"
    page_panels_dir.mkdir(exist_ok=True)
    
    # Save the panel
    panel_path = page_panels_dir / f"panel_{panel_num:03d}.png"
    panel_image.save(panel_path)
    return panel_path

def save_panel_transcript(page_num, panel_num, text, speaker="unsure"):
    """Save transcript for a specific panel."""
    # Create page-specific directory for transcripts
    page_transcript_dir = output_dir / f"page_{page_num:03d}"
    page_transcript_dir.mkdir(exist_ok=True)
    
    # Save the panel transcript
    transcript_path = page_transcript_dir / f"panel_{panel_num:03d}.txt"
    with open(transcript_path, "w") as fh:
        fh.write(f"<{speaker}>: {text}\n")
    return transcript_path

# Find all available image files
image_paths = []
for i in range(1, 101):  # Try pages 1-100
    image_path = input_dir / f"Comic_page_{i:03d}.png"
    if image_path.exists():
        image_paths.append(image_path)
    else:
        if i > 1:
            print(f"Stopping at page {i-1} (couldn't find {image_path})")
        break

if not image_paths:
    print(f"No image files found in {input_dir} matching the pattern Comic_page_NNN.png")
    exit(1)

print(f"Found {len(image_paths)} images to process")

try:
    # Load character bank if available (optional)
    character_bank = None
    char_images_dir = Path("inputs/characters")
    if char_images_dir.exists() and any(char_images_dir.iterdir()):
        character_bank = {
            "images": [],
            "names": []
        }
        
        # First collect paths and names
        for char_img in char_images_dir.glob("*.png"):
            character_bank["images"].append(str(char_img))
            character_bank["names"].append(char_img.stem)
        print(f"Found {len(character_bank['names'])} characters")
        
        # Then load all images
        character_bank["images"] = [read_image(x) for x in character_bank["images"]]
        print("Loaded all character images")
    
    # Load comic pages
    chapter_pages = [read_image(str(x)) for x in image_paths]  # grayscale for model
    color_pages = [np.array(Image.open(str(x))) for x in image_paths]  # color for cropping
    
    # Import the model
    print("Loading MAGI model...")
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
        if torch.cuda.is_available():
            model = model.cuda()
        model = model.eval()
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    print(f"Processing {len(chapter_pages)} comic pages...")
    
    # Process batch and detect characters if available
    with torch.no_grad():
        if character_bank:
            # Do chapter-wide prediction with character recognition
            print("Using character recognition for transcription")
            per_page_results = model.do_chapter_wide_prediction(
                chapter_pages, 
                character_bank, 
                use_tqdm=True, 
                do_ocr=True
            )
            
            # Create transcript with character names
            transcript = []
            for i, (gray_image, color_image, page_result) in enumerate(zip(chapter_pages, color_pages, per_page_results)):
                # Save visualization
                page_num = int(image_paths[i].stem.split('_')[-1])
                viz_path = output_dir / f"image_{page_num:03d}.png"
                model.visualise_single_image_prediction(gray_image, page_result, filename=str(viz_path))
                
                # Map text to character names
                speaker_name = {
                    text_idx: page_result["character_names"][char_idx] 
                    for text_idx, char_idx in page_result["text_character_associations"]
                }
                
                # Extract and save panels using color image
                if "panels" in page_result:
                    print(f"Extracting panels from page {page_num}...")
                    for panel_idx, panel_bbox in enumerate(page_result["panels"]):
                        # Save the panel image
                        panel_path = crop_panel(color_image, panel_bbox, page_num, panel_idx)
                        if panel_path:  # Only proceed if panel was successfully cropped
                            print(f"Saved panel {panel_idx:03d} from page {page_num:03d}")
                            
                            # Find text that belongs to this panel
                            panel_texts = []
                            for text_idx, text_bbox in enumerate(page_result["texts"]):
                                # Check if text is within panel bounds
                                if (text_bbox[0] >= panel_bbox[0] and text_bbox[2] <= panel_bbox[2] and
                                    text_bbox[1] >= panel_bbox[1] and text_bbox[3] <= panel_bbox[3]):
                                    if page_result["is_essential_text"][text_idx]:
                                        name = speaker_name.get(text_idx, "unsure")
                                        panel_texts.append(f"<{name}>: {page_result['ocr'][text_idx]}")
                        
                        # Save panel-specific transcript
                        if panel_texts:
                            save_panel_transcript(page_num, panel_idx, "\n".join(panel_texts))
                
                # Create full page transcript
                page_transcript = []
                for j in range(len(page_result["ocr"])):
                    if not page_result["is_essential_text"][j]:
                        continue
                    name = speaker_name.get(j, "unsure")
                    page_transcript.append(f"<{name}>: {page_result['ocr'][j]}")
                
                # Save page-specific transcript
                transcript_path = output_dir / f"transcript_{page_num:03d}.txt"
                with open(transcript_path, "w") as fh:
                    for line in page_transcript:
                        fh.write(line + "\n")
                transcript.extend(page_transcript)
                
                print(f"Processed page {page_num}")
            
            # Save full transcript
            with open(output_dir / "full_transcript.txt", "w") as fh:
                for line in transcript:
                    fh.write(line + "\n")
        else:
            # Basic panel and text detection without character recognition
            print("Using basic text detection (no character recognition)")
            # Process in batches to avoid memory issues
            batch_size = 1
            for i in range(0, len(chapter_pages), batch_size):
                batch_gray_images = chapter_pages[i:i+batch_size]
                batch_color_images = color_pages[i:i+batch_size]
                batch_paths = image_paths[i:i+batch_size]
                
                print(f"Processing batch {i//batch_size + 1}/{(len(chapter_pages) + batch_size - 1)//batch_size}")
                
                try:
                    # Use predict_detections_and_associations if do_chapter_wide_prediction is not available
                    results = model.predict_detections_and_associations(batch_gray_images)
                    text_bboxes = [x["texts"] for x in results]
                    ocr_results = model.predict_ocr(batch_gray_images, text_bboxes)
                    
                    # Save results for each image in batch
                    for j, (gray_image, color_image, result, ocr, img_path) in enumerate(zip(batch_gray_images, batch_color_images, results, ocr_results, batch_paths)):
                        # Extract page number for output filenames
                        page_num = int(img_path.stem.split('_')[-1])
                        
                        # Save visualization
                        viz_path = output_dir / f"image_{page_num:03d}.png"
                        model.visualise_single_image_prediction(gray_image, result, filename=str(viz_path))
                        
                        # Extract and save panels using color image
                        if "panels" in result:
                            print(f"Extracting panels from page {page_num}...")
                            for panel_idx, panel_bbox in enumerate(result["panels"]):
                                # Save the panel image
                                panel_path = crop_panel(color_image, panel_bbox, page_num, panel_idx)
                                if panel_path:  # Only proceed if panel was successfully cropped
                                    print(f"Saved panel {panel_idx:03d} from page {page_num:03d}")
                                    
                                    # Find text that belongs to this panel
                                    panel_texts = []
                                    for text_idx, text_bbox in enumerate(result["texts"]):
                                        # Check if text is within panel bounds
                                        if (text_bbox[0] >= panel_bbox[0] and text_bbox[2] <= panel_bbox[2] and
                                            text_bbox[1] >= panel_bbox[1] and text_bbox[3] <= panel_bbox[3]):
                                            # In basic mode, include all text within panel bounds
                                            panel_texts.append(f"<unsure>: {ocr[text_idx]}")
                                    
                                    # Save panel-specific transcript
                                    if panel_texts:
                                        save_panel_transcript(page_num, panel_idx, "\n".join(panel_texts))
                        
                        # Save page transcript
                        transcript_path = output_dir / f"transcript_{page_num:03d}.txt"
                        model.generate_transcript_for_single_image(result, ocr, filename=str(transcript_path))
                        
                        print(f"Processed page {page_num}")
                except Exception as batch_error:
                    print(f"Error in batch processing: {batch_error}")
                    # Try to free memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
    
    print("Processing complete!")

except Exception as e:
    print(f"Error: {e}")
    

# Restore original json.dumps
json.dumps = original_dumps