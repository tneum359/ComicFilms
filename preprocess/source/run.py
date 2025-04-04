#!/usr/bin/env python3
"""
Main pipeline for comic processing:
1. Web scraping
2. Panel extraction
3. Mask creation
4. Inpainting
5. Video generation
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from PIL import Image

# Add the source directory to the path if needed
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from webscraper import ComicScraper
from panel_extraction import extract_panels
from mask import create_text_bubble_mask, setup_bubble_detector
from inpainting import setup_inpainting_pipeline, inpaint_panel, process_all_panels
from video_gen import generate_video


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_starting_step():
    """Prompt user for which step to start the pipeline from"""
    steps = {
        1: "Web Scraping",
        2: "Panel Extraction",
        3: "Mask Creation",
        4: "Inpainting",
        5: "Video Generation"
    }
    
    print("\n======== Comic Processing Pipeline ========")
    print("Available steps:")
    for step_num, step_name in steps.items():
        print(f"{step_num}. {step_name}")
    
    while True:
        try:
            step = int(input("\nEnter the step number to start from (1-5): "))
            if 1 <= step <= 5:
                return step
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")

def check_required_data(step):
    """Check if required data is available for starting at a given step"""
    if step == 1:
        return True
    
    if step == 2:
        # Check if comic pages exist
        pages_dir = Path("data/pages")
        if not pages_dir.exists() or not list(pages_dir.glob("*.png")):
            logger.error("No comic pages found in data/pages. Please run step 1 first.")
            return False
    
    if step == 3:
        # Check if panels exist
        panels_dir = Path("data/panels")
        if not panels_dir.exists() or not list(panels_dir.glob("*.png")):
            logger.error("No panels found in data/panels. Please run step 2 first.")
            return False
    
    if step == 4:
        # Check if masks exist
        masks_dir = Path("data/masks")
        if not masks_dir.exists() or not list(masks_dir.glob("*_mask.png")):
            logger.error("No masks found in data/masks. Please run step 3 first.")
            return False
    
    if step == 5:
        # Check if inpainted panels exist
        inpainted_dir = Path("data/inpainted")
        if not inpainted_dir.exists() or not list(inpainted_dir.glob("*_inpainted.png")):
            logger.error("No inpainted panels found in data/inpainted. Please run step 4 first.")
            return False
    
    return True

def setup_directories(step):
    """Create necessary directories starting from the given step"""
    base_dir = Path("data")
    directories = []
    
    if step <= 1:
        directories.append(base_dir / "pages")
    if step <= 2:
        directories.append(base_dir / "panels")
    if step <= 3:
        directories.append(base_dir / "masks")
    if step <= 4:
        directories.append(base_dir / "inpainted")
    if step <= 5:
        directories.append(base_dir / "videos")
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main pipeline execution"""
    try:
        # Ask which step to start from
        starting_step = get_starting_step()
        
        # Setup directories needed for the selected steps
        setup_directories(starting_step)
        
        # Check if required data is available
        if not check_required_data(starting_step):
            return
        
        # Step 1: Web Scraping
        if starting_step == 1:
            print("\n======== Step 1: Web Scraping ========")
            url = input("Enter the comic URL (e.g. https://readcomiconline.li/Comic/Title/Issue-1): ")
            if not url.startswith("http"):
                print("Invalid URL format. URL must start with http:// or https://")
                url = input("Please enter a valid comic URL: ")
            
            # Configure and run the scraper
            scraper = ComicScraper(url=url, output_dir="data/pages", wait_time=5)
            logger.info("1) -------------------Starting web scraping...")
            image_paths = scraper.scrape_and_save_images()
            if not image_paths:
                logger.error("Failed to scrape comic images")
                return
        
        # Step 2: Panel Extraction
        if starting_step <= 2:
            logger.info("2) -------------------Starting panel extraction...")
            panel_paths = extract_panels("data/pages", "data/panels")
            if not panel_paths:
                logger.error("Failed to extract panels")
                return
        
        # Step 3: Mask Creation
        if starting_step <= 3:
            logger.info("3) -------------------Starting mask creation...")
            bubble_model = setup_bubble_detector()
            
            # Get panel files and process each one
            panels_dir = Path("data/panels")
            panel_paths = list(panels_dir.glob("*.png"))
            if not panel_paths:
                logger.error("No panel images found")
                return
                
            for panel_path in panel_paths:
                panel_image = Image.open(panel_path).convert('RGB')
                mask = create_text_bubble_mask(panel_image, bubble_model)
                mask_path = Path("data/masks") / f"{Path(panel_path).stem}_mask.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        
        # Step 4: Inpainting - MODIFIED to use process_all_panels
        if starting_step <= 4:
            logger.info("4) -------------------Starting inpainting...")
            # Use process_all_panels instead of manually looping
            process_all_panels()
        
        # Step 5: Video Generation
        if starting_step <= 5:
            logger.info("5) -------------------Starting video generation...")
            generate_video("data/inpainted", "data/videos")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 