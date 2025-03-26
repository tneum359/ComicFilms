#!/usr/bin/env python3
"""
Example usage of the Comic Web Scraper
Simple script to scrape comics from a URL
"""

import os
import sys
import time
import traceback
from PIL import Image
from webscraper import ComicScraper

def main():
    """Simple example of using the comic scraper"""
    # Get URL from command line or use default
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Default example URL
        url = "https://readcomiconline.li/Comic/Ultimate-Spider-Man-2024/Issue-1?id=224448"
        print(f"No URL provided, using example: {url}")
    
    # Set output directory
    output_dir = "inputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scraper with longer wait time for better image loading
    # ReadComicOnline needs more time to properly load
    wait_time = 20
    print(f"Using wait time of {wait_time} seconds to ensure all elements load properly")
    
    try:
        scraper = ComicScraper(
            url=url,
            output_dir=output_dir,
            wait_time=wait_time
        )
        
        # Run the scraper
        print(f"Scraping comic from: {url}")
        print("This may take several minutes for multi-page comics...")
        print("The script will navigate through all pages to capture each image")
        
        start_time = time.time()
        image_paths = scraper.scrape_and_save_images()
        elapsed_time = time.time() - start_time
        
        # Check results
        if image_paths:
            print(f"\nSuccess! Image created: {image_paths}")
            print(f"Time taken: {elapsed_time:.2f} seconds")
            try:
                # Count the number of image files that were combined
                temp_dir = os.path.join(output_dir, "temp_images")
                if os.path.exists(temp_dir):
                    files = [f for f in os.listdir(temp_dir) if f.endswith(('.jpg', '.png', '.webp'))]
                    print(f"Total pages saved: {len(files)}")
                else:
                    print("Image created successfully, but couldn't determine page count")
            except Exception as e:
                print(f"Image created successfully. Error checking page count: {e}")
            return 0
        else:
            print("\nFailed to create image. Check the log for details.")
            print(f"Time taken: {elapsed_time:.2f} seconds")
            return 1
    except Exception as e:
        print(f"\nError during scraping process: {e}")
        print("Detailed error information:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 