#!/usr/bin/env python3
"""
Comic Web Scraper - Extracts comic images from readcomiconline.
"""

import time
import logging
from pathlib import Path
from urllib.parse import urlparse, urljoin
import re
import html

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from PIL import Image

class ComicScraper:
    """Scrapes comic images from readcomiconline and saves them as PNG files."""
    
    def __init__(self, url, output_dir="data/pages", wait_time=3):
        """Initialize the comic scraper."""
        self.url = url
        self.output_dir = Path(output_dir)
        self.wait_time = wait_time
        self.temp_dir = self.output_dir / "temp_images"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ComicScraper")
        
        # Get comic name from URL
        self.comic_name = self._get_comic_name_from_url(url)
    
    def _get_comic_name_from_url(self, url):
        """Extract comic name from URL."""
        path = urlparse(url).path
        parts = [p for p in path.split('/') if p]
        
        if len(parts) >= 1:
            # Add issue number to comic name if available
            if len(parts) >= 3 and parts[1].lower() == 'issue':
                return f"{parts[0]}_{parts[2]}".replace('-', '_')
            return parts[0].replace('-', '_')
        else:
            return urlparse(url).netloc.split('.')[0]
    
    def scrape_and_save_images(self):
        """Scrape comic images and save them as PNG files."""
        # Set up driver with basic options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Load the webpage
            self.logger.info(f"Loading webpage: {self.url}")
            driver.get(self.url)
            
            # Wait for page to load
            time.sleep(self.wait_time)
            
            # Try to select high quality if quality dropdown exists
            try:
                quality_selector = "#selectQuality"
                quality_dropdown = driver.find_element(By.CSS_SELECTOR, quality_selector)
                if quality_dropdown:
                    options = quality_dropdown.find_elements(By.TAG_NAME, "option")
                    if options:
                        # Select high quality (last option)
                        quality_value = options[-1].get_attribute("value")
                        driver.execute_script(f"document.querySelector('{quality_selector}').value='{quality_value}';")
                        driver.execute_script(f"document.querySelector('{quality_selector}').dispatchEvent(new Event('change'));")
                        time.sleep(1)
                        self.logger.info("Set quality to high")
            except:
                self.logger.info("Quality selector not found or couldn't change quality")
            
            # Check if there's a page selector
            image_urls = []
            try:
                page_selector = driver.find_element(By.CSS_SELECTOR, "#selectPage")
                options = page_selector.find_elements(By.TAG_NAME, "option")
                
                if options:
                    page_count = len(options)
                    self.logger.info(f"Found {page_count} pages using page selector")
                    
                    # Get all pages
                    for i, option in enumerate(options):
                        try:
                            self.logger.info(f"Processing page {i+1}/{page_count}")
                            
                            # Select page using JavaScript
                            page_value = option.get_attribute("value")
                            driver.execute_script(f"""
                                const select = document.querySelector('#selectPage');
                                if(select) {{
                                    select.value = '{page_value}';
                                    select.dispatchEvent(new Event('change'));
                                }}
                            """)
                            
                            # Wait for image to load - increased wait time
                            time.sleep(4)
                            
                            # EXACTLY follow user instructions for extracting comic images
                            page_urls = driver.execute_script("""
                                // 1. Find the div#divImage container
                                const divImage = document.querySelector('#divImage');
                                if (!divImage) {
                                    console.log("div#divImage not found");
                                    return [];
                                }
                                
                                // 2. Find all img tags inside div#divImage
                                const imgElements = divImage.querySelectorAll('img');
                                console.log("Found " + imgElements.length + " images in div#divImage");
                                
                                const imageUrls = [];
                                
                                // 3. Process each image element
                                for (const img of imgElements) {
                                    // Print debugging info about each image
                                    console.log("Image source: " + img.src);
                                    console.log("Image has style: " + (img.getAttribute('style') ? "yes" : "no"));
                                    
                                    // Get direct src attribute for non-transparent images
                                    if (img.src && !img.src.includes('trans.png') && !img.src.includes('loading.gif')) {
                                        console.log("Found main comic image src: " + img.src);
                                        imageUrls.push(img.src);
                                    }
                                    
                                    // If it's a transparent image, check for background-image in style
                                    if (img.src && img.src.includes('trans.png')) {
                                        const style = img.getAttribute('style');
                                        if (style && style.includes('background-image')) {
                                            // Extract the URL from background-image style
                                            const match = style.match(/background-image: ?url\\(['"]?(.*?)['"]?\\)/i);
                                            if (match && match[1]) {
                                                console.log("Found background image in transparent overlay: " + match[1]);
                                                imageUrls.push(match[1].replace(/&amp;/g, '&').replace(/&quot;/g, '"'));
                                            }
                                        }
                                        
                                        // Also check computed style
                                        const computedStyle = window.getComputedStyle(img);
                                        const bgImage = computedStyle.getPropertyValue('background-image');
                                        if (bgImage && bgImage !== 'none') {
                                            const match = bgImage.match(/url\\(['"]?(.*?)['"]?\\)/i);
                                            if (match && match[1]) {
                                                console.log("Found computed background image: " + match[1]);
                                                imageUrls.push(match[1]);
                                            }
                                        }
                                    }
                                }
                                
                                // If still no images found, try to find any link with rel="image_src"
                                if (imageUrls.length === 0) {
                                    const linkElement = document.querySelector('link[rel="image_src"]');
                                    if (linkElement && linkElement.href) {
                                        console.log("Found image_src link: " + linkElement.href);
                                        imageUrls.push(linkElement.href);
                                    }
                                }
                                
                                // Print summary
                                console.log("Total image URLs found: " + imageUrls.length);
                                return imageUrls;
                            """)
                            
                            if page_urls and len(page_urls) > 0:
                                for url in page_urls:
                                    if url and url not in image_urls:
                                        # Clean URL: decode HTML entities
                                        cleaned_url = html.unescape(url)
                                        image_urls.append(cleaned_url)
                                
                                self.logger.info(f"Found {len(page_urls)} images on page {i+1}")
                            else:
                                self.logger.warning(f"No images found on page {i+1}, trying fallback method")
                                
                                # Fallback: crawl the entire page source for blogspot image URLs
                                page_source = driver.page_source
                                blogspot_pattern = r'(https?://[^"\'\s]+\.bp\.blogspot\.com/[^"\'\s]+)'
                                blogspot_matches = re.findall(blogspot_pattern, page_source)
                                
                                # Filter to likely comic page images (exclude small thumbnails and icons)
                                for url in blogspot_matches:
                                    # Clean the URL (remove HTML entities)
                                    url = html.unescape(url)
                                    
                                    # Skip URLs that are likely not comic images
                                    if 'loading.gif' in url or 'w=1' in url:
                                        continue
                                    
                                    if url not in image_urls:
                                        image_urls.append(url)
                                        self.logger.info(f"Found image via page source on page {i+1}")
                                        break
                        except Exception as e:
                            self.logger.error(f"Error processing page {i+1}: {e}")
                    
                    self.logger.info(f"Found {len(image_urls)} unique images using page selector")
                
            except Exception as e:
                self.logger.info(f"Page selector navigation failed: {e}")
            
            # Filter and deduplicate URLs
            image_urls = [url for url in image_urls if url and (url.startswith('http://') or url.startswith('https://'))]
            image_urls = list(dict.fromkeys(image_urls))
            
            if not image_urls:
                self.logger.error("No comic images found")
                return None
            
            self.logger.info(f"Found {len(image_urls)} images to download")
            
            # Download images
            image_paths = self._download_images(image_urls)
            
            if not image_paths:
                self.logger.error("Failed to download any valid images")
                return None
                
            # Convert to PNG and save to output directory
            png_paths = self._save_images_as_png(image_paths)
            
            return png_paths
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
        finally:
            driver.quit()
            
            # Clean up temp files
            try:
                for file in self.temp_dir.iterdir():
                    file.unlink()
                self.temp_dir.rmdir()
            except:
                pass
    
    def _download_images(self, image_urls):
        """Download images from URLs."""
        image_paths = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': self.url
        }
        
        for i, url in enumerate(image_urls, 1):
            try:
                self.logger.info(f"Downloading image {i}/{len(image_urls)}: {url[:60]}...")
                
                # Add cache busting parameter to avoid cached images
                if '?' in url:
                    url += f"&cb={int(time.time())}"
                else:
                    url += f"?cb={int(time.time())}"
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Log image size
                self.logger.info(f"Image size: {len(response.content)} bytes")
                
                # Don't skip based on size - we'll verify with dimensions instead
                # This is because the actual comic images could have varying sizes
                
                # Determine file extension from content type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                elif 'png' in content_type:
                    ext = 'png'
                elif 'webp' in content_type:
                    ext = 'webp'
                else:
                    ext = 'jpg'  # Default
                
                # Save the image
                image_path = self.temp_dir / f"image_{i:03d}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify the image
                try:
                    img = Image.open(image_path)
                    img.verify()
                    # Re-open to get dimensions
                    img = Image.open(image_path)
                    width, height = img.size
                    
                    self.logger.info(f"Image dimensions: {width}x{height}")
                    
                    # Skip very small images (likely icons or UI elements)
                    # But use a smaller threshold to avoid missing actual comic pages
                    if width < 300 or height < 300:
                        self.logger.info(f"Skipping small image (dimensions: {width}x{height})")
                        if image_path.exists():
                            image_path.unlink()
                        continue
                    
                    # Valid image
                    image_paths.append(image_path)
                except Exception as e:
                    self.logger.error(f"Invalid image: {e}")
                    if image_path.exists():
                        image_path.unlink()
            
            except Exception as e:
                self.logger.error(f"Error downloading image {i}: {e}")
        
        return image_paths
    
    def _save_images_as_png(self, image_paths):
        """Save images as PNG files directly to output directory."""
        png_paths = []
        for idx, img_path in enumerate(sorted(image_paths, key=lambda x: x.name)):
            png_path = self.output_dir / f"{self.comic_name}_page_{idx+1:03d}.png"
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                        background.save(png_path, 'PNG')
                    else:
                        img.save(png_path, 'PNG')
                    png_paths.append(png_path)
                    self.logger.info(f"Saved image {idx+1} to {png_path}")
            except Exception as e:
                self.logger.error(f"Error saving image as PNG: {e}")
        
        return png_paths


def main():
    """Main entry point for the comic scraper."""
    # Prompt for URL instead of using command line arguments
    url = input("Enter the comic URL (e.g. https://readcomiconline.li/Comic/Title/Issue-1): ")
    
    # Default output directory to data/pages
    output_dir = "data/pages"
    
    # Create scraper and run it
    scraper = ComicScraper(
        url=url,
        output_dir=output_dir,
        wait_time=3
    )
    
    png_paths = scraper.scrape_and_save_images()
    
    if png_paths:
        print(f"\nSuccess! {len(png_paths)} PNG images saved to: {output_dir}")
        return 0
    else:
        print("\nFailed to save images.")
        return 1


if __name__ == "__main__":
    main()