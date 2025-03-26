#!/usr/bin/env python3
"""
Comic Web Scraper - Extracts comic images from websites and creates a PDF.
Uses automatic image selector detection with a generic, unified approach.
"""

import os
import time
import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import img2pdf

class ComicScraper:
    """Scrapes comic images from websites and saves them as PNG files."""
    
    def __init__(self, url, output_dir, wait_time=5):
        """Initialize the comic scraper."""
        self.url = url
        self.output_dir = Path(output_dir)
        self.wait_time = wait_time
        self.temp_dir = self.output_dir / "temp_images"
        self.images_dir = self.output_dir / "images"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
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
        # Set up driver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")  # Larger window size
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation
        chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute CDP commands to hide WebDriver usage
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
            """
        })
        
        try:
            # Load the webpage
            self.logger.info(f"Loading webpage: {self.url}")
            driver.get(self.url)
            
            # Wait for page to load
            time.sleep(self.wait_time)
            
            # Try to select high quality if a quality dropdown exists
            try:
                # Look for common quality selector patterns
                quality_selectors = ["#selectQuality", ".quality-selector", "select[name*='quality']"]
                for selector in quality_selectors:
                    try:
                        quality_dropdown = WebDriverWait(driver, 3).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if quality_dropdown:
                            # Try to set to highest quality
                            options = quality_dropdown.find_elements(By.TAG_NAME, "option")
                            if options:
                                # Look for high quality option (usually last in the list)
                                quality_value = options[-1].get_attribute("value")
                                driver.execute_script(f"document.querySelector('{selector}').value='{quality_value}';")
                                driver.execute_script(f"document.querySelector('{selector}').dispatchEvent(new Event('change'));")
                                time.sleep(2)  # Wait for quality change to apply
                                self.logger.info(f"Set quality using selector: {selector}")
                                break
                    except:
                        continue
            except Exception as e:
                self.logger.info(f"Couldn't set quality: {e}")
            
            # Unified approach to getting images
            image_urls = []
            
            # Method 1: Try to get all images at once using various selectors
            self.logger.info("Attempting to find comic images with direct selectors")
            direct_urls = self._try_direct_image_selectors(driver)
            if direct_urls:
                image_urls.extend(direct_urls)
            
            # Method 2: Try navigation through pages if needed
            if len(image_urls) < 3:
                self.logger.info("Few images found with direct selectors, trying navigation...")
                page_urls = self._try_navigation_approach(driver)
                if page_urls:
                    # If navigation found more images, use those instead
                    if len(page_urls) > len(image_urls):
                        image_urls = page_urls
                    else:
                        image_urls.extend(page_urls)
            
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
    
    def _try_direct_image_selectors(self, driver):
        """Try various selectors to find comic images directly."""
        # Common selectors for comics across different sites
        selectors = [
            "#divImage img",           # common comic container
            ".chapter-container img",  # many manga sites
            ".comic-container img",    # common comic layout
            ".comic img",              # simpler comic sites
            ".comic-page img",         # comic page layout
            "#comic img",              # common comic ID
            "article img",             # article-based comics
            ".item-article img",       # item-based articles
            ".page img",               # page-based layout
            "#content img",            # content-based layout
            ".chapter-img",            # chapter-based layout
            "img.chapter-img",         # chapter img class
            ".viewer-img",             # viewer-based layout
            ".picture",                # picture layout
            "#viewer img",             # viewer ID
            ".container img",          # general container
            "div[id*='image'] img",    # any div with image in id
            "div[class*='image'] img", # any div with image in class
            "div[id*='comic'] img",    # any div with comic in id
            "div[class*='comic'] img", # any div with comic in class
            "div[id*='page'] img",     # any div with page in id
            "div[class*='page'] img",  # any div with page in class
            "img[id*='image']",        # direct image with image in id
            "img[class*='image']",     # direct image with image in class
            "img[id*='comic']",        # direct image with comic in id
            "img[class*='comic']",     # direct image with comic in class
            "img[id*='page']",         # direct image with page in id
            "img[class*='page']"       # direct image with page in class
        ]
        
        image_urls = []
        
        # Try each selector
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    urls = [img.get_attribute('src') for img in elements if self._is_likely_comic_image(img)]
                    if urls:
                        self.logger.info(f"Found {len(urls)} images with selector: {selector}")
                        image_urls.extend(urls)
            except Exception as e:
                self.logger.debug(f"Error with selector {selector}: {e}")
        
        # If no good selectors, look for any large images
        if not image_urls:
            self.logger.info("No images found with common selectors, looking for large images")
            all_images = driver.find_elements(By.TAG_NAME, "img")
            for img in all_images:
                if self._is_likely_comic_image(img):
                    src = img.get_attribute('src')
                    if src:
                        image_urls.append(src)
        
        return image_urls
    
    def _is_likely_comic_image(self, img_element):
        """Check if an image is likely to be a comic panel based on size and attributes."""
        try:
            # Check size
            width = img_element.size['width']
            height = img_element.size['height']
            src = img_element.get_attribute('src')
            
            # Skip small images, ads, icons, etc.
            if width < 300 or height < 300:
                return False
                
            # Skip UI elements by URL pattern
            if src and any(pattern in src.lower() for pattern in [
                '/content/images/', 'discord.svg', 'user-small', 'read.png', 
                'previous.png', 'next.png', 'error.png', 'adb.png', 'ad.', 
                'banner', 'logo', 'icon'
            ]):
                return False
                
            # Check for comic-related attributes or classes
            class_name = img_element.get_attribute('class') or ''
            alt_text = img_element.get_attribute('alt') or ''
            
            comic_indicators = ['comic', 'panel', 'page', 'chapter', 'manga', 'image']
            for indicator in comic_indicators:
                if (indicator in class_name.lower() or 
                    indicator in alt_text.lower() or 
                    indicator in src.lower()):
                    return True
            
            # If it's large and doesn't match skip patterns, it's likely a comic
            return True
            
        except:
            return False  # Skip problematic elements
    
    def _try_navigation_approach(self, driver):
        """Try to navigate through pages to get all images."""
        all_images = []
        original_url = driver.current_url
        current_page = 1
        max_pages = 100  # Safety limit
        
        # Try to find page selector if exists
        try:
            page_selectors = ["#selectPage", ".page-select", "#pageSelector", 
                             "select[name*='page']", "select[id*='page']", "select[class*='page']"]
            for selector in page_selectors:
                try:
                    page_selector = driver.find_element(By.CSS_SELECTOR, selector)
                    options = page_selector.find_elements(By.TAG_NAME, "option")
                    if options:
                        max_pages = len(options)
                        self.logger.info(f"Detected {max_pages} pages using page selector")
                        
                        # For each page option, select it and get the image
                        for i, option in enumerate(options, 1):
                            try:
                                self.logger.info(f"Processing page selector option {i}/{max_pages}")
                                
                                # Select the page using JavaScript
                                page_value = option.get_attribute("value")
                                driver.execute_script(f"document.querySelector('{selector}').value='{page_value}';")
                                driver.execute_script(f"document.querySelector('{selector}').dispatchEvent(new Event('change'));")
                                
                                # Wait for image to load
                                time.sleep(2)
                                
                                # Find the image on this page
                                found_image = False
                                for img_selector in ["#divImage img", ".chapter-img", ".comic-img", 
                                                    "div[id*='image'] img", "div[class*='image'] img"]:
                                    try:
                                        img_elem = driver.find_element(By.CSS_SELECTOR, img_selector)
                                        if img_elem and self._is_likely_comic_image(img_elem):
                                            src = img_elem.get_attribute('src')
                                            if src and src not in all_images:
                                                all_images.append(src)
                                                found_image = True
                                                break
                                    except:
                                        continue
                                
                                if not found_image:
                                    # If no specific selectors worked, try all large images
                                    all_img_elements = driver.find_elements(By.TAG_NAME, "img")
                                    for img in all_img_elements:
                                        if self._is_likely_comic_image(img):
                                            src = img.get_attribute('src')
                                            if src and src not in all_images:
                                                all_images.append(src)
                                                found_image = True
                            except Exception as e:
                                self.logger.error(f"Error processing page option {i}: {e}")
                        
                        # If we found images using the page selector, return them
                        if all_images:
                            return all_images
                except:
                    continue
        except Exception as e:
            self.logger.info(f"No page selector found or error: {e}")
        
        # Reset to original url if we tried page selectors but didn't find images
        if driver.current_url != original_url:
            driver.get(original_url)
            time.sleep(2)
        
        # Try navigation through next page links/buttons
        self.logger.info("Trying navigation through next page buttons")
        current_page = 1
        
        while current_page <= max_pages:
            self.logger.info(f"Processing navigation page {current_page}")
            
            # Find the current page's image using various selectors
            page_image_found = False
            for selector in ["#divImage img", ".page-img", ".comic-page img", "#comic img", 
                            "div[id*='image'] img", "div[class*='image'] img"]:
                try:
                    img_elem = driver.find_element(By.CSS_SELECTOR, selector)
                    if img_elem and self._is_likely_comic_image(img_elem):
                        src = img_elem.get_attribute('src')
                        if src and src not in all_images:
                            all_images.append(src)
                            page_image_found = True
                            break
                except:
                    continue
            
            if not page_image_found:
                # If no specific selector worked, try all large images
                all_img_elements = driver.find_elements(By.TAG_NAME, "img")
                for img in all_img_elements:
                    if self._is_likely_comic_image(img):
                        src = img.get_attribute('src')
                        if src and src not in all_images:
                            all_images.append(src)
                            page_image_found = True
            
            # Try to go to next page
            next_clicked = False
            next_selectors = ["#nextLink", ".next", "a.next", "[rel='next']", ".next-page", 
                             "#next", "a[title='Next Page']", "a[title*='Next']", 
                             "a[href*='next']", "a:contains('Next')", "a span:contains('Next')",
                             "button:contains('Next')", "button[aria-label*='Next']"]
            
            for next_selector in next_selectors:
                try:
                    next_elements = driver.find_elements(By.CSS_SELECTOR, next_selector)
                    for next_link in next_elements:
                        if next_link and next_link.is_displayed():
                            # Check if it contains text with "next" case insensitive
                            link_text = next_link.text.lower()
                            if 'next' in link_text or link_text == '' or link_text == '>':
                                next_link.click()
                                time.sleep(2)  # Wait for page to load
                                
                                # Check if URL changed to verify navigation worked
                                if driver.current_url != original_url and current_page == 1:
                                    original_url = driver.current_url  # Update for further comparisons
                                    
                                current_page += 1
                                next_clicked = True
                                break
                    
                    if next_clicked:
                        break
                except:
                    continue
            
            if not next_clicked:
                self.logger.info("No next page found, ending navigation")
                break
        
        return all_images
    
    def _download_images(self, image_urls):
        """Download images from URLs."""
        image_paths = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': self.url
        }
        
        for i, url in enumerate(image_urls, 1):
            try:
                self.logger.info(f"Downloading image {i}/{len(image_urls)}")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
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
                    image_paths.append(image_path)
                except Exception as e:
                    self.logger.error(f"Invalid image: {e}")
                    if image_path.exists():
                        image_path.unlink()
            
            except Exception as e:
                self.logger.error(f"Error downloading image: {e}")
        
        return image_paths
    
    def _save_images_as_png(self, image_paths):
        """Save images as PNG files."""
        png_paths = []
        for idx, img_path in enumerate(sorted(image_paths, key=lambda x: x.name)):
            png_path = self.images_dir / f"{self.comic_name}_page_{idx+1:03d}.png"
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
    parser = argparse.ArgumentParser(description='Scrape comic images from a website and save them as PNG files.')
    parser.add_argument('url', help='URL of the comic page to scrape')
    parser.add_argument('--output-dir', '-o', default='./output', help='Output directory')
    parser.add_argument('--wait-time', '-w', type=int, default=5, help='Wait time in seconds')
    
    args = parser.parse_args()
    
    scraper = ComicScraper(
        url=args.url,
        output_dir=args.output_dir,
        wait_time=args.wait_time
    )
    
    png_paths = scraper.scrape_and_save_images()
    
    if png_paths:
        print(f"\nSuccess! {len(png_paths)} PNG images saved to: {scraper.images_dir}")
        return 0
    else:
        print("\nFailed to save images.")
        return 1


if __name__ == "__main__":
    main() 