# Comic Web Scraper

A tool to extract comic images from websites and compile them into a clean PDF without ads or side panels.

## Features

- Scrapes comic images from web pages using a headless browser
- Automatically detects comic panels (or accepts custom CSS selectors)
- Downloads images and compiles them into a PDF
- Works with most comic websites including those with dynamic content
- Removes ads, side panels, and other non-comic content

## Prerequisites

- Python 3.7+
- Chrome or Chromium browser installed
- ChromeDriver that matches your Chrome version (can be installed via webdriver-manager)

## Installation

1. Clone this repository
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Make sure you have Chrome or Chromium installed

## Usage

Run the script with the following command:

```bash
python comic_scraper.py "https://example.com/comic-page-url" --output-dir ./my_comics
```

### Options

- `url`: The URL of the comic page to scrape (required)
- `--output-dir`, `-o`: Directory to save the PDF and temporary files (default: ./output)
- `--wait-time`, `-w`: Time to wait for the page to load in seconds (default: 5)
- `--image-selector`, `-s`: CSS selector for comic images (e.g., ".chapter-container img")
- `--no-cleanup`: Do not delete temporary files after processing

### Examples

Basic usage:
```bash
python comic_scraper.py "https://readcomicsonline.com/comic-name/chapter-1"
```

With custom output directory:
```bash
python comic_scraper.py "https://readcomicsonline.com/comic-name/chapter-1" -o ./my_comics
```

Using a specific CSS selector (for websites where auto-detection fails):
```bash
python comic_scraper.py "https://readcomicsonline.com/comic-name/chapter-1" -s ".chapter-container img"
```

Longer wait time for slow-loading websites:
```bash
python comic_scraper.py "https://readcomicsonline.com/comic-name/chapter-1" -w 10
```

Keep temporary files for inspection:
```bash
python comic_scraper.py "https://readcomicsonline.com/comic-name/chapter-1" --no-cleanup
```

## How It Works

1. **Loading and Parsing**: Uses Selenium WebDriver to load the webpage with JavaScript execution
2. **Comic Image Detection**: Automatically identifies comic panel images or uses a provided CSS selector
3. **Image Download**: Downloads only the comic images, skipping ads and irrelevant content
4. **PDF Creation**: Compiles the downloaded images into a single PDF document

## Troubleshooting

- If no images are found, try increasing the `--wait-time` parameter
- If the wrong images are selected, provide a specific CSS selector with `--image-selector`
- For websites with complex image loading, try using browser developer tools to find the right selector
- Check the log file in the output directory for detailed information

## Limitations

- Requires a web connection
- Performance depends on the website structure and loading speed
- Some websites may use anti-scraping measures that prevent automated access
- Image quality depends on what's available on the website

## Legal Considerations

This tool is provided for educational purposes only. Please:

- Respect website terms of service and robots.txt
- Only download content that you have the right to access
- Consider using rate limiting to avoid overwhelming servers
- Do not use for mass downloading copyrighted content without permission 