# take_screenshot.py
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def take_screenshot(url, output_path):
    """Take a screenshot of a webpage and save it to a file"""
    print(f"Taking screenshot of {url}...")
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1280,800")
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(3)
        
        # Take screenshot
        driver.save_screenshot(output_path)
        print(f"Screenshot saved to {output_path}")
    except Exception as e:
        print(f"Error taking screenshot: {e}")
    finally:
        # Close the driver
        driver.quit()

if __name__ == "__main__":
    # Take a screenshot of the finance agent web interface
    take_screenshot("http://localhost:5000", "screenshot.png")
