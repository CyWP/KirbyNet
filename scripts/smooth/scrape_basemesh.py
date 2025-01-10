"""
This script was used to extract all usable meshes (single component) from
thebasemesh.com. Huge thanks to Tim Steer for such a great project.
"""

import os
import requests
import tarfile
import zipfile
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import argparse
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time

import trimesh

def download_file(url, output_dir):
    """Download a file from a URL."""
    try:
        local_filename = os.path.join(output_dir, os.path.basename(urlparse(url).path))
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except:
        print(f"Failed download: {local_filename}")
        return None
    
    print(f"Downloaded: {local_filename}")
    return local_filename

def extract_file(file_path):
    """Extract downloaded file based on its type."""
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
        print(f"Extracted: {file_path}")
    elif file_path.endswith(".tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(os.path.dirname(file_path))
        print(f"Extracted: {file_path}")
    else:
        print(f"File is not an archive: {file_path}")

def delete_non_obj_files(folder_path):
    # Walk through all files in the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is not an OBJ file
            if not file.lower().endswith('.obj'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def delete_multicomponent_obj_files(folder_path):
    # Walk through all files in the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an .obj file
            if file.lower().endswith('.obj'):
                file_path = os.path.join(root, file)
                try:
                    # Load the .obj file using trimesh
                    mesh = trimesh.load_mesh(file_path)
                    
                    # Split the mesh into connected components
                    components = mesh.split()
                    
                    # If there is more than one component, it's made of multiple pieces
                    if len(components) > 1:
                        os.remove(file_path)  # Delete the file if it's made of multiple pieces
                        print(f"Deleted multi-piece .obj file: {file_path}")
                    else:
                        print(f"Single-piece .obj file: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def extract_download_links(url, output_dir=None, num_pages=21):
    """Extract downloadable links and download the files from all pages."""
    # Selenium setup
    pages_visited = 0
    options = Options()
    options.binary_location = "/usr/bin/chromium"  # Path to Chromium binary
    options.add_argument("--headless")  # Optional: Run in headless mode (no GUI)
    options.add_argument("--no-sandbox")  # Recommended for Linux
    options.add_argument("--disable-dev-shm-usage")  # Recommended for Linux

    # Set up Chromedriver service
    service = Service("/usr/bin/chromedriver")  # Path to Chromedriver binary
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        while pages_visited<=num_pages:
            pages_visited += 1
            # Get the current page's content dynamically rendered by Selenium
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Extract all <a> tags with an href attribute
            links = [a['href'] for a in soup.find_all('a', href=True)]

            # Filter for download links (e.g., file extensions)
            download_links = [link for link in links if any(ext in link for ext in ['.zip', '.rar', '.7z', '.stl', '.tar.gz'])]
            
            print(f"Found {len(download_links)} downloadable link(s) on current page:")
            for link in download_links:
                print(f"Downloading: {link}")
                file_path = download_file(link, output_dir)
                if file_path:
                    extract_file(file_path)

            # Click the "Next" button using Selenium
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, "[aria-label='Next Page']")
                ActionChains(driver).move_to_element(next_button).click(next_button).perform()
                time.sleep(2)  # Wait for the page to load after clicking
                print("Clicked 'Next', moving to the next page...")
            except Exception as e:
                print("No more 'Next' button found or an error occurred:", e)
                break
    finally:
        driver.quit()

def main():
    output_dir = os.path.abspath("./basemesh")
    url = "https://www.thebasemesh.com/model-library"
    os.makedirs(output_dir, exist_ok=True)

    extract_download_links(url, output_dir, 21)
    delete_non_obj_files(output_dir)
    delete_multicomponent_obj_files(output_dir)

if __name__ == "__main__":
    main()