#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

folder_url = "https://drive.google.com/drive/folders/1mp2JmisBz6vUiaBRU4zSl9CZSRe1mr_I?usp=drive_link"

def install_gdown():
    """Ensure gdown is installed."""
    try:
        import gdown  # noqa
    except ImportError:
        print("gdown is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def download_folder(url, output_dir):
   # Ensure gdown is installed
    install_gdown()

    # Run gdown command to download the folder
    try:
        print(f"Downloading Google Drive folder from: {url}")
        command = ["gdown", "--folder", url, "-O", output_dir]
        subprocess.check_call(command)
        print(f"Folder successfully downloaded to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to download the folder. Details: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download a Google Drive folder using gdown.")
    parser.add_argument("-o", "--output", default=".", help="Output directory to save the downloaded files (default: current directory).")
    parser.add_argument("-u", "--url", default="", help="Url to Google Drive folder.")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Download folder
    download_folder(args.url, output_dir)

if __name__ == "__main__":
    main()

