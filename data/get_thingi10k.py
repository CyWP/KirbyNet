import os
import sys
import argparse
from subprocess import check_call, CalledProcessError
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Download files from Thingiverse based on file IDs.")
    parser.add_argument("--output", "-o", help="Output directory", default="./")
    parser.add_argument("--ids", "-i", help="Path to IDs file", default="./ids.txt")
    parser.add_argument("--threads", "-t", type=int, help="Number of parallel download threads", default=4)
    return parser.parse_args()

def read_ids(file_path):
    abs_path = os.path.abspath(file_path)
    integers = []

    with open(abs_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.split()
            for part in parts:
                if part.lstrip('-').isdigit():
                    integers.append(int(part))
                else:
                    print(f"Warning: Non-integer value '{part}' found on line {line_number}. Skipping.")
    return integers

def download_file(file_id, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    url = f"https://www.thingiverse.com/download:{file_id}"
    r = requests.head(url)
    link = r.headers.get("Location")
    if not link:
        print(f"File {file_id} is no longer available on Thingiverse.")
        return

    _, ext = os.path.splitext(link)
    output_file = os.path.join(output_dir, f"{file_id}{ext.lower()}")
    print(f"Downloading {output_file}")
    
    try:
        command = ["wget", "-q", "-O", output_file, "--tries=10", link]
        check_call(command)
    except CalledProcessError:
        print(f"Failed to download file {file_id}. Please check your internet connection or the link.")

def main():
    args = parse_args()
    file_ids = read_ids(args.ids)
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_id = {executor.submit(download_file, file_id, args.output): file_id for file_id in file_ids}
        for future in as_completed(future_to_id):
            file_id = future_to_id[future]
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred while downloading file {file_id}: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

