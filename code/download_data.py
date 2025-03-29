import os
import requests
from pathlib import Path

def download_shakespeare():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # URL for the Shakespeare dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    # Download the file
    print("Downloading Shakespeare dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save the file
        output_path = data_dir / "shakespeare.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Dataset downloaded successfully to {output_path}")
        print(f"File size: {len(response.text)} characters")
    else:
        print(f"Error downloading file. Status code: {response.status_code}")

if __name__ == "__main__":
    download_shakespeare() 