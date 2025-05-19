import requests
import os
from tqdm import tqdm

url = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz"
local_filename = "Project_CodeNet.tar.gz"

# Get current size if file exists
resume_byte_pos = 0
if os.path.exists(local_filename):
    resume_byte_pos = os.path.getsize(local_filename)

# Send HTTP request with Range header
headers = {"Range": f"bytes={resume_byte_pos}-"}
response = requests.get(url, headers=headers, stream=True)

# Get total size (if available)
total_size = int(response.headers.get("Content-Length", 0)) + resume_byte_pos

# Start download with tqdm progress bar
with open(local_filename, "ab") as f, tqdm(
    total=total_size,
    initial=resume_byte_pos,
    unit="B",
    unit_scale=True,
    desc=local_filename,
) as bar:
    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
        if chunk:
            f.write(chunk)
            bar.update(len(chunk))

print("Download complete.")
