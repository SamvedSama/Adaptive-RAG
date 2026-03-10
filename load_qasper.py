import os
import requests
import time
import tarfile
import json

# Setup directories
data_dir = "data"
pdf_dir = os.path.join(data_dir, "raw_pdfs")
os.makedirs(pdf_dir, exist_ok=True)

# 1. Download the official QASPER dataset archive directly from AllenAI
tgz_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
tgz_path = os.path.join(data_dir, "qasper-train-dev.tgz")

if not os.path.exists(tgz_path):
    print("Downloading official QASPER archive from AllenAI...")
    response = requests.get(tgz_url, stream=True)
    with open(tgz_path, 'wb') as f:
        f.write(response.content)

# 2. Extract the archive
print("Extracting archive...")
with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(path=data_dir)

# 3. Find the extracted JSON file
json_path = None
for root, dirs, files in os.walk(data_dir):
    if "qasper-train-v0.3.json" in files:
        json_path = os.path.join(root, "qasper-train-v0.3.json")
        break

if not json_path:
    print("Error: Could not find the JSON file after extracting.")
    exit()

# 4. Read the JSON to get the official arXiv IDs
print("Reading dataset...")
with open(json_path, "r", encoding="utf-8") as f:
    qasper_data = json.load(f)

paper_ids = list(qasper_data.keys())
num_papers_to_download = 40
print(f"Preparing to download {num_papers_to_download} PDFs from arXiv...")

# 5. Download the PDFs
for i in range(num_papers_to_download):
    paper_id = paper_ids[i]
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    
    if os.path.exists(pdf_path):
        print(f"[{i+1}/{num_papers_to_download}] Already have {paper_id}.pdf")
        continue
        
    print(f"[{i+1}/{num_papers_to_download}] Downloading {paper_id}.pdf...")
    try:
        # Added a User-Agent because arXiv will block default Python requests 
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"   Success!")
        else:
            print(f"   Failed. HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
        
    time.sleep(3) # Be nice to arXiv's servers

print("Finished! Check your data/raw_pdfs/ folder.")