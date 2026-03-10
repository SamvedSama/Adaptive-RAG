"""
load_qasper.py
--------------
Downloads 100 official QASPER papers from arXiv.
All paper IDs come directly from the official QASPER dataset JSON.
Run this before ingestion.py.

Usage:
    python load_qasper.py
"""

import os
import requests
import time
import tarfile
import json

# ── Setup directories ──────────────────────────────────────────────────────────
data_dir = "data"
pdf_dir = os.path.join(data_dir, "raw_pdfs")
os.makedirs(pdf_dir, exist_ok=True)

# ── Step 1: Download official QASPER archive ───────────────────────────────────
tgz_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
tgz_path = os.path.join(data_dir, "qasper-train-dev.tgz")

if not os.path.exists(tgz_path):
    print("Downloading official QASPER archive from AllenAI...")
    response = requests.get(tgz_url, stream=True)
    with open(tgz_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")
else:
    print("QASPER archive already downloaded, skipping.")

# ── Step 2: Extract the archive ────────────────────────────────────────────────
print("Extracting archive...")
with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(path=data_dir)

# ── Step 3: Find the extracted JSON file ───────────────────────────────────────
json_path = None
for root, dirs, files in os.walk(data_dir):
    if "qasper-train-v0.3.json" in files:
        json_path = os.path.join(root, "qasper-train-v0.3.json")
        break

if not json_path:
    print("Error: Could not find the JSON file after extracting.")
    exit()

# ── Step 4: Read paper IDs from QASPER ────────────────────────────────────────
print("Reading dataset...")
with open(json_path, "r", encoding="utf-8") as f:
    qasper_data = json.load(f)

all_paper_ids = list(qasper_data.keys())
print(f"Total papers available in QASPER: {len(all_paper_ids)}")

# Target 100 PDFs — skip ones already downloaded
NUM_TARGET = 100

# ── Step 5: Download up to 100 PDFs ───────────────────────────────────────────
already_have = [f.replace(".pdf", "") for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
print(f"Already have: {len(already_have)} PDFs")

to_download = [pid for pid in all_paper_ids if pid not in already_have]
still_needed = NUM_TARGET - len(already_have)

if still_needed <= 0:
    print(f"Already have {len(already_have)} PDFs. Nothing to download.")
    exit()

print(f"Need to download: {still_needed} more PDFs to reach {NUM_TARGET} total")
print("-" * 50)

downloaded = 0
failed = 0
skipped = 0

for i, paper_id in enumerate(to_download):
    if downloaded >= still_needed:
        break

    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")

    # Skip if already exists
    if os.path.exists(pdf_path):
        skipped += 1
        continue

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    print(f"[{len(already_have) + downloaded + 1}/{NUM_TARGET}] Downloading {paper_id}.pdf...")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, headers=headers, timeout=30)

        if response.status_code == 200 and b"%PDF" in response.content[:10]:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            size_kb = len(response.content) // 1024
            print(f"   ✓ Success! ({size_kb} KB)")
            downloaded += 1
        else:
            print(f"   ✗ Failed — HTTP {response.status_code}, skipping")
            failed += 1

    except Exception as e:
        print(f"   ✗ Error: {e}, skipping")
        failed += 1

    time.sleep(2)  # be polite to arXiv servers

# ── Final Summary ──────────────────────────────────────────────────────────────
total_pdfs = len([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

print("\n" + "=" * 50)
print(f"Download complete!")
print(f"  New downloads : {downloaded}")
print(f"  Failed/skipped: {failed}")
print(f"  Total PDFs now: {total_pdfs}")
print("=" * 50)
print(f"\nNext step: python ingestion.py")