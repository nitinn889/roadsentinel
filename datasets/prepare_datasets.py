# RoadSentinel Dataset Preparation & Organization Tool
import os
import zipfile
import shutil
import glob
import random
import json

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def prepare_pothole_mix(downloads_dir, datasets_dir):
    zip_path = os.path.join(downloads_dir, "pothole-mix-v1.0-20220526.zip")
    target_dir = os.path.join(datasets_dir, "pothole_mix")
    if os.path.exists(zip_path):
        os.makedirs(target_dir, exist_ok=True)
        extract_zip(zip_path, target_dir)
        print("Pothole Mix v2 prepared.")
    else:
        print("Pothole Mix v2 zip not found. Skipping.")

def prepare_rdd2022_full(downloads_dir, datasets_dir):
    zip_path = os.path.join(downloads_dir, "RDD2022_released_through_CRDDC2022.zip")
    if os.path.exists(zip_path):
        print("Extracting RDD2022 subsets...")
        # Extract only China_Drone and India from the large zip file
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if member.startswith("China_Drone/") or member.startswith("India/"):
                    z.extract(member, os.path.join(datasets_dir, "rdd2022_full"))
        print("RDD2022 subsets extracted.")
    else:
        print("RDD2022 Full zip not found. Skipping.")

def prepare_water_filled(downloads_dir, datasets_dir):
    zip_path = os.path.join(downloads_dir, "tp95cdvgm8-1.zip")
    target_dir = os.path.join(datasets_dir, "water_filled_potholes")
    if os.path.exists(zip_path):
        os.makedirs(target_dir, exist_ok=True)
        extract_zip(zip_path, target_dir)
        print("Water-Filled and Dry Potholes prepared.")
    else:
        print("Water-Filled zip not found. Skipping.")

def prepare_pothole_600(downloads_dir, datasets_dir):
    zip_path = os.path.join(downloads_dir, "Pothole-600.zip")
    target_dir = os.path.join(datasets_dir, "pothole_600")
    if os.path.exists(zip_path):
        os.makedirs(target_dir, exist_ok=True)
        extract_zip(zip_path, target_dir)
        print("Pothole-600 prepared.")
    else:
        print("Pothole-600 zip not found. Skipping.")

def prepare_mwpd(downloads_dir, datasets_dir):
    zip_path = os.path.join(downloads_dir, "s5hx9n2jc3-2.zip")
    target_dir = os.path.join(datasets_dir, "mwpd")
    if os.path.exists(zip_path):
        os.makedirs(target_dir, exist_ok=True)
        extract_zip(zip_path, target_dir)
        print("MWPD prepared.")
    else:
        print("MWPD zip not found. Skipping.")

def prepare_qr4change(downloads_dir, datasets_dir):
    zip_path = os.path.join(downloads_dir, "zndzygc3p3-2.zip")
    target_dir = os.path.join(datasets_dir, "qr4change")
    if os.path.exists(zip_path):
        os.makedirs(target_dir, exist_ok=True)
        extract_zip(zip_path, target_dir)
        print("QR4Change prepared.")
    else:
        print("QR4Change zip not found. Skipping.")

def create_pi5_smoketest_subset(datasets_dir):
    print("Creating Pi5 smoke test subset...")
    subset_dir = os.path.join(datasets_dir, "pi5_smoketest_subset")
    os.makedirs(os.path.join(subset_dir, "images"), exist_ok=True)
    
    # We will search for Chitholian extracted directory. If not present, we will fallback to Pothole Mix images
    chitholian_dir = os.path.join(datasets_dir, "chitholian")
    images_found = []
    
    if os.path.exists(chitholian_dir):
        # Extract images from chitholian
        images_found = glob.glob(os.path.join(chitholian_dir, "**", "*.jpg"), recursive=True)
    
    if not images_found:
        print("Chitholian dataset not found. Falling back to Pothole Mix images for Pi5 smoke test.")
        pothole_mix_images = glob.glob(os.path.join(datasets_dir, "pothole_mix", "**", "images", "*.jpg"), recursive=True)
        if pothole_mix_images:
            images_found = pothole_mix_images
            
    if not images_found:
        print("No source images found for Pi5 smoke test subset.")
        return
        
    # Sample 80 images
    sampled = random.sample(images_found, min(80, len(images_found)))
    
    # Write COCO annotations
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pothole"}]
    }
    
    for idx, img_path in enumerate(sampled):
        filename = f"smoke_pothole_{idx:03d}.jpg"
        dest_path = os.path.join(subset_dir, "images", filename)
        shutil.copy(img_path, dest_path)
        
        # Add entry to coco
        coco_data["images"].append({
            "id": idx,
            "file_name": filename,
            "width": 512,
            "height": 512
        })
        # Dummy box for smoke testing validation
        coco_data["annotations"].append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "bbox": [100, 100, 200, 200],
            "area": 40000,
            "iscrowd": 0
        })
        
    with open(os.path.join(subset_dir, "annotations.json"), "w") as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Created Pi5 smoke test subset with {len(sampled)} images.")

if __name__ == "__main__":
    downloads_dir = "RoadSentinel_datasets/downloads"
    datasets_dir = "RoadSentinel_datasets"
    prepare_pothole_mix(downloads_dir, datasets_dir)
    prepare_rdd2022_full(downloads_dir, datasets_dir)
    prepare_water_filled(downloads_dir, datasets_dir)
    prepare_pothole_600(downloads_dir, datasets_dir)
    prepare_mwpd(downloads_dir, datasets_dir)
    prepare_qr4change(downloads_dir, datasets_dir)
    create_pi5_smoketest_subset(datasets_dir)
