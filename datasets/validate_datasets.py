# RoadSentinel Dataset Validation Tool
import os
import glob
import hashlib
from PIL import Image
import xml.etree.ElementTree as ET
import json

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_image_corrupt(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return False
    except Exception:
        return True

def validate_rdd2022(rdd_path):
    print(f"\n=== Validating RDD2022 Subset in {rdd_path} ===")
    results = {"total_images": 0, "corrupt": 0, "duplicates": 0}
    hashes = {}
    
    for root, _, files in os.walk(rdd_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                results["total_images"] += 1
                filepath = os.path.join(root, file)
                if check_image_corrupt(filepath):
                    results["corrupt"] += 1
                h = get_file_hash(filepath)
                if h in hashes:
                    results["duplicates"] += 1
                else:
                    hashes[h] = filepath
                    
    print(f"RDD2022 Results: Total Images: {results["total_images"]}, Corrupt: {results["corrupt"]}, Duplicates: {results["duplicates"]}")
    return results

def validate_pothole_mix(pm_path):
    print(f"\n=== Validating Pothole Mix in {pm_path} ===")
    images = glob.glob(os.path.join(pm_path, "**", "images", "*"), recursive=True)
    images = [f for f in images if os.path.isfile(f) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    results = {"images": len(images), "masks": 0, "missing_masks": 0, "corrupt": 0, "empty_masks": 0, "duplicates": 0}
    hashes = {}
    
    for img_path in images:
        if check_image_corrupt(img_path):
            results["corrupt"] += 1
            
        h = get_file_hash(img_path)
        if h in hashes:
            results["duplicates"] += 1
        else:
            hashes[h] = img_path
            
        # Find corresponding mask
        # Usually structured as: /images/xxx.jpg and /masks/xxx.png or similar
        mask_path = img_path.replace("images", "masks")
        # Try changing extension to png or matching image extension
        base, _ = os.path.splitext(mask_path)
        possible_masks = [base + ".png", base + ".jpg", base + ".jpeg", mask_path]
        mask_found = False
        for pm in possible_masks:
            if os.path.exists(pm):
                results["masks"] += 1
                mask_found = True
                # Check empty mask
                try:
                    with Image.open(pm) as mask_img:
                        if mask_img.getbbox() is None:
                            results["empty_masks"] += 1
                except Exception:
                    pass
                break
        if not mask_found:
            results["missing_masks"] += 1
            
    print(f"Pothole Mix Results: Images: {results["images"]}, Masks: {results["masks"]}, Missing Masks: {results["missing_masks"]}, Empty Masks: {results["empty_masks"]}, Corrupt: {results["corrupt"]}, Duplicates: {results["duplicates"]}")
    return results

def validate_bbox_dataset(path, name):
    print(f"\n=== Validating Bounding Box Dataset {name} in {path} ===")
    images = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)
    results = {"images": len(images), "annotations": 0, "missing_annotations": 0, "corrupt": 0, "invalid_boxes": 0}
    
    for img_path in images:
        if check_image_corrupt(img_path):
            results["corrupt"] += 1
            
        # Check XML (Pascal VOC)
        xml_path = os.path.splitext(img_path)[0] + ".xml"
        # Check if xml is in annotations folder
        if not os.path.exists(xml_path):
            # Try substituting /images/ with /annotations/
            xml_path = img_path.replace("images", "annotations").replace("xmls", "").replace(".jpg", ".xml")
            
        if os.path.exists(xml_path):
            results["annotations"] += 1
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                        if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                            results["invalid_boxes"] += 1
            except Exception:
                results["invalid_boxes"] += 1
        else:
            results["missing_annotations"] += 1
            
    print(f"{name} Results: Images: {results["images"]}, Annotations: {results["annotations"]}, Missing Annotations: {results["missing_annotations"]}, Invalid Boxes: {results["invalid_boxes"]}, Corrupt: {results["corrupt"]}")
    return results

if __name__ == "__main__":
    base_dir = "RoadSentinel_datasets"
    if os.path.exists(os.path.join(base_dir, "rdd2022")):
        validate_rdd2022(os.path.join(base_dir, "rdd2022"))
    if os.path.exists(os.path.join(base_dir, "pothole_mix")):
        validate_pothole_mix(os.path.join(base_dir, "pothole_mix"))
