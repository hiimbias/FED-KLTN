import cv2
import numpy as np

def Extractor(Images):
    Detector = cv2.SIFT_create() # Initialize SIFT Detector
    desc_seq = []
    count = 0

    for img in Images:
        # Skip invalid images
        if img is None or img.size == 0:
            print(f"Image Number {count} is empty or corrupted, skipping.")
            count += 1
            continue

        # Ensure the image is grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert to uint8 if necessary
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Detect and compute SIFT descriptors
        kp, desc = Detector.detectAndCompute(img, None)

        if desc is None or len(kp) == 0:
            print(f"No keypoints found in Image Number {count}, skipping.")
            count += 1
            continue

        print(f"Image Number {count} has been extracted!")
        desc_seq.append(desc)
        count += 1

    # Concatenate all descriptors and save
    descriptors_data = np.concatenate(desc_seq, axis=0) if desc_seq else np.empty((0, 128))
    filename = "../generator/NEW_SIFTDescriptors_FER2013.npy"
    np.save(filename, descriptors_data)
    print(f"{filename} has been saved to disk!")

# Load and preprocess images
image_dir = "/dataset/Fer_X.npy"
images = np.load(image_dir)  # Load images
images = images.astype(np.uint8)  # Convert to uint8

# Pass preprocessed images to Extractor
Extractor(images)
