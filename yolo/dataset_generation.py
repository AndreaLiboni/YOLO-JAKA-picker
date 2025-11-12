import os
import random
import cv2
import numpy as np
from shutil import copyfile
from pathlib import Path
import xml.etree.ElementTree as ET

def crop_transparent(img):
    # Find all non-transparent pixels
    gray = cv2.cvtColor(img[:, :, 3], cv2.COLOR_GRAY2BGR)
    _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(alpha)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop the image
    return img[y:y+h, x:x+w]

def rotate_image(img, angle=None):
    angle = random.uniform(0, 360) if angle is None else angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute the bounding box size after rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to consider translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def compute_overlap(box1, box2):
    _, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    _, b2_x1, b2_y1, b2_x2, b2_y2 = box2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area

def is_overlapping(new_box, existing_boxes, threshold=0.25):
    _, x1_new, y1_new, x2_new, y2_new = new_box
    for box in existing_boxes:
        if compute_overlap(new_box, box) > threshold:
            return True
    return False


def place_images_on_background(background_size, num_images, image_folder, output_image, output_labels, background, classes, img_dim=0.1, brightness_variation=0):
    # open background image and resize it
    background_img = cv2.imread(background, cv2.IMREAD_UNCHANGED)
    background_img = cv2.resize(background_img, (background_size[0], background_size[1]))
    #convert to rgba to use it as background
    background_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2BGRA)

    background = np.full((background_size[1], background_size[0], 4), background_rgb, dtype=np.uint8)
    
    # Get list of images
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    placed_boxes = []
    target_height = int(background_size[1] * img_dim)
    
    for _ in range(num_images):
        img_name = random.choice(images)
        img_path = os.path.join(image_folder, img_name)
        # CLASS
        class_name = img_name.split('_')[1].split('.')[0]
        class_name = classes[class_name]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # adjust brightness, not alpha
        if brightness_variation > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # brightness_factor = 1.0 + random.uniform(-brightness_variation, brightness_variation)
            brightness_factor = 1 - brightness_variation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
            img[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        if img is None or img.shape[-1] != 4:
            continue
        
        # img = crop_transparent(img)
        height, width = img.shape[:2]
        
        if width > height:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            height, width = img.shape[:2]
        
        # resize_factor = target_height / height
        resize_factor = img_dim
        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)
        img = rotate_image(img)
        height, width = img.shape[:2]
        
        x_offset = random.randint(0, background_size[0] - width)
        y_offset = random.randint(0, background_size[1] - height)
        
        new_box = (class_name, x_offset, y_offset, x_offset + width, y_offset + height)
        
        # Remove overlapping boxes from placed_boxes
        placed_boxes = [box for box in placed_boxes if not is_overlapping(new_box, [box])]
        placed_boxes.append(new_box)
        
        alpha_s = img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            background[y_offset:y_offset+height, x_offset:x_offset+width, c] = (
                alpha_s * img[:, :, c] + alpha_l * background[y_offset:y_offset+height, x_offset:x_offset+width, c]
            )
        background[y_offset:y_offset+height, x_offset:x_offset+width, 3] = img[:, :, 3]
    
    final_boxes = []
        
    # Save final image and labels
    cv2.imwrite(output_image, background, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    with open(output_labels, 'w') as f:
        for box in placed_boxes:
            cl, x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / background_size[0]
            y_center = (y1 + y2) / 2 / background_size[1]
            bb_width = (x2 - x1) / background_size[0]
            bb_height = (y2 - y1) / background_size[1]
            class_index = None
            for i in range(len(classes)):
                if classes[list(classes.keys())[i]] == cl:
                    class_index = i
                    break
            if class_index is None:
                raise ValueError(f"Class {cl} not found in classes dictionary.")
            f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {bb_width:.6f} {bb_height:.6f}\n")
    
    print(f"Image and label saved as {output_image}/.txt")


def overlay_labels_on_image(image_path, labels_path, output_image):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width, _ = image.shape
    
    with open(labels_path, 'r') as f:
        labels = f.readlines()
    
    for label in labels:
        parts = label.strip().split()
        if len(parts) != 5:
            continue
        
        _, x_center, y_center, bbox_width, bbox_height = map(float, parts)
        
        x1 = int((x_center - bbox_width / 2) * width)
        y1 = int((y_center - bbox_height / 2) * height)
        x2 = int((x_center + bbox_width / 2) * width)
        y2 = int((y_center + bbox_height / 2) * height)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)
    
    cv2.imwrite(output_image, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Labeled image saved as {output_image}")

def parse_cvat_annotations(cvat_root, classes):

    test_images = [img for img in os.listdir(os.path.join(cvat_root, 'images', 'Validation')) if img.endswith('.png') or img.endswith('.jpg')]

    # Parse the CVAT XML file
    tree = ET.parse(os.path.join(cvat_root, 'annotations.xml'))
    root = tree.getroot()

    images = {}
    for image_tag in root.findall('image'):
        image_name = image_tag.get('name')
        width = int(image_tag.get('width'))
        height = int(image_tag.get('height'))
        subset = image_tag.get('subset')
        images[image_name] = []

        for box in image_tag.findall("box"):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            x_center = (xtl + xbr) / 2 / width
            y_center = (ytl + ybr) / 2 / height
            bb_width = (xbr - xtl) / width
            bb_height = (ybr - ytl) / height
            cl = None
            for i in range(len(classes)):
                if classes[list(classes.keys())[i]] == box.get('label'):
                    cl = i
                    break
            if cl is None:
                raise ValueError(f"Class {box.get('label')} not found in classes dictionary.")
            images[image_name].append([
                cl,
                x_center,
                y_center,
                bb_width,
                bb_height
            ])
    return images

def generate_yolo_dataset(output_dir, num_images, train_img_folder, background_folder, classes, val_img_folder=None):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(images_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "test"), exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    train_split = int(num_images * 1)
    background_path = [img for img in os.listdir(background_folder) if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg')]

    for i in range(num_images):
        img_filename = f"{i+1}.jpg"
        label_filename = f"{i+1}.txt"
        subset = "train" if i < train_split else "val"
        background_image_path = os.path.join(background_folder, random.choice(background_path))
        # get background size
        img = cv2.imread(background_image_path)
        
        place_images_on_background(
            background_size=(img.shape[1], img.shape[0]),
            num_images=random.choice([1, 2, 3, 4, 5, 8, 10]),
            img_dim=1, # random.uniform(0.22, 0.18),
            background=background_image_path,
            image_folder=train_img_folder,
            output_image=os.path.join(images_dir, subset, img_filename),
            classes=classes,
            output_labels=os.path.join(labels_dir, subset, label_filename),
            brightness_variation=0,
        )
    
    # if validation folder is provided, parse CVAT annotations
    if val_img_folder is not None:
        images = parse_cvat_annotations(val_img_folder, classes)
        val_split = int(len(images) * 0.9)
        for i, (img_name, boxes_list) in enumerate(images.items()):
            img_filename = f"{i+1}.jpg"
            label_filename = f"{i+1}.txt"
            subset = "val" if i < val_split else "test"

            copyfile(
                src=os.path.join(val_img_folder, 'images', 'Validation', img_name),
                dst=os.path.join(images_dir, subset, img_filename)
            )

            with open(os.path.join(labels_dir, subset, label_filename), "w") as f:
                for box in boxes_list:
                    f.write(" ".join(map(str, box)) + "\n")
        
    # if no validation folder is provided, use the generated images for training and validation
    else:
        num_val_imgs = int(num_images * 0.2)
        num_test_imgs = int(num_images * 0.05)
        for subset, split_num in zip(['val', 'test'], [num_val_imgs, num_test_imgs]):
            for i in range(split_num):
                img_filename = f"{i+1}.jpg"
                label_filename = f"{i+1}.txt"
                subset = subset
                background_image_path = os.path.join(background_folder, random.choice(background_path))
                # get background size
                img = cv2.imread(background_image_path)
                
                place_images_on_background(
                    background_size=(img.shape[1], img.shape[0]),
                    num_images=random.choice([1, 2, 3, 4, 5, 8, 10]),
                    img_dim=1, # random.uniform(0.22, 0.18),
                    background=background_image_path,
                    image_folder=train_img_folder,
                    output_image=os.path.join(images_dir, subset, img_filename),
                    classes=classes,
                    output_labels=os.path.join(labels_dir, subset, label_filename),
                    brightness_variation=0,
                )
    
    # Create the data.yaml file
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write('path: ' + str(Path().resolve()) + '/' + output_dir + ' \ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n')
        for i, class_name in enumerate(classes.values()):
            f.write(f"    {i}: {class_name}\n")

    
    print(f"YOLO dataset with {num_images} images generated in {output_dir}")

def extract_object_from_background(img, background):
    diff = cv2.absdiff(img, background)
    # Add transparency channel if pixel is 0 in diff
    mask = []
    for c in range(3):
        _, channel_mask = cv2.threshold(diff[:, :, c], 30, 255, cv2.THRESH_BINARY)
        mask.append(channel_mask)
    combined_mask = cv2.bitwise_or(mask[0], cv2.bitwise_or(mask[1], mask[2]))
    alpha_channel = combined_mask
    
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha_channel]
    dst = cv2.merge(rgba, 4)
    extracted = crop_transparent(dst)
    return extracted

def prepare_single_objects(objects_img_folder, background_img_path):
    background_img = cv2.imread(background_img_path, cv2.IMREAD_UNCHANGED)
    output_folder = os.path.join(objects_img_folder, 'extracted')
    os.makedirs(output_folder, exist_ok=True)

    for path in [img for img in os.listdir(objects_img_folder) if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg')]:
        img = cv2.imread(os.path.join(objects_img_folder, path), cv2.IMREAD_UNCHANGED)
        extracted = extract_object_from_background(img, background_img)
        print(f"Extracted object saved as {os.path.join(output_folder, path.split('.')[0] + '.png')}")
        cv2.imwrite(os.path.join(output_folder, path.split('.')[0] + '.png'), extracted, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    return output_folder


# Example usage
# place_images_on_background(
#     background_size=(1280, 720),  # Size of the final image
#     num_images=2,                 # Number of images to place
#     img_dim=0.3,                # Resize factor for images
#     image_folder='bustine',         # Folder with images to place
#     output_image='output.jpg',    # Output image file
#     output_labels='output.txt'    # Output YOLO labels file
# )

# overlay_labels_on_image(
#     image_path='output.jpg',
#     labels_path='output.txt',
#     output_image='output_labeled.png' # Output image with labels
# )

# prepare_single_objects(
#     objects_img_folder="data/objects",
#     background_img_path="data/background.jpg"
# )

generate_yolo_dataset(
    output_dir="datasets",
    num_images=100,
    train_img_folder="data/objects/extracted",
    # val_img_folder="data/bustine/all/CVAT_validation",
    background_folder="data",
    classes={
        'F': 'front',
        'B': 'back',
        'W': 'white'
    }
)

