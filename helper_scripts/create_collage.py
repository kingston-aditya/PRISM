import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import re
import math

main_dir = '/home/saividyaranya/PRISM/all_output_logs/infer_images'

subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
subdirs.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

try:
    font = ImageFont.truetype("arial.ttf", 12)
except IOError:
    font = ImageFont.load_default()

image_names = sorted([f for f in os.listdir(subdirs[0]) if f.endswith('.png')])

for img_name in image_names:
    images = []
    folder_names = []
    for subdir in subdirs:
        img_path = os.path.join(subdir, img_name)
        if os.path.exists(img_path):
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is not None:
                    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    images.append((img, img_path)) 
                    folder_names.append(os.path.basename(subdir)) 
                else:
                    print(f"Failed to read {img_path} with OpenCV.")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    num_images = len(images)
    if num_images == 0:
        continue 
    cols = math.ceil(math.sqrt(num_images)) 
    rows = math.ceil(num_images / cols)   

    collage_width = 1000  # Width of the collage
    collage_height = 1000  # Height of the collage
    thumb_width = collage_width // cols
    thumb_height = collage_height // rows


    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
    draw = ImageDraw.Draw(collage)


    x_offset = 0
    y_offset = 0
    for i, (img, img_path) in enumerate(images):
        try:

            img.thumbnail((thumb_width, thumb_height - 20))  # Leave space for text
            collage.paste(img, (x_offset, y_offset))
            text_position = (x_offset, y_offset + thumb_height - 20)
            draw.text(text_position, folder_names[i], fill="black", font=font)
            x_offset += thumb_width
            if x_offset >= collage_width:
                x_offset = 0
                y_offset += thumb_height
        except Exception as e:
            print(f"Error processing thumbnail for {img_path}: {e}")

    collage.save(f'collage_{img_name}_with_names_sorted.png')