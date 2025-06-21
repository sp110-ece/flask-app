import albumentations as A
import os
import pandas as pd
import cv2
transform = A.Compose([
    # Simulate different input resolutions (this changes geometric projection via down/upscaling)
    # Applied with a low probability (p=0.3) for "small conditions"
    A.OneOf([
        A.Downscale(scale_min=0.5, scale_max=0.8, p=1.0), # Mild downscaling
        A.Downscale(scale_min=0.6, scale_max=0.9, p=1.0), # Another mild downscaling range
    ], p=0.3), # Apply this downscale simulation with 30% probability

    # Resize to standard target shape (deterministic geometric operation)
    A.Resize(height=256, width=256),

    # Simulate Y-axis leaning via affine shear (a type of warp)
    # Apply with a moderate probability (p=0.4) and small shear range
    A.Affine(
        shear={"y": (-5, 5)},  # Small shear only on Y-axis (vertical lean)
        rotate=0,              # Explicitly no rotation
        scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, # Small random scaling
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # Small random translation
        fit_output=False,      # Keep original image dimensions, potentially adding black borders
        p=0.4                  # Apply with 40% probability
    ),

    # Non-linear elastic deformation (another type of warp)
    # Apply with a very low probability (p=0.2) and small parameters
    A.ElasticTransform(
        alpha=120,          # Controls the intensity of the deformation
        sigma=120 * 0.05,   # Controls the smoothness of the deformation field
        alpha_affine=120 * 0.03, # Controls the intensity of the affine component of the deformation
        border_mode=cv2.BORDER_REFLECT_101, # How to handle pixels outside the image boundary
        p=0.2               # Apply with 20% probability
    ),

    # Grid distortion (another type of warp)
    # Apply with a low probability (p=0.2) and few steps
    A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.2), # Small distortion

    # Perspective transformation (another type of warp)
    # Apply with a low probability (p=0.2) and small scale
    A.Perspective(scale=(0.02, 0.05), p=0.2), # Small perspective distortion

    # Color and brightness changes (no geometric effect)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
    
    # Final conversion to float (no geometric effect)
    A.ToFloat(max_value=255.0)
],
    # Keypoint parameters are essential for geometric transforms to correctly adjust keypoint coordinates.
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)


df = pd.read_csv("./data/labels/standing_quad_pose.csv")
output_img_dir = "./data/aug_images"
output_csv = "./data/labels/keypoints.csv"

augmented_rows = []

for __, row in df.iterrows():
    img_path = os.path.join("data","images", row['image'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not os.path.exists(img_path):
        print("fade")
        print(img_path)
        exit()
    keypoints = []
    for i in range(1, len(row), 3):
        keypoints.append((row[i], row[i+1]))

    for i in range(20):
            # print("trying to load: ", img_path)
            transformation = transform(image=image, keypoints=keypoints)
            aug_image = transformation['image']
            aug_image = (aug_image * 255).astype('uint8')
            aug_keypoint = transformation['keypoints']
            base = os.path.splitext(row['image'])[0]
            new_img_name = f"{base}_aug_{i}.jpg"
            save_path = os.path.join(output_img_dir, new_img_name)
            cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            aug_row = [new_img_name]
            for kp_idx, (x,y) in enumerate(aug_keypoint):
                 v = row[3+kp_idx*3 - 1]
                 aug_row.extend([x, y, v])
            augmented_rows.append(aug_row)

columns = df.columns.tolist()
augmented_df = pd.DataFrame(augmented_rows, columns=columns)
augmented_df.to_csv(output_csv, index=False)



