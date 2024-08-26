import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def resize_image(image, target_height):
    aspect_ratio = image.width / image.height
    new_width = int(target_height * aspect_ratio)
    return image.resize((new_width, target_height), Image.LANCZOS)

def create_frame(paths, target_height=500):
    images = [Image.open(path) for path in paths]
    resized_images = [resize_image(img, target_height) for img in images]
    
    new_width = sum(img.width for img in resized_images)
    new_image = Image.new('RGB', (new_width, target_height))
    
    x_offset = 0
    for img in resized_images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return np.array(new_image)

def process_images(input_dir, files):
    return create_frame([os.path.join(input_dir, f) for f in files])

def get_sorted_files(input_dir, prefix, extension):
    return sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith(extension)])

def main(args):
    input_dir = os.path.join(args.output_dir, 'imgs')
    
    file_types = [
        ('diffusion_', '.jpg'),
        ('bev_', '.png'),
        ('agent_', '.jpg')
    ]
    
    files = [get_sorted_files(input_dir, prefix, ext) for prefix, ext in file_types]
    
    if not all(len(file_list) == len(files[0]) for file_list in files):
        print("Error: Number of files in each category doesn't match.")
        return

    first_frame = process_images(input_dir, [f[0] for f in files])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(args.output_dir, args.output_video), fourcc, args.fps, (width, height))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_images, input_dir, [file_list[i] for file_list in files])
            for i in range(len(files[0]))
        ]
        for future in tqdm(futures, total=len(futures), desc="Processing images"):
            frame = future.result()
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()

    input_video_path = os.path.join(args.output_dir, args.output_video)
    temp_compressed_video = 'temp_compressed.mp4'
    temp_compressed_path = os.path.join(args.output_dir, temp_compressed_video)

    # Compress the video
    command = f'ffmpeg -i {input_video_path} -vcodec libx265 -crf 28 {temp_compressed_path}'
    os.system(command)

    # Replace the original video with the compressed one
    os.remove(input_video_path)
    os.rename(temp_compressed_path, input_video_path)
    print(f"Video saved as {args.output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--output_video", type=str, default="output_video.mp4", help="Output video name")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    args = parser.parse_args()
    
    main(args)