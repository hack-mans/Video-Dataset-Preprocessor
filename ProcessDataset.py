import os
import sys
import subprocess

from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch


def create_subfolders(root_folder, subfolder_names):
    # Create subfolders within the root folder
    subfolders = [os.path.join(root_folder, subfolder_name) for subfolder_name in subfolder_names]
    for subfolder in subfolders:
        os.makedirs(subfolder, exist_ok=True)


# Function to split videos into scenes and save images in the specified folder
def split_video_into_scenes(video_path, videos_folder, images_folder, num_images, threshold=27.0):
    # Open the video, create a scene manager, and add a detector.
    # Split video into scenes, save image for each scene.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    split_video_ffmpeg(video_path, scene_list,
                       output_file_template=os.path.join(videos_folder, '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4'),
                       show_progress=True)
    save_images(scene_list, video, num_images=num_images, frame_margin=1, image_extension='jpg', encoder_param=95,
                image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
                output_dir=images_folder, show_progress=True, scale=None, height=None, width=None, video_manager=None)


# Function to create captions for images in a specified folder
def caption_video_images(folder_path):
    # Initialize the BLIP2 processor and model
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Iterate through the image files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Open the image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')

            # Process the image
            inputs = processor(image, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Create a text file with the same name as the image
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(folder_path, txt_filename)

            # Write the generated text to the text file
            with open(txt_path, "w") as txt_file:
                txt_file.write(generated_text)


# Function to edit .txt caption files in the specified folder
def amend_captions(folder_location, text_to_append, append_position):
    try:
        # Iterate through all the files in the folder
        for filename in os.listdir(folder_location):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_location, filename)

                # Open the file for reading and writing
                with open(file_path, 'r+') as file:
                    file_content = file.read()
                    # Move the file pointer to the start and write the text input
                    file.seek(0)

                    if append_position == 'prefix':
                        file.write(text_to_append + file_content)
                    elif append_position == 'suffix':
                        file.write(file_content + text_to_append)
                    else:
                        print("Invalid append position. Please enter 'prefix' or 'suffix'.")
                        continue

                    file.truncate()

                print(f"Modified: {filename}")
    except FileNotFoundError:
        print("Folder not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def resize_and_crop(input_folder, output_folder, target_resolution):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)

        # Use FFMPEG to resize and crop the video without re-encoding
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'crop=ih*16/9:ih,scale={target_resolution[0]}:{target_resolution[1]}',
            '-c:a', 'copy',
            '-strict', 'experimental',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed: {input_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_path}: {e}")


def exit_application():
    user_input = input("Are you sure you want to exit? (Y/N): ").strip().lower()

    if user_input == "y":
        print("Exiting the application.")
        sys.exit()
    elif user_input == "n":
        print("Choose another menu item.")
        main()
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")
        exit_application()


def main():
    # Ask the user for their choice
    print("Select a function:")
    print("1. Process a video file into scenes with captions")
    print("2. Generate captions for a folder of images")
    print("3. Amend existing captions with new text")
    print("4. Resize and crop images and videos")
    print("5. Exit")

    choice = input("Enter the number of your choice: ")

    # Convert the user's input to an integer
    choice = int(choice)

    # Use conditional statements to call the selected functions
    if choice == 1:
        # Prompt user for input video file path and output folder path.
        input_video_path = input("Enter the path to the input video file: ")
        output_folder = input("Enter the path to the output folder: ")
        num_images = input("Enter the number of still frames to be saved: ")

        # Try to convert the user input into an integer
        try:
            num_images = int(num_images)
        except ValueError:
            print("Invalid input. Please enter an integer")
            main()

        # Extract the input video file name without extension and add to output path.
        video_name_without_extension = os.path.splitext(os.path.basename(input_video_path))[0]
        output_subfolder = os.path.join(output_folder, video_name_without_extension)

        # Create 'videos' and 'images' subfolders in the output folder.
        videos_folder = os.path.join(output_subfolder, 'videos')
        images_folder = os.path.join(output_subfolder, 'images')
        os.makedirs(videos_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        # Run video splitting function.
        split_video_into_scenes(input_video_path, videos_folder, images_folder, num_images)
        print("Scene splitting and image extraction completed.")

        # Run captioning function.
        caption_video_images(images_folder)
        print(f"Generated captions for video images and saved to {images_folder}")
        main()
    elif choice == 2:
        # Define the folder path containing the images
        folder_path = input("Enter the path to the folder containing images: ")

        # Run captioning function.
        caption_video_images(folder_path)
        print(f"Generated captions for video images and saved to {folder_path}")
        main()
    elif choice == 3:
        # Prompt user for input folder and text input
        folder_location = input("Enter the folder location: ")
        text_to_append = input("Enter the text to append to each .txt file: ")
        append_position = input("Append as 'prefix' or 'suffix'? ").lower()

        # Run caption amend function
        amend_captions(folder_location, text_to_append, append_position)
        print("All .txt caption files have been edited and saved.")
        main()
    elif choice == 4:
        images_folder = input("Enter the images folder path: ")
        videos_folder = input("Enter the videos folder path: ")

        # Create subfolders for images and videos
        image_subfolders = ["576x320", "1024x576"]
        video_subfolders = ["576x320", "1024x576"]

        create_subfolders(images_folder, image_subfolders)
        create_subfolders(videos_folder, video_subfolders)

        low_resolution = [576, 320]
        high_resolution = [1024, 576]

        # Process images and videos separately
        resize_and_crop(images_folder, os.path.join(images_folder, "576x320"), low_resolution)
        resize_and_crop(images_folder, os.path.join(images_folder, "1024x576"), high_resolution)
        resize_and_crop(videos_folder, os.path.join(videos_folder, "576x320"), low_resolution)
        resize_and_crop(videos_folder, os.path.join(videos_folder, "1024x576"), high_resolution)
        main()
    elif choice == 5:
        exit_application()
    else:
        print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
