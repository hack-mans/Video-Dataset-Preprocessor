# Video-Dataset-Preprocessor

Python environment + script for preprocessing video files into scenes with captions
(using BLIP2 - https://huggingface.co/Salesforce/blip2-opt-2.7b - requires ~13GB VRAM).
Also featuring the ability to amend caption files to add new text as prefix or suffix, and automatically crop and resize video clips into 576x320 and 1024x576 resolutions.

Menu:
- 1 Process a video file into scenes with captions
- 2 Generate captions for a folder of images
- 3 Amend existing captions with new text
- 4 Resize and crop images and videos
- 5 Exit

## Instructions
Requires FFPMEG so make sure you have installed a version locally.
https://ffmpeg.org/download.html

- From Command Line, clone the repo and setup the virtual environment
```
git clone https://github.com/hack-mans/Video-Dataset-Preprocessor.git
cd Video-Dataset-Preprocessor
pip install virtualenv
virtualenv venv
.\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/huggingface/transformers.git
pip install scenedetect[opencv] --upgrade
```
- Place video files in root directory
- Run the python script and provide the file names / paths when asked
```
python ProcessDataset.py
```
