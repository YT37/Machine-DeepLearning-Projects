# ObjectDetector

This project Detects objects in Images & Videos using SingleShotMultiboxDetector.

# Credits

- The SSD Algorithm was taken from [this](https://github.com/amdegroot/ssd.pytorch) repositary which is modified to work with my implimentation.

# Requirements

- Python 3.6 or up
- PyTorch 1.1.0
- TorchVision 0.3.0
- OpenCV
- ImageIO
- FFmpeg
- TQDM

# Instructions To Use

- I have already trained the model on VOC dataset but if you want to train on any other dataset check out [this](https://github.com/amdegroot/ssd.pytorch/blob/master/README.md) documentation.

- Extract the Weights.pth from the Weights.zip file first before running.

- Install FFmpeg using [these](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg) instructions.

- If you want to detect objects in Images or Videos Uncomment the respective parts of the code.

- As the project requires old versions of some packages make an enviorment if required.

- Make sure all files should be in the same folder before running.

- **Windows, Mac and Linux**
  ```
  pip install -r requirements.txt
  ```
- **Windows**
  ```
  python ObjectDetector.py
  ```
- **Mac or Linux**
  ```
  python3 ObjectDetector.py
  ```
