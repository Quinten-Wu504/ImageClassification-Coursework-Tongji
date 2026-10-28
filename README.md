# Image Classification

Computer Vision coursework at Tongji University.

- **MNIST:** a PyTorch fully connected network (784 → 100 → 10) trained with Adam for handwritten digit recognition.
- **Cat breeds:** TensorFlow Lite inference with a Google Teachable Machine model, supporting images and a live webcam.

## Run

Dependencies: NumPy, OpenCV, Pillow, TensorFlow, PyTorch, torchvision and tqdm. `requirements.txt` preserves the original version list; it is a historical reference, not a verified lockfile for current Python versions.

Run from the repository root after installing compatible dependencies:

```bash
mkdir MNISTModel
python main.py --mnist --epoch 10
python main.py --gtm --static --test_image_name BritishShorthair01.jpg
python main.py --gtm --realtime
```

MNIST downloads automatically into `Dataset/`. Cat inference requires `GoogleTMModel/model.tflite`, its matching `GoogleTMModel/labels.txt`, and test images under `Dataset/Cat/test/`. Press `q` to close webcam inference. Both TensorFlow and PyTorch are imported by the main entry point.

The optional shell scripts activate a Conda environment named `CVCourse`. The cat dataset is included under `Dataset/Cat/`: 108 training images (27 per breed: British Shorthair, Maine Coon, Ragdoll and Siamese) and 20 test images. MNIST data, trained models and the matching TensorFlow Lite labels are also included.
