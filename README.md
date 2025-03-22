# Depth Estimation API

## Description
This API is built with FastAPI and leverages transformer models to perform depth estimation on input images. It uses the "Intel/zoedepth-nyu" model to generate depth maps, which can be visualized in different modes (grayscale, color, overlay). Additional processing such as smoothing, thresholding, and edge detection can be applied to enhance the output. The API returns the processed depth map as a PNG image.

## Features
- **Depth Map Generation**: Uses the "Intel/zoedepth-nyu" model to estimate depth from an input image.
- **Visualization Modes**: Supports grayscale, color mapping, and overlay on the original image.
- **Image Processing Options**:
  - Smoothing via Gaussian Blur.
  - Thresholding to highlight specific depth ranges.
  - Edge detection using the Canny algorithm.
- **Flexible Query Parameters**: Allowing custom adjustments for smoothing kernel size, threshold values, and more.

## Technologies Used
- Python
- FastAPI
- Uvicorn
- Transformers (Hugging Face)
- Torch
- OpenCV
- NumPy
- Pillow

## Installation and Setup

### Requirements
- Python 3.8 or later.

### Installing Dependencies
Install the required libraries using pip:
```sh
pip install fastapi uvicorn transformers torch opencv-python numpy pillow
```

## How to Run the API
Start the API using Uvicorn:
```sh
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at: `http://0.0.0.0:8000`

## API Endpoint

### POST `/process_depth_map`
Processes an uploaded image to generate a depth map with various options.

#### Request Parameters:
- **file** (required): The image file to process.
- **mode** (query, default "gray"): Visualization mode: "gray", "color", or "overlay".
- **smoothing** (query, default false): Whether to apply Gaussian blur smoothing.
- **kernel_size** (query, default 5): Kernel size for smoothing (must be an odd number).
- **threshold** (query, default false): Whether to apply thresholding.
- **min_depth** (query, default 50): Minimum depth value for thresholding.
- **max_depth** (query, default 200): Maximum depth value for thresholding.
- **edges** (query, default false): Whether to apply edge detection.

#### Response:
Returns a PNG image with the processed depth map.

#### Example Request:
Send a POST request to `/process_depth_map` with an image file and desired query parameters using a tool like curl or Postman.
