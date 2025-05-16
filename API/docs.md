# DeOldify Image Colorization API

This documentation provides a complete guide to setting up and using the FastAPI-based DeOldify Image Colorization API.

## Overview

The DeOldify API provides a simple interface for colorizing black and white images. It uses the DeOldify AI model to add realistic colors to grayscale images, with options for different colorization styles and quality levels.

## Files Structure

```
deoldify-api/
├── app.py                 # FastAPI application 
├── requirements.txt       # Python dependencies
├── DeOldify/              # DeOldify repository (cloned from GitHub)
│   └── models/            # Directory for model weights
│       ├── ColorizeArtistic_gen.pth  # Artistic colorization model
│       └── ColorizeStable_gen.pth    # Stable colorization model (optional)
├── uploads/               # Temporary directory for uploaded images
└── outputs/               # Directory for colorized images
```

## Prerequisites

- Python 3.7+ (3.8 or 3.9 recommended)
- Git
- PyTorch 2.x compatible system
- Internet connection (for downloading models)
- 4GB+ RAM (8GB+ recommended)

## Step-by-Step Setup Guide

### 1. Create Project Directory

```bash
mkdir deoldify-api
cd deoldify-api
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Clone DeOldify Repository

```bash
git clone https://github.com/jantic/DeOldify.git
```

### 4. Install Required Packages

Create a `requirements.txt` file with the following content:

```
torch>=2.0.0
torchvision>=0.15.0
fastai>=1.0.60
pillow
fastapi
python-multipart
uvicorn
aiofiles
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

### 5. Download Model Weights

Download the DeOldify model weights:

1. Create the models directory:
   ```bash
   mkdir -p DeOldify/models
   ```

2. Download the model files:
   - Artistic model (recommended): [ColorizeArtistic_gen.pth](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth)
   - Stable model (optional): [ColorizeStable_gen.pth](https://data.deepai.org/deoldify/ColorizeStable_gen.pth)

3. Place the downloaded files in the `DeOldify/models` directory.

### 6. Create the FastAPI Application

Save the provided FastAPI script as `app.py` in your project directory.

### 7. Create Output Directories

```bash
mkdir uploads outputs
```

## Running the Server

Start the FastAPI server with:

```bash
python app.py
```

The server will be available at http://localhost:8000

For development with auto-reload:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### GET /
- Returns basic information about the API

### GET /models
- Lists available colorization models

### POST /colorize
- **Purpose**: Upload and colorize an image
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file`: The image file to colorize (required)
  - `render_factor`: Quality level, 10-45 (default: 35)
  - `model_type`: Either "artistic" or "stable" (default: "artistic")
- **Response**: JSON with colorized image URL and metadata

### GET /download/{filename}
- Download a specific processed image by filename

## Using the API

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/colorize" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/blackandwhite_image.jpg" \
  -F "render_factor=35" \
  -F "model_type=artistic"
```

### Example Python Request

```python
import requests

url = "http://localhost:8000/colorize"
files = {"file": open("path/to/your/blackandwhite_image.jpg", "rb")}
data = {"render_factor": 35, "model_type": "artistic"}

response = requests.post(url, files=files, data=data)
print(response.json())

# Access the colorized image
image_url = response.json()["colorized_image_url"]
colorized_image = requests.get(f"http://localhost:8000{image_url}")
with open("colorized_image.jpg", "wb") as f:
    f.write(colorized_image.content)
```

## Render Factor Explained

The `render_factor` parameter controls the quality and speed of colorization:

- **10-15**: Low quality but very fast
- **20-30**: Medium quality, moderate speed
- **35-40**: High quality, slower processing
- **45**: Maximum quality, slowest processing

## Troubleshooting

### Common Issues

#### Model Not Found Error
```
Colorization failed: Model file not found
```
**Solution**: Make sure the model files are downloaded and placed in the `DeOldify/models` directory.

#### Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution**: Try lowering the `render_factor` parameter or run on CPU if using a GPU with limited memory.

#### Import Error
```
Error importing DeOldify
```
**Solution**: Make sure DeOldify is properly cloned and all required packages are installed.

#### PyTorch Version Issues
```
WeightsUnpickler error
```
**Solution**: Make sure the PyTorch 2.x compatibility code is not modified, as it handles loading models with newer PyTorch versions.

### When All Else Fails

1. Activate your virtual environment
2. Update packages:
   ```
   pip install --upgrade torch torchvision fastapi uvicorn
   ```
3. Check if models are properly downloaded and in the correct location
4. Restart the server

## Performance Considerations

- CPU colorization will be significantly slower than GPU
- Higher render factors increase memory usage
- Process larger images may require more RAM
- Consider adding background task queuing for production usage

## Security Notes

- The API as provided has no authentication
- For production use, add proper authentication and rate limiting
- Add validation for uploaded file types and sizes
- Consider implementing a cleanup strategy for uploaded/processed files

# Increasing Color Intensity in DeOldify

This guide explains how to adjust the colorization parameters to get stronger, more vibrant colors in your DeOldify colorized images.

## Basic Approaches

There are several ways to increase the color intensity in DeOldify colorizations:

1. **Use the Artistic Model** - The artistic model produces more vibrant colors than the stable model
2. **Adjust the Render Factor** - Higher render factors allow for more detailed colorization
3. **Add Post-Processing** - Enhance the colorized image with saturation adjustments
4. **Modify the API** - Add color intensity controls to the API endpoint

## 1. Using the Artistic Model

The DeOldify system offers two colorization models:

- **Artistic Model** - Creates more vibrant and saturated colors but may be less historically accurate
- **Stable Model** - Produces more conservative and realistic colors

If you want stronger colors, always use the artistic model by setting `model_type="artistic"` in your API requests:

```bash
curl -X POST "http://localhost:8000/colorize" \
  -F "file=@image.jpg" \
  -F "render_factor=35" \
  -F "model_type=artistic"
```

In the Python client:

```python
requests.post(url, files={"file": open("image.jpg", "rb")}, 
              data={"render_factor": 35, "model_type": "artistic"})
```

## 2. Optimizing the Render Factor

The render factor (value between 10-45) controls the resolution at which the colorization is performed:

- Higher values produce more detailed colorization but don't necessarily increase color intensity
- The optimal range for vibrant colors is usually 35-40
- Going above 40 can actually reduce color vibrancy in some cases as the model becomes more conservative

Experiment with render factors between 35-40 to find the sweet spot for your specific images.

## 3. Adding Post-Processing for Enhanced Saturation

The most effective way to increase color intensity is to add post-processing to the API. Here's how to modify the FastAPI app to include saturation enhancement:

### Updating the API Code

Add this function to your `app.py` file:

```python
from PIL import ImageEnhance

def enhance_colors(image, saturation_factor=1.5, contrast_factor=1.2):
    """
    Enhance the colors of an image by increasing saturation and contrast.
    
    Args:
        image (PIL.Image): The image to enhance
        saturation_factor (float): Factor to increase saturation (1.0 = original, 2.0 = double saturation)
        contrast_factor (float): Factor to increase contrast (1.0 = original)
        
    Returns:
        PIL.Image: The enhanced image
    """
    # Enhance saturation
    enhancer = ImageEnhance.Color(image)
    saturated_image = enhancer.enhance(saturation_factor)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(saturated_image)
    enhanced_image = enhancer.enhance(contrast_factor)
    
    return enhanced_image
```

### Modifying the Colorize Endpoint

Then update your `/colorize` endpoint to include saturation adjustment parameters:

```python
@app.post("/colorize")
async def colorize_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    render_factor: int = Form(35),
    model_type: str = Form("artistic"),
    saturation: float = Form(1.0),  # Add saturation parameter, default = original
    contrast: float = Form(1.0),    # Add contrast parameter, default = original
):
    # ...existing code...
    
    # Colorize the image
    try:
        # Get the appropriate colorizer
        colorizer = get_colorizer(artistic=(model_type == "artistic"))
        
        # Colorize the image
        print(f"Colorizing image with render_factor={render_factor}, model_type={model_type}")
        result = colorizer.get_transformed_image(upload_path, render_factor=render_factor)
        
        # Apply color enhancement if requested
        if saturation > 1.0 or contrast > 1.0:
            print(f"Enhancing colors: saturation={saturation}, contrast={contrast}")
            result = enhance_colors(result, saturation_factor=saturation, contrast_factor=contrast)
        
        # Save the result
        result.save(output_path)
        
        # ...rest of the function...
```

### Using the Enhanced API

Now you can adjust the saturation and contrast when making API requests:

```bash
curl -X POST "http://localhost:8000/colorize" \
  -F "file=@image.jpg" \
  -F "render_factor=35" \
  -F "model_type=artistic" \
  -F "saturation=1.8" \
  -F "contrast=1.3"
```

## 4. Creating a Dedicated High-Intensity Endpoint

You can also add a specialized endpoint for high-intensity colorization that applies preset enhancement:

```python
@app.post("/colorize/vibrant")
async def colorize_image_vibrant(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    render_factor: int = Form(38),  # Higher default render factor
    intensity: str = Form("medium"),  # Options: low, medium, high, extreme
):
    # Map intensity levels to saturation/contrast values
    intensity_map = {
        "low": (1.3, 1.1),
        "medium": (1.6, 1.2),
        "high": (1.9, 1.3),
        "extreme": (2.2, 1.4)
    }
    
    # Get saturation and contrast values based on intensity
    saturation, contrast = intensity_map.get(intensity, (1.6, 1.2))
    
    # Use the artistic model for more vibrant colors
    model_type = "artistic"
    
    # Process similarly to the regular colorize endpoint but with enhancement
    # ... implementation similar to colorize endpoint ...
    
    # Include the enhancement step
    result = enhance_colors(result, saturation_factor=saturation, contrast_factor=contrast)
```

## 5. Advanced: Direct Colorization Parameters

For advanced users, you can modify the colorization process directly by adjusting how the DeOldify model processes images.

Add a `gamma` parameter to adjust the color gamma during processing:

```python
def colorize_with_gamma(colorizer, image_path, render_factor=35, gamma=1.0):
    """Colorize an image with gamma adjustment for color intensity"""
    # Get the base colorized image
    colorized = colorizer.get_transformed_image(image_path, render_factor=render_factor)
    
    if gamma != 1.0:
        # Convert to numpy array for gamma adjustment
        import numpy as np
        img_array = np.array(colorized).astype(np.float32) / 255
        
        # Apply gamma correction to increase color intensity
        img_array = np.power(img_array, 1/gamma)
        
        # Convert back to 8-bit image
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        # Convert back to PIL Image
        from PIL import Image
        colorized = Image.fromarray(img_array)
    
    return colorized
```

## Recommended Settings for Vibrant Colors

For the most vibrant colors without looking unnatural:

| Intensity Level | Model Type | Render Factor | Saturation | Contrast | Gamma |
|-----------------|------------|---------------|------------|----------|-------|
| Standard        | artistic   | 35            | 1.0        | 1.0      | 1.0   |
| Enhanced        | artistic   | 38            | 1.5        | 1.2      | 1.0   |
| Vibrant         | artistic   | 40            | 1.8        | 1.3      | 1.2   |
| Ultra Vibrant   | artistic   | 40            | 2.2        | 1.4      | 1.4   |

## Batch Processing Example

If you need to process multiple images with enhanced colors, here's a Python script example:

```python
import requests
import os
from PIL import Image, ImageEnhance

def enhance_and_colorize_batch(api_url, image_dir, output_dir, saturation=1.8, contrast=1.3):
    """Batch process multiple images with enhanced color settings"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(image_dir, filename)
            
            # API call with enhanced settings
            files = {"file": open(image_path, "rb")}
            data = {
                "render_factor": 38, 
                "model_type": "artistic",
                "saturation": saturation,
                "contrast": contrast
            }
            
            response = requests.post(f"{api_url}/colorize", files=files, data=data)
            files["file"].close()
            
            if response.status_code == 200:
                result = response.json()
                print(f"Successfully processed {filename}")
                
                # Download the colorized image
                image_url = result["colorized_image_url"]
                img_data = requests.get(f"{api_url}{image_url}").content
                
                # Save to output directory
                output_path = os.path.join(output_dir, f"enhanced_{filename}")
                with open(output_path, 'wb') as f:
                    f.write(img_data)
            else:
                print(f"Failed to process {filename}: {response.text}")

# Example usage
# enhance_and_colorize_batch("http://localhost:8000", "input_images", "output_images", 1.8, 1.3)
```

## Conclusion

For stronger colors in DeOldify:

1. Always use the artistic model
2. Use render factors between 35-40
3. Add post-processing with saturation and contrast enhancement
4. Consider advanced techniques like gamma correction for specific use cases

Experiment with different settings to find the optimal balance between color vibrancy and natural appearance for your particular images.