"""
DeOldify FastAPI Server with Color Enhancement

This script creates a FastAPI server that provides endpoints for colorizing images using DeOldify
with additional options for color intensity enhancement.
"""
import os
import sys
import uuid
import torch
import functools
import shutil
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import ImageEnhance

# Enable PyTorch 2.x compatibility
try:
    torch.serialization.add_safe_globals([functools.partial])
    print("Successfully added functools.partial to PyTorch's safe globals")
except (AttributeError, TypeError) as e:
    print(f"Note: Could not register safe globals: {e}")

# Monkey patch torch.load to always use weights_only=False for PyTorch 2.x
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs and torch.__version__.startswith('2.'):
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
print("Patched torch.load to use weights_only=False")

# Add DeOldify to the Python path
deoldify_path = os.path.join(os.getcwd(), "DeOldify")
sys.path.append(deoldify_path)

try:
    from DeOldify.deoldify.visualize import *
    print("Successfully imported DeOldify")
except ImportError as e:
    print(f"Error importing DeOldify: {e}")
    print("Make sure DeOldify is properly installed and dependencies are met.")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="DeOldify Image Colorization API",
    description="API for colorizing black and white images using DeOldify with color intensity controls",
    version="1.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and outputs
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files directory to serve images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Initialize colorizers (lazy loading)
artistic_colorizer = None
stable_colorizer = None

def get_colorizer(artistic=True, models_dir=None):
    """Get the appropriate colorizer, initializing it if needed."""
    global artistic_colorizer, stable_colorizer
    
    # Set the default models directory if not provided
    if models_dir is None:
        models_dir = os.path.join(deoldify_path, "models")
    
    if artistic and artistic_colorizer is None:
        print(f"Initializing artistic colorizer with models from: {models_dir}")
        artistic_colorizer = get_image_colorizer(
            artistic=True,
            root_folder=Path(models_dir).parent
        )
    elif not artistic and stable_colorizer is None:
        print(f"Initializing stable colorizer with models from: {models_dir}")
        stable_colorizer = get_image_colorizer(
            artistic=False,
            root_folder=Path(models_dir).parent
        )
    
    return artistic_colorizer if artistic else stable_colorizer

def enhance_colors(image, saturation_factor=1.0, contrast_factor=1.0, gamma=1.0):
    """
    Enhance the colors of an image by increasing saturation and contrast.
    
    Args:
        image (PIL.Image): The image to enhance
        saturation_factor (float): Factor to increase saturation (1.0 = original, 2.0 = double saturation)
        contrast_factor (float): Factor to increase contrast (1.0 = original)
        gamma (float): Gamma correction factor (1.0 = original)
        
    Returns:
        PIL.Image: The enhanced image
    """
    # No enhancement needed if all factors are 1.0
    if saturation_factor == 1.0 and contrast_factor == 1.0 and gamma == 1.0:
        return image
    
    # Enhance saturation
    if saturation_factor != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
    
    # Enhance contrast
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
    
    # Apply gamma correction if needed
    if gamma != 1.0:
        import numpy as np
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255
        
        # Apply gamma correction
        img_array = np.power(img_array, 1/gamma)
        
        # Convert back to 8-bit image
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        # Convert back to PIL Image
        from PIL import Image
        image = Image.fromarray(img_array)
    
    return image

def cleanup_old_files(background_tasks: BackgroundTasks, file_path: str):
    """Add a task to remove a file after it's been served."""
    def remove_file(path: str):
        try:
            os.unlink(path)
        except Exception as e:
            print(f"Error removing file {path}: {e}")
    
    background_tasks.add_task(remove_file, file_path)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DeOldify Image Colorization API is running",
        "endpoints": {
            "POST /colorize": "Upload and colorize an image with custom settings",
            "POST /colorize/vibrant": "Colorize with preset vibrant color settings",
            "GET /models": "List available models",
        },
        "version": "1.1.0",
    }

@app.get("/models")
async def list_models():
    """List available colorization models."""
    models_dir = os.path.join(deoldify_path, "models")
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        return {"models_directory": models_dir, "exists": False, "models": []}
    
    # Get list of model files
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    return {
        "models_directory": models_dir,
        "exists": True,
        "models": models,
        "available_types": ["artistic", "stable"]
    }

@app.post("/colorize")
async def colorize_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    render_factor: int = Form(35),
    model_type: str = Form("artistic"),
    saturation: float = Form(1.0),  # Default = no saturation change
    contrast: float = Form(1.0),    # Default = no contrast change
    gamma: float = Form(1.0)        # Default = no gamma correction
):
    """
    Endpoint to upload and colorize an image with customizable color intensity.
    
    - file: The image file to colorize
    - render_factor: Quality of colorization (10-45, higher is better but slower)
    - model_type: Either "artistic" (more vibrant) or "stable" (more realistic)
    - saturation: Color saturation factor (1.0 = original, 2.0 = double saturation)
    - contrast: Contrast factor (1.0 = original, 2.0 = double contrast) 
    - gamma: Gamma correction factor (1.0 = original, values < 1.0 = darker, > 1.0 = brighter)
    """
    # Validate inputs
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if render_factor < 10 or render_factor > 45:
        raise HTTPException(status_code=400, detail="Render factor must be between 10 and 45")
    
    if model_type not in ["artistic", "stable"]:
        raise HTTPException(status_code=400, detail="Model type must be either 'artistic' or 'stable'")
    
    if saturation < 0 or saturation > 3.0:
        raise HTTPException(status_code=400, detail="Saturation must be between 0 and 3.0")
    
    if contrast < 0 or contrast > 3.0:
        raise HTTPException(status_code=400, detail="Contrast must be between 0 and 3.0")
    
    if gamma < 0.5 or gamma > 2.0:
        raise HTTPException(status_code=400, detail="Gamma must be between 0.5 and 2.0")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    upload_filename = f"{unique_id}_input{file_extension}"
    output_filename = f"{unique_id}_colorized{file_extension}"
    
    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save uploaded file
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        file.file.close()
    
    # Colorize the image
    try:
        # Get the appropriate colorizer
        colorizer = get_colorizer(artistic=(model_type == "artistic"))
        
        # Colorize the image
        print(f"Colorizing image with render_factor={render_factor}, model_type={model_type}")
        result = colorizer.get_transformed_image(upload_path, render_factor=render_factor)
        
        # Apply color enhancement if needed
        if saturation != 1.0 or contrast != 1.0 or gamma != 1.0:
            print(f"Enhancing colors: saturation={saturation}, contrast={contrast}, gamma={gamma}")
            result = enhance_colors(result, saturation_factor=saturation, 
                                   contrast_factor=contrast, gamma=gamma)
        
        # Save the colorized image
        result.save(output_path)
        
        # Clean up the upload file (will be done in background)
        background_tasks.add_task(os.unlink, upload_path)
        
        # Return the colorized image
        output_url = f"/outputs/{output_filename}"
        
        return {
            "status": "success",
            "message": "Image colorized successfully",
            "original_filename": file.filename,
            "colorized_image_url": output_url,
            "render_factor": render_factor,
            "model_type": model_type,
            "color_enhancement": {
                "saturation": saturation,
                "contrast": contrast,
                "gamma": gamma
            }
        }
    except Exception as e:
        # Clean up files on error
        if os.path.exists(upload_path):
            os.unlink(upload_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        # Provide detailed error information
        error_message = str(e)
        detail = f"Colorization failed: {error_message}"
        
        # Check for common errors
        if "No such file or directory" in error_message and "models" in error_message:
            detail = "Model file not found. Make sure the models are correctly installed in the DeOldify/models directory."
        
        raise HTTPException(status_code=500, detail=detail)

@app.post("/colorize/vibrant")
async def colorize_image_vibrant(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    intensity: str = Form("medium"),  # Options: low, medium, high, extreme
):
    """
    Endpoint to colorize an image with preset vibrant color settings.
    
    - file: The image file to colorize
    - intensity: Preset intensity level (low, medium, high, extreme)
    """
    # Map intensity levels to render_factor, saturation, contrast, gamma
    intensity_map = {
        "low": (35, 1.3, 1.1, 1.0),      # render_factor, saturation, contrast, gamma
        "medium": (38, 1.6, 1.2, 1.0),
        "high": (40, 1.9, 1.3, 1.2),
        "extreme": (40, 2.2, 1.4, 1.3)
    }
    
    if intensity not in intensity_map:
        raise HTTPException(status_code=400, 
                           detail=f"Intensity must be one of: {', '.join(intensity_map.keys())}")
    
    # Get preset values based on intensity
    render_factor, saturation, contrast, gamma = intensity_map[intensity]
    
    # Always use artistic model for vibrant colors
    model_type = "artistic"
    
    # Forward to the main colorize endpoint
    return await colorize_image(
        background_tasks=background_tasks,
        file=file,
        render_factor=render_factor,
        model_type=model_type,
        saturation=saturation,
        contrast=contrast,
        gamma=gamma
    )

@app.get("/download/{filename}")
async def download_image(background_tasks: BackgroundTasks, filename: str):
    """Download a specific output file by name."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Schedule file cleanup after serving (optional)
    # cleanup_old_files(background_tasks, file_path)
    
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    # Check if models exist
    models_dir = os.path.join(deoldify_path, "models")
    if not os.path.exists(models_dir):
        print(f"Warning: Models directory not found at {models_dir}")
        print("Please make sure the model files are in the DeOldify/models directory")
    else:
        models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not models:
            print(f"Warning: No model files found in {models_dir}")
            print("Please download the model files and place them in the DeOldify/models directory")
        else:
            print(f"Found models: {', '.join(models)}")
    
    # Run the server
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)