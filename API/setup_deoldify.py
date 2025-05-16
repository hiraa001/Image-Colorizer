#!/usr/bin/env python3
"""
DeOldify Setup and Usage Script (GitHub Version) - Updated for Modern PyTorch
This script:
1. Sets up a virtual environment
2. Clones the official DeOldify repository
3. Installs the correct dependencies with modern PyTorch
4. Downloads the model weights
5. Provides a simple function to colorize images
"""

import os
import sys
import subprocess
import argparse
import platform
import shutil

def create_virtual_env(env_name="deoldify-env"):
    """Create a virtual environment for DeOldify."""
    print(f"Creating virtual environment: {env_name}")
    
    # Check if venv exists
    if os.path.exists(env_name):
        print(f"Environment {env_name} already exists.")
        return env_name
    
    # Create venv
    subprocess.run([sys.executable, "-m", "venv", env_name], check=True)
    print(f"Virtual environment {env_name} created successfully.")
    return env_name

def get_python_version():
    """Get the Python version."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"{major}.{minor}"

def install_pytorch(pip_cmd):
    """Install PyTorch with appropriate version based on system."""
    print("Installing PyTorch...")
    
    python_version = get_python_version()
    print(f"Python version: {python_version}")
    
    # Try different PyTorch versions, starting with the most recent that should work with DeOldify
    pytorch_versions = [
        # Version, CUDA version
        ("2.0.1", "cu118"),
        ("1.13.1", "cu117"),
        ("1.12.1", "cu116"),
        ("1.12.1", "cu113"),
        ("1.13.0", "cu117"),
    ]
    
    # For CPU-only installation (fallback)
    cpu_versions = [
        "2.0.1",
        "1.13.1",
        "1.12.1"
    ]
    
    # Try with CUDA first
    success = False
    for torch_version, cuda_version in pytorch_versions:
        try:
            print(f"Trying PyTorch {torch_version} with {cuda_version}...")
            result = subprocess.run([
                *pip_cmd, "install", 
                f"torch=={torch_version}", 
                f"torchvision=={torch_version.rsplit('.', 1)[0]}.1",
                "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
            ], check=False)
            
            if result.returncode == 0:
                print(f"Successfully installed PyTorch {torch_version} with {cuda_version}")
                success = True
                break
            else:
                print(f"Failed to install PyTorch {torch_version} with {cuda_version}")
        except Exception as e:
            print(f"Error installing PyTorch {torch_version} with {cuda_version}: {e}")
    
    # If CUDA installation fails, try CPU version
    if not success:
        print("CUDA installation failed, trying CPU-only versions...")
        for torch_version in cpu_versions:
            try:
                print(f"Trying PyTorch {torch_version} (CPU)...")
                result = subprocess.run([
                    *pip_cmd, "install", 
                    f"torch=={torch_version}", 
                    f"torchvision=={torch_version.rsplit('.', 1)[0]}.1",
                ], check=False)
                
                if result.returncode == 0:
                    print(f"Successfully installed PyTorch {torch_version} (CPU)")
                    success = True
                    break
                else:
                    print(f"Failed to install PyTorch {torch_version} (CPU)")
            except Exception as e:
                print(f"Error installing PyTorch {torch_version} (CPU): {e}")
    
    if not success:
        print("WARNING: All PyTorch installation attempts failed. Trying to install latest PyTorch...")
        try:
            subprocess.run([*pip_cmd, "install", "torch", "torchvision"], check=True)
            print("Installed latest PyTorch version")
            success = True
        except Exception as e:
            print(f"Error installing latest PyTorch: {e}")
            raise RuntimeError("Failed to install PyTorch. Please install it manually.")
    
    return success

def install_dependencies(env_name):
    """Install all necessary dependencies in the virtual environment."""
    print("Installing dependencies...")
    
    # Determine the path to python and pip in the virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(env_name, "Scripts", "python.exe")
        pip_cmd = [python_path, "-m", "pip"]
    else:  # Unix/Linux/Mac
        python_path = os.path.join(env_name, "bin", "python")
        pip_cmd = [python_path, "-m", "pip"]
    
    # Install base requirements (using python -m pip to avoid issues)
    subprocess.run([*pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Clone DeOldify repository if it doesn't exist
    if not os.path.exists("DeOldify"):
        print("Cloning DeOldify repository...")
        subprocess.run(["git", "clone", "https://github.com/jantic/DeOldify.git"], check=True)
    
    # Install PyTorch with compatible version
    install_pytorch(pip_cmd)
    
    # Install fastai and other dependencies 
    print("Installing fastai and other dependencies...")
    fastai_versions = ["2.7.12", "2.5.3", "1.0.61"]
    fastcore_versions = ["1.5.29", "1.3.27", "1.0.20"]
    
    # Try different fastai versions
    fastai_success = False
    for fastai_version in fastai_versions:
        try:
            print(f"Trying fastai {fastai_version}...")
            result = subprocess.run([*pip_cmd, "install", f"fastai=={fastai_version}"], check=False)
            if result.returncode == 0:
                print(f"Successfully installed fastai {fastai_version}")
                fastai_success = True
                break
        except Exception as e:
            print(f"Error installing fastai {fastai_version}: {e}")
    
    if not fastai_success:
        print("WARNING: All fastai installation attempts failed. Trying to install latest fastai...")
        try:
            subprocess.run([*pip_cmd, "install", "fastai"], check=True)
            print("Installed latest fastai version")
        except Exception as e:
            print(f"Error installing latest fastai: {e}")
    
    # Try different fastcore versions
    fastcore_success = False
    for fastcore_version in fastcore_versions:
        try:
            print(f"Trying fastcore {fastcore_version}...")
            result = subprocess.run([*pip_cmd, "install", f"fastcore=={fastcore_version}"], check=False)
            if result.returncode == 0:
                print(f"Successfully installed fastcore {fastcore_version}")
                fastcore_success = True
                break
        except Exception as e:
            print(f"Error installing fastcore {fastcore_version}: {e}")
    
    if not fastcore_success:
        print("WARNING: All fastcore installation attempts failed. Trying to install latest fastcore...")
        try:
            subprocess.run([*pip_cmd, "install", "fastcore"], check=True)
            print("Installed latest fastcore version")
        except Exception as e:
            print(f"Error installing latest fastcore: {e}")
    
    # Install other dependencies
    other_deps = [
        "pillow",
        "numpy",
        "scipy",
        "jupyter",
        "matplotlib"
    ]
    
    print("Installing other dependencies...")
    for dep in other_deps:
        try:
            subprocess.run([*pip_cmd, "install", dep], check=True)
        except Exception as e:
            print(f"Warning: Failed to install {dep}: {e}")
    
    # Install DeOldify in editable mode
    try:
        os.chdir("DeOldify")
        subprocess.run([*pip_cmd, "install", "-e", "."], check=False)
        os.chdir("..")
        print("Installed DeOldify in development mode")
    except Exception as e:
        print(f"Warning: Failed to install DeOldify in development mode: {e}")
        if os.path.exists("DeOldify"):
            os.chdir("..")
    
    print("Dependencies installation completed.")

def download_models():
    """Download DeOldify model weights."""
    print("Downloading model weights...")
    
    # Create models directory if it doesn't exist
    os.makedirs("DeOldify/models", exist_ok=True)
    
    # Define model URLs
    models = {
        "ColorizeArtistic_gen.pth": "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        "ColorizeStable_gen.pth": "https://data.deepai.org/deoldify/ColorizeStable_gen.pth"
    }
    
    # Download models
    for model_name, url in models.items():
        model_path = os.path.join("DeOldify", "models", model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                print(f"Downloaded {model_name} successfully.")
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
                print(f"Please download manually from {url} and place in the DeOldify/models directory.")

def create_wrapper_module():
    """Create a simplified wrapper module for DeOldify."""
    wrapper_dir = "deoldify_wrapper"
    os.makedirs(wrapper_dir, exist_ok=True)
    
    # Create __init__.py
    with open(os.path.join(wrapper_dir, "__init__.py"), "w") as f:
        f.write("# DeOldify wrapper module\n")
    
    # Create colorizer.py wrapper
    colorizer_path = os.path.join(wrapper_dir, "colorizer.py")
    with open(colorizer_path, "w") as f:
        f.write("""
import os
import sys
import warnings
from pathlib import Path

# Add DeOldify to path
deoldify_path = Path(__file__).parent.parent / 'DeOldify'
sys.path.append(str(deoldify_path))

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import DeOldify components
try:
    from deoldify.visualize import *
    from fastai.vision.all import *
except ImportError as e:
    print(f"Error importing DeOldify: {e}")
    print("Make sure DeOldify is properly installed and dependencies are met.")
    sys.exit(1)

class Colorizer:
    def __init__(self, artistic=True, weights_path=None):
        self.artistic = artistic
        self.model_path = weights_path
        
        # Set default model paths if not provided
        if self.model_path is None:
            models_dir = deoldify_path / 'models'
            if self.artistic:
                self.model_path = models_dir / 'ColorizeArtistic_gen.pth'
            else:
                self.model_path = models_dir / 'ColorizeStable_gen.pth'
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Initialize the colorizer
        try:
            self.colorizer = get_image_colorizer(artistic=self.artistic)
            print(f"Initialized {'artistic' if self.artistic else 'stable'} colorizer")
        except Exception as e:
            print(f"Error initializing colorizer: {e}")
            raise
    
    def colorize(self, image_path, render_factor=35, output_path=None):
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Colorize the image
        try:
            result = self.colorizer.get_transformed_image(
                str(image_path), 
                render_factor=render_factor
            )
        except Exception as e:
            print(f"Error during colorization: {e}")
            raise
        
        # Save or return the result
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            result.save(output_path)
            return output_path
        else:
            return result
""")
    
    print(f"Created wrapper module at: {wrapper_dir}")
    return wrapper_dir

def create_colorization_script(wrapper_dir):
    """Create a simple script to colorize images using DeOldify."""
    script_path = "colorize_deoldify.py"
    
    script_content = f"""
import os
import sys
import argparse
from pathlib import Path

# Import the wrapper module
from {wrapper_dir}.colorizer import Colorizer

def colorize_image(input_path, output_dir="colored_images", render_factor=35, artistic=True):
    print(f"Colorizing image: {{input_path}}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get output path
    input_filename = os.path.basename(input_path)
    name, ext = os.path.splitext(input_filename)
    output_path = os.path.join(output_dir, f"{{name}}_colorized{{ext}}")
    
    # Initialize colorizer
    try:
        colorizer = Colorizer(artistic=artistic)
        
        # Colorize the image
        colorizer.colorize(input_path, render_factor=render_factor, output_path=output_path)
        
        print(f"Colorized image saved to: {{output_path}}")
        return output_path
    except Exception as e:
        print(f"Error during colorization: {{e}}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Colorize black and white images using DeOldify")
    parser.add_argument("image_path", help="Path to the black and white image")
    parser.add_argument("--output_dir", default="colored_images", help="Directory to save colorized images")
    parser.add_argument("--render_factor", type=int, default=35, help="Render factor (higher = better quality but slower)")
    parser.add_argument("--artistic", action="store_true", help="Use artistic model (more vibrant colors but less reliable)")
    parser.add_argument("--stable", dest="artistic", action="store_false", help="Use stable model (more reliable but less vibrant)")
    parser.set_defaults(artistic=True)
    
    args = parser.parse_args()
    
    # Print system info
    import torch
    print(f"PyTorch version: {{torch.__version__}}")
    print(f"CUDA available: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        print(f"CUDA version: {{torch.version.cuda}}")
        print(f"GPU: {{torch.cuda.get_device_name(0)}}")
        print("Using GPU for colorization.")
    else:
        print("Using CPU for colorization (this will be slow).")
    
    # Colorize the image
    try:
        output_path = colorize_image(
            args.image_path, 
            args.output_dir, 
            args.render_factor, 
            args.artistic
        )
        
        print(f"Successfully colorized image at: {{output_path}}")
    except Exception as e:
        print(f"Colorization failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    with open(script_path, "w") as f:
        f.write(script_content.strip())
    
    print(f"Colorization script created at: {script_path}")
    return script_path

def create_run_script(env_name, script_path):
    """Create a shell script to activate the environment and run the colorization."""
    if os.name == 'nt':  # Windows
        run_script = "run_deoldify.bat"
        script_content = f"""
@echo off
call {env_name}\\Scripts\\activate.bat
python {script_path} %*
"""
    else:  # Unix/Linux/Mac
        run_script = "run_deoldify.sh"
        script_content = f"""#!/bin/bash
source {env_name}/bin/activate
python {script_path} "$@"
"""
    
    with open(run_script, "w") as f:
        f.write(script_content.strip())
    
    # Make the script executable on Unix systems
    if os.name != 'nt':
        os.chmod(run_script, 0o755)
    
    print(f"Run script created at: {run_script}")
    return run_script

def main():
    parser = argparse.ArgumentParser(description="Set up DeOldify from GitHub repository")
    parser.add_argument("--env_name", default="deoldify-env", help="Name of the virtual environment")
    args = parser.parse_args()
    
    # Create virtual environment
    env_name = create_virtual_env(args.env_name)
    
    # Install dependencies
    install_dependencies(env_name)
    
    # Download models
    download_models()
    
    # Create wrapper module
    wrapper_dir = create_wrapper_module()
    
    # Create the colorization script
    script_path = create_colorization_script(wrapper_dir)
    
    # Create run script
    run_script = create_run_script(env_name, script_path)
    
    print("\nSetup complete! To colorize an image, run:")
    if os.name == 'nt':  # Windows
        print(f"  .\\{run_script} your_image.jpg")
        print("\nFor higher quality (but slower) results:")
        print(f"  .\\{run_script} your_image.jpg --render_factor 45")
    else:  # Unix/Linux/Mac
        print(f"  ./{run_script} your_image.jpg")
        print("\nFor higher quality (but slower) results:")
        print(f"  ./{run_script} your_image.jpg --render_factor 45")
    
    print("\nOptions:")
    print("  --artistic      Use artistic model (more vibrant colors but less reliable) [default]")
    print("  --stable        Use stable model (more reliable colors)")
    print("  --render_factor Value between 7-45 (higher = better quality but slower) [default: 35]")

if __name__ == "__main__":
    main()