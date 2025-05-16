"""
DeOldify API Test Client

A simple script to test the DeOldify API by sending an image and displaying the result.
"""
import os
import requests
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import io

def colorize_image(api_url, image_path, render_factor=35, model_type="artistic"):
    """
    Send an image to the DeOldify API for colorization.
    
    Args:
        api_url (str): The base URL of the DeOldify API
        image_path (str): Path to the input image file
        render_factor (int): Quality factor (10-45)
        model_type (str): Either "artistic" or "stable"
        
    Returns:
        tuple: (success (bool), result (dict or error message), colorized_image (PIL.Image or None))
    """
    # Construct the API endpoint URL
    colorize_url = f"{api_url.rstrip('/')}/colorize"
    
    # Prepare the files and data for the request
    files = {
        "file": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")
    }
    
    data = {
        "render_factor": render_factor,
        "model_type": model_type
    }
    
    print(f"Sending request to {colorize_url}")
    print(f"Image: {image_path}")
    print(f"Render factor: {render_factor}")
    print(f"Model type: {model_type}")
    
    try:
        # Send the request to the API
        response = requests.post(colorize_url, files=files, data=data)
        
        # Close the file
        files["file"][1].close()
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("Colorization successful!")
            
            # Get the colorized image URL and download it
            image_url = result["colorized_image_url"]
            full_image_url = f"{api_url.rstrip('/')}{image_url}"
            
            print(f"Downloading colorized image from: {full_image_url}")
            image_response = requests.get(full_image_url)
            
            if image_response.status_code == 200:
                # Load the image data into a PIL Image
                colorized_image = Image.open(io.BytesIO(image_response.content))
                return True, result, colorized_image
            else:
                return False, f"Failed to download colorized image: {image_response.status_code}", None
        else:
            error_msg = f"API request failed with status code {response.status_code}"
            try:
                error_detail = response.json().get("detail", "No error details")
                error_msg += f": {error_detail}"
            except:
                pass
            return False, error_msg, None
            
    except Exception as e:
        return False, f"Error connecting to API: {str(e)}", None

def main():
    parser = argparse.ArgumentParser(description="Test the DeOldify API")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--api_url", default="http://localhost:8000", help="URL of the DeOldify API")
    parser.add_argument("--render_factor", type=int, default=35, help="Render factor (10-45)")
    parser.add_argument("--model_type", choices=["artistic", "stable"], default="artistic", help="Model type")
    parser.add_argument("--save", action="store_true", help="Save the colorized image")
    parser.add_argument("--output", help="Output file path (if not specified, will use input filename with _colorized suffix)")
    
    args = parser.parse_args()
    
    # Verify that the input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Input image not found at {args.image_path}")
        return
    
    # Determine the output path if saving is enabled
    output_path = None
    if args.save:
        if args.output:
            output_path = args.output
        else:
            input_filename, input_ext = os.path.splitext(args.image_path)
            output_path = f"{input_filename}_colorized{input_ext}"
    
    # Colorize the image
    success, result, colorized_image = colorize_image(
        args.api_url,
        args.image_path,
        args.render_factor,
        args.model_type
    )
    
    if success and colorized_image:
        # Display the result
        plt.figure(figsize=(15, 10))
        
        # Load and display the original image
        original_image = Image.open(args.image_path)
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(original_image, cmap='gray' if original_image.mode == 'L' else None)
        plt.axis('off')
        
        # Display the colorized image
        plt.subplot(1, 2, 2)
        plt.title("Colorized")
        plt.imshow(colorized_image)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the colorized image if requested
        if output_path:
            print(f"Saving colorized image to {output_path}")
            colorized_image.save(output_path)
        
        plt.show()
        
        print("Done!")
    else:
        print(f"Colorization failed: {result}")

if __name__ == "__main__":
    main()