import os
import pathlib
import numpy as np
import cv2
from py5canvas import *
from dearpygui.dearpygui import *
import torch
from torchvision.transforms import v2

# Device configuration
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Set video dimensions and initialize video input
w, h = 512 * 2, 512 * 2
vid = VideoInput(size=(w, h))

# Paths to pre-trained models
PIX2PIX_PATH = pathlib.Path("pix2pix_face2comics.iter_180000_scripted.pt")
REMBRANDT_PATH = pathlib.Path("pix2pix_rembrandt.iter_10879_scripted.pt")

# Ensure the model files exist before loading
if not PIX2PIX_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {PIX2PIX_PATH}")

if not REMBRANDT_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {REMBRANDT_PATH}")

# Load the Pix2Pix model
G = torch.jit.load(PIX2PIX_PATH, map_location=device)
print(f"Pix2Pix model loaded with {sum(p.numel() for p in G.parameters()):,} parameters.")

# Load the Rembrandt model
R = torch.jit.load(REMBRANDT_PATH, map_location=device)
print(f"Rembrandt model loaded with {sum(p.numel() for p in R.parameters()):,} parameters.")

# Global default values for GUI sliders and modes
default_values = {
    "blend_factor": 0.5,
    "noise_intensity": 0.1,
    "brightness": 1.0,
    "contrast": 1.0,
    "style": "default",
    "full_mode": "none",  # Options: "original", "edges", "generated", "none"
}

# Function to add noise to an image
def add_noise(image, intensity=0.1):
    noise = np.random.normal(0, intensity, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image

# Function to adjust brightness and contrast of an image
def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.clip(contrast * image + (brightness - 1.0), 0, 1)
    return (image * 255).astype(np.uint8)

# Generate a transformed image using a specified model
def generate(model, image):
    image = torch.permute(torch.tensor(image.copy()), (2, 0, 1))
    image = v2.ToImage()(image)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3, :, :]

    image = v2.Resize((256, 256), antialias=True)(image)
    image = v2.ToDtype(torch.float32, scale=True)(image).to(device)[None, ...]

    with torch.no_grad():
        outputs = model(image).detach().cpu()

    output = outputs[0].permute(1, 2, 0) * 0.5 + 0.5  # Normalize to [0, 1]
    return (output.numpy() * 255).astype(np.uint8)

# Apply edge detection using OpenCV (commented out as it's not currently used)
# def apply_canny_cv2(img, thresh1=100, thresh2=200):
#     grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(grey_img, thresh1, thresh2)
#     return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# GUI setup
def setup_gui():
    with window(label="Adjust Effects", width=400, height=300):
        add_slider_float(label="Blend Factor", tag="blend_factor", default_value=default_values["blend_factor"], min_value=0.0, max_value=1.0)
        add_slider_float(label="Noise Intensity", tag="noise_intensity", default_value=default_values["noise_intensity"], min_value=0.0, max_value=1.0)
        add_slider_float(label="Brightness", tag="brightness", default_value=default_values["brightness"], min_value=0.5, max_value=2.0)
        add_slider_float(label="Contrast", tag="contrast", default_value=default_values["contrast"], min_value=0.5, max_value=2.0)
        add_combo(label="Full Mode", tag="full_mode", items=["none", "original", "edges", "generated"], default_value=default_values["full_mode"])
        add_button(label="Exit", callback=stop_dearpygui)

# Py5 visualization setup
def setup():
    create_canvas(w, h)
    frame_rate(20)

# Draw frames and process them in real-time
def draw():
    # Fetch real-time GUI values
    blend_factor = get_value("blend_factor")
    noise_intensity = get_value("noise_intensity")
    brightness = get_value("brightness")
    full_mode = get_value("full_mode")
    contrast = get_value("contrast")

    # Read frame from the video input
    frame = vid.read()

    # Validate frame
    if frame is None:
        background(0)
        text("No video feed detected.", 20, 20)
        return

    # Apply adjustments
    frame = adjust_brightness_contrast(frame, brightness=brightness, contrast=contrast)
    frame = add_noise(frame, intensity=noise_intensity)

    # Generate outputs from both models
    rembrandt_output = generate(R, frame)
    pix2pix_output = generate(G, frame)

    # Display frames based on the selected mode
    background(0)

    if full_mode == "original":
        image(frame, [0, 0], [w, h])
    elif full_mode == "edges":
        image(rembrandt_output, [0, 0], [w, h])
    elif full_mode == "generated":
        image(pix2pix_output, [0, 0], [w, h])
    else:  # Default to split view
        image(frame, [0, 0], [w // 3, h])
        image(rembrandt_output, [w // 3, 0], [w // 3, h])
        image(pix2pix_output, [2 * w // 3, 0], [w // 3, h])

# Main execution
if __name__ == "__main__":
    create_context()
    setup_gui()
    create_viewport(title="Pix2Pix with GUI", width=800, height=400)
    setup_dearpygui()
    show_viewport()

    import threading

    # Run GUI in a separate thread
    gui_thread = threading.Thread(target=start_dearpygui, daemon=True)
    gui_thread.start()

    # Run Py5 visualization
    run()
