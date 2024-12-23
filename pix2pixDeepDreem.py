import os
import pathlib
import numpy as np
import cv2
from skimage import feature
import tensorflow as tf
import torch
from torchvision.transforms import v2
from py5canvas import *
from dearpygui.dearpygui import *

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
w, h = 512, 512
vid = VideoInput(size=(w, h))

# Available models
models = {
    "Pix2Pix - Edge to Comics": "pix2pix_edge2comics.iter_16583_scripted.pt",
    "Pix2Pix - Face to Comics": "pix2pix_face2comics.iter_270000_scripted.pt",
}
current_model_path = pathlib.Path(models["Pix2Pix - Face to Comics"])

# Ensure the default model file exists
if not current_model_path.exists():
    raise FileNotFoundError(f"Model file not found at {current_model_path}")

# Load the Pix2Pix model
G = torch.jit.load(current_model_path, map_location=device)
print(f"Loaded model: {current_model_path}")

# Initialize DeepDream model
base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
layers = [base_model.get_layer(name).output for name in ["mixed3", "mixed5"]]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

# Global default values
default_values = {
    "blend_factor": 0.5,
    "noise_intensity": 0.1,
    "brightness": 1.0,
    "contrast": 1.0,
    "style": "default",
    "generation_mode": "Pix2Pix",  # Options: "Pix2Pix", "DeepDream"
    "current_model": "Pix2Pix - Face to Comics",  # Ensure this matches a key in `models`
}

# Function to dynamically load a Pix2Pix model
def load_model(selected_model):
    """Load a new Pix2Pix model based on user selection."""
    global G, current_model_path
    current_model_path = pathlib.Path(models[selected_model])
    if not current_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {current_model_path}")
    G = torch.jit.load(current_model_path, map_location=device)
    print(f"Loaded model: {current_model_path}")

# DeepDream functions
def calc_loss(img, model):
    """Calculate loss for DeepDream by maximizing activations."""
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        return tf.reduce_mean(layer_activations[0])
    return tf.reduce_sum([tf.reduce_mean(act) for act in layer_activations])

@tf.function
def deep_dream_step(img, model, step_size):
    """Perform one optimization step for DeepDream."""
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    gradients = tape.gradient(loss, img)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    img = img + step_size * gradients
    img = tf.clip_by_value(img, -1, 1)
    return img

def deep_dream(image, steps=10, step_size=0.01):
    """Apply DeepDream to an input image."""
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))  # Resize to model input
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    for step in range(steps):
        img = deep_dream_step(img, dream_model, step_size)
    return img.numpy()

# Generate function for Pix2Pix
def generate_pix2pix(model, image, blend_factor, noise_intensity, brightness, contrast):
    """Generate Pix2Pix output with enhancements."""
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
    output_np = output.numpy()

    # Post-process output
    output_np = np.clip(contrast * output_np + brightness - 1.0, 0, 1)
    noise = np.random.normal(0, noise_intensity, output_np.shape)
    output_np = np.clip(output_np + noise, 0, 1)

    return output_np

# GUI setup
def slider_callback(sender, app_data):
    """Update default values when sliders are adjusted."""
    default_values[sender] = app_data
    print(f"Updated {sender} to {app_data}")  # Debug output for testing

def setup_gui():
    """Set up GUI with sliders and controls."""
    with window(label="Adjust Effects", width=400, height=500):
        add_combo(label="Generation Mode", tag="generation_mode", items=["Pix2Pix", "DeepDream"], default_value="Pix2Pix")
        add_combo(label="Model", tag="current_model", items=list(models.keys()), default_value="Pix2Pix - Face to Comics")
        add_slider_float(label="Blend Factor", tag="blend_factor", default_value=0.5, min_value=0.0, max_value=1.0, callback=slider_callback)
        add_slider_float(label="Noise Intensity", tag="noise_intensity", default_value=0.1, min_value=0.0, max_value=1.0, callback=slider_callback)
        add_slider_float(label="Brightness", tag="brightness", default_value=1.0, min_value=0.5, max_value=2.0, callback=slider_callback)
        add_slider_float(label="Contrast", tag="contrast", default_value=1.0, min_value=0.5, max_value=2.0, callback=slider_callback)
        add_button(label="Exit", callback=stop_dearpygui)

# Main draw loop
def draw():
    """Main rendering loop."""
    try:
        # Get values from GUI
        generation_mode = get_value("generation_mode")
        selected_model = get_value("current_model")
        blend_factor = default_values["blend_factor"]
        noise_intensity = default_values["noise_intensity"]
        brightness = default_values["brightness"]
        contrast = default_values["contrast"]

        # Load model if changed
        if selected_model != default_values["current_model"]:
            default_values["current_model"] = selected_model
            load_model(selected_model)

        # Fetch the current video frame
        frame = vid.read()

        # Generate result based on selected mode
        if generation_mode == "Pix2Pix":
            result = generate_pix2pix(G, frame, blend_factor, noise_intensity, brightness, contrast)
        else:
            result = deep_dream(frame, steps=10, step_size=0.02)

        # Render the result
        background(0)
        image(result, [0, 0], [width, height])
    except Exception as e:
        print(f"Error in draw loop: {e}")

# Initialization
if __name__ == "__main__":
    create_context()
    setup_gui()
    create_viewport(title="Pix2Pix & DeepDream", width=600, height=500)
    setup_dearpygui()
    show_viewport()

    # Run GUI in a separate thread
    import threading
    gui_thread = threading.Thread(target=start_dearpygui, daemon=True)
    gui_thread.start()

    # Run the Py5 canvas visualization
    run()
