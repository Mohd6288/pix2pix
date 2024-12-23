import cv2
import numpy as np
import tensorflow as tf
import threading
import logging
from functools import partial

from py5canvas import *
from dearpygui.dearpygui import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define video dimensions globally
w, h = 512 * 2, 512 * 2  # Example: 1024x1024 resolution

# Device configuration using updated TensorFlow method
device = "cuda" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu"
logging.info(f"Using {device} device")

# Initialize DeepDream model
try:
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    layers = [base_model.get_layer(name).output for name in ["mixed3", "mixed5"]]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    logging.info("DeepDream model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize DeepDream model: {e}")
    raise e

# Camera initialization
camera = cv2.VideoCapture(0)  # Use default camera
if not camera.isOpened():
    logging.error("Failed to open camera")
    raise RuntimeError("Failed to open camera")
logging.info("Camera initialized successfully.")

frame_count = 0

# Lock for thread-safe access to GUI values
gui_lock = threading.Lock()

# Global default values for GUI sliders and modes
default_values = {
    "step_size": 0.01,
    "steps": 10,
    "brightness": 1.0,
    "contrast": 1.0,
}

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

def deep_dream(image, steps=10, step_size=0.01, width=w, height=h):
    """Apply DeepDream to an input image."""
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))  # Resize to model input
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    for step in range(steps):
        img = deep_dream_step(img, dream_model, step_size)
    img = img.numpy()
    img = (img + 1.0) / 2.0  # Normalize to [0,1]
    img = np.clip(img, 0, 1)
    img = cv2.resize(img, (width, height))  # Resize back to original dimensions
    img = (img * 255).astype(np.uint8)
    return img

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """
    Adjusts brightness and contrast of an image.
    """
    image = image.astype(np.float32)
    image = image * contrast
    image = image + (brightness - 1.0) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# GUI callbacks
def slider_callback(sender, app_data, user_data):
    with gui_lock:
        default_values[sender] = app_data
    logging.debug(f"Updated {sender} to {app_data}")

def combo_callback(sender, app_data, user_data):
    with gui_lock:
        default_values[sender] = app_data
    logging.debug(f"Updated {sender} to {app_data}")

def exit_callback(sender, app_data):
    logging.info("Exiting application...")
    stop_dearpygui()
    py5.stop()
    cleanup()

# GUI setup
def setup_gui():
    """Set up GUI with sliders and controls."""
    with window(label="Adjust Effects", width=400, height=300):
        add_slider_float(label="Step Size", tag="step_size", default_value=default_values["step_size"],
                        min_value=0.001, max_value=0.1, callback=partial(slider_callback))
        add_slider_int(label="Steps", tag="steps", default_value=default_values["steps"],
                      min_value=1, max_value=50, callback=partial(slider_callback))
        add_slider_float(label="Brightness", tag="brightness", default_value=default_values["brightness"],
                        min_value=0.5, max_value=2.0, callback=partial(slider_callback))
        add_slider_float(label="Contrast", tag="contrast", default_value=default_values["contrast"],
                        min_value=0.5, max_value=2.0, callback=partial(slider_callback))
        add_button(label="Exit", callback=exit_callback)

# Py5 visualization setup
def setup():
    create_canvas(w, h)
    frame_rate(20)  # Adjust as needed for performance

def draw():
    """Main rendering loop."""
    global frame_count
    try:
        # Read the frame from the camera
        ret, frame = camera.read()
        if not ret:
            logging.warning("Failed to read frame from camera")
            background(0)
            fill(255)
            text("No video feed detected.", 20, 20)
            return

        frame_count += 1
        processed_frame = frame.copy()

        # Process every 10th frame
        if frame_count % 10 == 0:
            with gui_lock:
                step_size = default_values.get("step_size", 0.01)
                steps = default_values.get("steps", 10)
                brightness = default_values.get("brightness", 1.0)
                contrast = default_values.get("contrast", 1.0)

            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply DeepDream
            result = deep_dream(frame_rgb, steps=steps, step_size=step_size, width=w, height=h)

            # Adjust brightness and contrast
            result = adjust_brightness_contrast(result, brightness=brightness, contrast=contrast)

            processed_frame = result

        # Render the result
        background(0)
        image(processed_frame, [0, 0], [width, height])
    except Exception as e:
        logging.error(f"Error in draw loop: {e}")

# Cleanup
def cleanup():
    """Release resources on exit."""
    if camera.isOpened():
        camera.release()
        logging.info("Camera released.")
    else:
        logging.info("Camera was already released.")

# Main execution
def main():
    create_context()
    setup_gui()
    create_viewport(title="DeepDream Visualizer", width=600, height=500)
    setup_dearpygui()
    show_viewport()

    # Run GUI in a separate thread
    gui_thread = threading.Thread(target=start_dearpygui, daemon=True)
    gui_thread.start()

    try:
        # Run the Py5 canvas visualization
        run()
    except Exception as e:
        logging.error(f"Error in Py5 run loop: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
