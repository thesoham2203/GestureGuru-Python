import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the original model
model = load_model("Model/keras_model.h5", compile=False)

# Convert model to JSON
model_json = model.to_json()

# Remove 'groups': 1 manually from JSON string
import json
model_config = json.loads(model_json)

for layer in model_config["config"]["layers"]:
    if layer["class_name"] == "DepthwiseConv2D":
        if "groups" in layer["config"]:
            del layer["config"]["groups"]  # Remove 'groups' key

# Save the fixed model
fixed_model = tf.keras.models.model_from_json(json.dumps(model_config))
fixed_model.save("Model/fixed_model.h5")
print("Fixed model saved successfully.")
