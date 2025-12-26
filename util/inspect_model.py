
import tensorflow as tf
import os

MODEL_PATH = 'best_efficient_tuned_model_EfficientNetV2B0.keras'

def inspect():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Model Summary ---")
    model.summary()

    print("\n--- Layers ---")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
        if isinstance(layer, tf.keras.Model):
            print(f"  -> This is a nested model. Inputs: {layer.input_shape}, Outputs: {layer.output_shape}")
            # print("  -> Nested model summary:")
            # layer.summary()
            
            # Check if we can access its output
            try:
                print(f"  -> Layer output: {layer.output}")
            except Exception as e:
                print(f"  -> Could not access output: {e}")

if __name__ == "__main__":
    inspect()
