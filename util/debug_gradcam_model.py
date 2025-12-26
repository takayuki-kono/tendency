import tensorflow as tf
import os

def inspect():
    model_path = "best_efficient_tuned_model_EfficientNetV2B0.keras"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Model inputs: {model.inputs}")
    print(f"Model outputs: {model.outputs}")

    base_model = None
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} (Type: {type(layer).__name__})")
        if 'efficientnet' in layer.name.lower() or 'resnet' in layer.name.lower() or 'densenet' in layer.name.lower():
            base_model = layer
            print(f"  -> Found base model candidate: {layer.name}")

    if base_model:
        print("\nInspecting Base Model:")
        try:
            print(f"  base_model.input: {base_model.input}")
        except Exception as e:
            print(f"  base_model.input error: {e}")
            
        try:
            print(f"  base_model.output: {base_model.output}")
        except Exception as e:
            print(f"  base_model.output error: {e}")

        # Check if it's a Model instance
        if isinstance(base_model, tf.keras.Model):
            print("  Base model is a tf.keras.Model instance.")
            # Try to see internal layers
            try:
                print(f"  Internal layers count: {len(base_model.layers)}")
                print(f"  First internal layer: {base_model.layers[0].name}")
            except:
                pass
        else:
            print("  Base model is NOT a tf.keras.Model instance (just a Layer).")

        # Try to construct the grad model graph here to reproduce error
        try:
            print("\nAttempting to create grad_model...")
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[base_model.output, model.output]
            )
            print("  grad_model created successfully.")
        except Exception as e:
            print(f"  grad_model creation failed: {e}")

if __name__ == "__main__":
    inspect()
