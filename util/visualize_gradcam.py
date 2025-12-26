import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

# Set backend to Agg to avoid display issues in non-interactive environments
import matplotlib
matplotlib.use('Agg')

def get_img_array(img_path, size):
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, output_name=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    # Find the base model (EfficientNet) within the full model
    # The structure is: Input -> Augmentation -> Preprocess -> BaseWrapper -> Pooling -> Dense...
    # We need to skip the augmentation layers and find the base model layer.

    # Search for the base model layer
    base_model = None
    for layer in model.layers:
        # Check for common base model names or types
        if 'efficientnet' in layer.name.lower() or 'resnet' in layer.name.lower() or 'densenet' in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError(f"Could not find base model layer with name containing 'efficientnet', 'resnet', or 'densenet'")

    print(f"Found base model: {base_model.name}")

    # Create a new inference model that bypasses data augmentation
    # We'll manually apply preprocessing to the input
    # Structure: PreprocessedInput -> BaseModel -> ... -> Outputs

    # Get the preprocess function
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

    if 'efficientnet' in base_model.name.lower():
        preprocess_func = efficientnet_preprocess
    elif 'resnet' in base_model.name.lower():
        preprocess_func = resnet_preprocess
    elif 'densenet' in base_model.name.lower():
        preprocess_func = densenet_preprocess
    else:
        preprocess_func = efficientnet_preprocess  # default

    # Apply preprocessing to input
    preprocessed_input = preprocess_func(img_array)

    # Build a new model that starts from the base model
    # We need to trace through the graph from base_model output to final outputs

    # Find the base model's output in the full model's computational graph
    base_model_output = None

    # Check if base_model has multiple nodes (uses in the graph)
    if hasattr(base_model, '_inbound_nodes') and len(base_model._inbound_nodes) > 0:
        # Get the most recent node (the one used in this model)
        node = base_model._inbound_nodes[-1]
        base_model_output = node.output_tensors[0] if isinstance(node.output_tensors, list) else node.output_tensors
    else:
        # Fallback: recreate the call
        # This may not work if the model has complex structure
        raise ValueError("Cannot determine base model output from graph")

    print(f"Base model output tensor: {base_model_output}")

    # Create grad model with preprocessed input
    # We need to create a simpler path: BaseModel(input) -> outputs
    # Let's build this by calling the base model and subsequent layers

    # Alternative approach: Create a functional model from base_model input to outputs
    # We'll trace the layers after the base model

    # Find all layers after the base model
    base_model_index = model.layers.index(base_model)
    subsequent_layers = model.layers[base_model_index + 1:]

    print(f"Subsequent layers: {[l.name for l in subsequent_layers]}")

    # Build the forward pass manually
    new_input = tf.keras.Input(shape=(224, 224, 3))
    x = new_input

    # Apply base model
    x = base_model(x, training=False)
    base_output_for_grad = x

    # Apply subsequent layers up to the branching point
    # Find the last common layer before task outputs
    task_output_layers = []
    shared_layers = []

    for layer in subsequent_layers:
        # Skip input layers
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        # Check if this is a task output layer
        if 'task_' in layer.name and '_output' in layer.name:
            task_output_layers.append(layer)
        else:
            shared_layers.append(layer)

    print(f"Shared layers: {[l.name for l in shared_layers]}")
    print(f"Task output layers: {[l.name for l in task_output_layers]}")

    # Apply shared layers sequentially
    for layer in shared_layers:
        x = layer(x)

    # Now x is the output of the last shared layer
    # Apply each task output layer to this
    task_outputs = []
    for task_layer in task_output_layers:
        task_outputs.append(task_layer(x))

    # Create the grad model
    grad_model = tf.keras.models.Model(
        inputs=new_input,
        outputs=[base_output_for_grad] + task_outputs
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Use preprocessed input
        outputs = grad_model(preprocessed_input)
        last_conv_layer_output = outputs[0]  # First output is the base model output
        preds = outputs[1:]  # Remaining are task predictions

        # preds is a list of outputs (multi-task).
        # We need to pick one task to visualize. Default to the first one or specified name.
        if output_name:
            # Find the index of the output_name
            target_output_index = 0
            for i, name in enumerate(model.output_names):
                if output_name in name:
                    target_output_index = i
                    break
            target_pred = preds[target_output_index]
        else:
            target_pred = preds[0]  # Default to first task

        if pred_index is None:
            pred_index = tf.argmax(target_pred[0])

        class_channel = target_pred[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, prediction_info=None):
    # Load the original image
    img = cv2.imread(img_path)

    # Resize heatmap to match original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply colormap
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    print(f"Saved Grad-CAM image to {cam_path}")

    # Print prediction info if available
    if prediction_info:
        print(f"\nPrediction details:")
        print(f"  Task: {prediction_info['task_name']}")
        print(f"  Predicted class: {prediction_info['predicted_class']} (index: {prediction_info['class_index']})")
        print(f"  Confidence: {prediction_info['confidence']:.2%}")
        print(f"  All probabilities: {prediction_info['probabilities']}")

def save_multi_task_gradcam(img_path, heatmaps, predictions_info, cam_path="cam_all_tasks.jpg", alpha=0.6):
    """
    各タスクのヒートマップを異なる色で表示

    Args:
        heatmaps: list of 4 heatmaps (one per task)
        predictions_info: list of prediction info dicts
        alpha: overlay transparency
    """
    # Load the original image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # タスクごとの色（RGB）
    task_colors = [
        (1.0, 0.0, 0.0),  # Task A: 赤
        (0.0, 1.0, 0.0),  # Task B: 緑
        (0.0, 0.0, 1.0),  # Task C: 青
        (1.0, 1.0, 0.0),  # Task D: 黄色
    ]

    task_names = ["拡張ゾーン", "内向/外向", "水平/垂直思考", "耳横拡張"]

    # 合成用の空画像（RGB）
    combined_heatmap = np.zeros((h, w, 3), dtype=np.float32)

    print("\n=== Multi-Task Grad-CAM ===")
    for i, (heatmap, pred_info, color) in enumerate(zip(heatmaps, predictions_info, task_colors)):
        # Resize and normalize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_normalized = heatmap_resized / (heatmap_resized.max() + 1e-8)

        # Apply task-specific color
        for c in range(3):
            combined_heatmap[:, :, c] += heatmap_normalized * color[c] * 255

        print(f"Task {i} ({task_names[i]}): {pred_info['predicted_class']} ({pred_info['confidence']:.1%})")

    # Clip values to [0, 255]
    combined_heatmap = np.clip(combined_heatmap, 0, 255).astype(np.uint8)

    # Superimpose on original image
    superimposed = cv2.addWeighted(img, 1-alpha, combined_heatmap, alpha, 0)

    # Add legend
    legend_height = 120
    legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255

    for i, (color, name, pred_info) in enumerate(zip(task_colors, task_names, predictions_info)):
        y_pos = 20 + i * 25
        # Color box
        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # RGB to BGR
        cv2.rectangle(legend, (10, y_pos-10), (30, y_pos+5), color_bgr, -1)
        # Text
        text = f"Task {chr(65+i)} ({name}): {pred_info['predicted_class']} ({pred_info['confidence']:.1%})"
        cv2.putText(legend, text, (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Combine image and legend
    result = np.vstack([superimposed, legend])

    # Save
    cv2.imwrite(cam_path, result)
    print(f"\nSaved multi-task Grad-CAM to {cam_path}")
    print("Color legend: Red=Task A, Green=Task B, Blue=Task C, Yellow=Task D")

def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM")
    parser.add_argument("--image_path", required=True, help="Path to the image to visualize")
    parser.add_argument("--model_path", default="best_efficient_tuned_model_EfficientNetV2B0.keras", help="Path to the trained model")
    parser.add_argument("--task_index", type=int, default=0, help="Index of the task to visualize (0-3)")
    parser.add_argument("--all_tasks", action="store_true", help="Visualize all tasks with different colors in one image")
    parser.add_argument("--output_dir", default="vis_output", help="Directory to save results")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading model from {args.model_path}...")
    try:
        # Need to provide custom objects if any, but standard layers should be fine.
        # If using mixed precision, we might need to set policy or load with compile=False
        model = tf.keras.models.load_model(args.model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Processing image {args.image_path}...")
    img_size = (224, 224) # Assumed from training script
    img_array = get_img_array(args.image_path, img_size)

    # First, get predictions from the model
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    preprocessed = preprocess_input(img_array.copy())
    predictions = model.predict(preprocessed, verbose=0)

    # Task labels from training script
    TASK_LABELS = [
        ['a', 'b', 'c'],  # Task A
        ['d', 'e'],        # Task B
        ['f', 'g'],        # Task C
        ['h', 'i']         # Task D
    ]
    task_names = ["拡張ゾーン", "内向/外向", "水平/垂直思考", "耳横拡張"]

    if args.all_tasks:
        # Generate heatmaps for all tasks
        print("\nGenerating Grad-CAM for all tasks...")
        all_heatmaps = []
        all_predictions = []
        filename = os.path.basename(args.image_path)

        for task_idx in range(4):
            target_output_name = f'task_{chr(97+task_idx)}_output'
            task_pred = predictions[task_idx]
            class_index = np.argmax(task_pred[0])
            confidence = task_pred[0][class_index]
            predicted_class = TASK_LABELS[task_idx][class_index]

            prediction_info = {
                'task_name': f"{target_output_name} ({task_names[task_idx]})",
                'predicted_class': predicted_class,
                'class_index': class_index,
                'confidence': confidence,
                'probabilities': [f"{TASK_LABELS[task_idx][i]}:{task_pred[0][i]:.3f}" for i in range(len(task_pred[0]))]
            }

            print(f"Task {task_idx} ({task_names[task_idx]}): {predicted_class} ({confidence:.1%})")

            # Generate heatmap
            heatmap = make_gradcam_heatmap(img_array, model, "efficientnetv2b0_base",
                                          pred_index=class_index, output_name=target_output_name)

            all_heatmaps.append(heatmap)
            all_predictions.append(prediction_info)

            # Save individual task visualization
            individual_save_path = os.path.join(args.output_dir, f"gradcam_task{task_idx}_{predicted_class}_{filename}")
            save_and_display_gradcam(args.image_path, heatmap, individual_save_path, prediction_info=prediction_info)

        # Save multi-task visualization
        save_path = os.path.join(args.output_dir, f"gradcam_all_tasks_{filename}")
        save_multi_task_gradcam(args.image_path, all_heatmaps, all_predictions, save_path)

        print(f"\nGenerated {len(all_heatmaps)} individual task images + 1 combined image")

    else:
        # Single task visualization
        target_output_name = f'task_{chr(97+args.task_index)}_output'
        print(f"\nVisualizing attention for task: {target_output_name}")

        # Get prediction for this task
        task_pred = predictions[args.task_index]
        class_index = np.argmax(task_pred[0])
        confidence = task_pred[0][class_index]
        predicted_class = TASK_LABELS[args.task_index][class_index]

        prediction_info = {
            'task_name': f"{target_output_name} ({task_names[args.task_index]})",
            'predicted_class': predicted_class,
            'class_index': class_index,
            'confidence': confidence,
            'probabilities': [f"{TASK_LABELS[args.task_index][i]}:{task_pred[0][i]:.3f}" for i in range(len(task_pred[0]))]
        }

        print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")

        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, "efficientnetv2b0_base",
                                       pred_index=class_index, output_name=target_output_name)

        # Save result
        filename = os.path.basename(args.image_path)
        save_path = os.path.join(args.output_dir, f"gradcam_task{args.task_index}_{predicted_class}_{filename}")
        save_and_display_gradcam(args.image_path, heatmap, save_path, prediction_info=prediction_info)

if __name__ == "__main__":
    main()
