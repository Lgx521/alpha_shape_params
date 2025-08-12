# Adaptive Alpha Shape Inference and Visualization

This script provides a powerful tool to visualize the output of a trained PointNet++ model designed to predict per-point alpha values for 3D surface reconstruction. It loads a point cloud from the ShapeNet dataset, runs inference to get the alpha values, and displays a side-by-side comparison of the input points, an alpha value heatmap, and the final reconstructed mesh using Open3D's alpha shape algorithm.

  <!-- You can replace this with a real screenshot -->

## Key Features

-   **Model Inference**: Loads a pre-trained PyTorch model (`.pth` file) for inference.
-   **Per-Point Alpha Prediction**: Runs a point cloud sample through the network to predict an optimal alpha value for each point.
-   **CGAL-Free Reconstruction**: Uses the built-in `open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape` for mesh creation, removing the need for CGAL.
-   **Rich Visualization**: Provides an interactive Open3D window with three views:
    1.  **Alpha Heatmap (Left)**: The input point cloud colored by the predicted alpha value.
    2.  **Original Point Cloud (Center)**: The raw input points, shown in gray.
    3.  **Reconstructed Mesh (Right)**: The final 3D mesh generated from the points and the median predicted alpha.

---

## Prerequisites

Before you begin, ensure you have the following installed:

-   Python 3.8+
-   NVIDIA GPU with CUDA (for GPU acceleration)
-   A trained model checkpoint file (e.g., `pointnet_alpha_v6_epoch_60.pth`).
-   The [ShapeNetCore.v2](https://www.shapenet.org/) dataset, extracted to a known directory.

## Installation

1.  **Clone the repository or download the script.**

2.  **Install the required Python libraries.** It is highly recommended to use a virtual environment.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install core dependencies
    pip install torch torchvision torchaudio open3d trimesh matplotlib
    ```

3.  **Install PyTorch Geometric and PyTorch3D.** These libraries often have specific installation requirements based on your PyTorch and CUDA versions. Please follow their official instructions for a reliable setup.

    -   **PyTorch Geometric:** [Official Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
    -   **PyTorch3D:** [Official Installation Guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

---

## Usage

The script is controlled via command-line arguments. You must provide the path to your trained model and the ShapeNet dataset.

### Basic Command Structure

```bash
python visualization.py --model_path <PATH_TO_MODEL.pth> --shapenet_path <PATH_TO_SHAPENET_ROOT> [OPTIONS]
```

### Command-Line Arguments

| Argument           | Description                                                                                                                              | Required | Default Value    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------- |
| `--model_path`     | The file path to your trained model checkpoint (`.pth` file).                                                                            | **Yes**  | `None`           |
| `--shapenet_path`  | The path to the root directory of your `ShapeNetCore.v2` dataset.                                                                        | **Yes**  | `None`           |
| `--num_points`     | The number of points to sample from the mesh. **This should match the number of points used during training.**                             | No       | `2048`           |
| `--sample_index`   | The specific index of the model to visualize from the dataset's file list. If omitted, a random model will be chosen for each execution. | No       | `None` (Random)  |

---

## Examples

Here are some precise examples of how to run the script.

### Example 1: Visualize a Random Model

This is the most common use case. It will pick a random object from the dataset every time you run it.

```bash
python visualization.py --model_path ./pointnet_alpha_v6_epoch_60.pth --shapenet_path ./ShapeNetCore.v2
```

### Example 2: Visualize a Specific Model by Index

If you want to consistently view or debug a specific object, use the `--sample_index` argument. The index corresponds to the model's position in the list of all found `.ply` files.

-   To view the **first** model in the dataset (index `0`):

    ```bash
    python visualization.py --model_path ./pointnet_alpha_v6_epoch_60.pth --shapenet_path ./ShapeNetCore.v2 --sample_index 0
    ```

-   To view the **50th** model in the dataset (index `49`):

    ```bash
    python visualization.py --model_path ./pointnet_alpha_v6_epoch_60.pth --shapenet_path ./ShapeNetCore.v2 --sample_index 49
    ```

### Example 3: Using a Different Number of Sampled Points

If your model was trained with a different point density (e.g., 4096 points), you must specify it.

```bash
python visualization.py \
    --model_path ./my_4096_pt_model.pth \
    --shapenet_path ./ShapeNetCore.v2 \
    --num_points 4096 \
    --sample_index 101
```

## Troubleshooting

-   **`Error(s) in loading state_dict for ...`**: This is the most common error. It means the model architecture defined in `visualization.py` (e.g., `PyG_PointNet2_Alpha_Predictor`) does not match the architecture that was used to train and save the `.pth` file you are loading. Ensure you are instantiating the correct model class in the `visualize_inference` function.
-   **`FATAL ERROR: PyG not installed correctly` / `PyTorch3D not found`**: These errors mean the specialized libraries were not installed correctly. Please refer to their official installation guides linked above, as a simple `pip install` may not be sufficient.
