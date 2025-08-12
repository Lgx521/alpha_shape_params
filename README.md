# Alpha shape params prediction
Utilizing Reinforcement learning  
**NOTE**: This is still a immuture repo

---

Excellent point. Using Conda is a very common and robust workflow for managing complex deep learning dependencies.

Here is the updated `README.md` file with instructions tailored for a Conda environment.

---

# Adaptive Alpha Shape Inference and Visualization

This script provides a powerful tool to visualize the output of a trained PointNet++ model designed to predict per-point alpha values for 3D surface reconstruction. It loads a point cloud from the ShapeNet dataset, runs inference to get the alpha values, and displays a side-by-side comparison of the input points, an alpha value heatmap, and the final reconstructed mesh using Open3D's alpha shape algorithm.


*(Image shows a sample output: Heatmap (left), Original Points (center), Reconstructed Mesh (right))*

## Key Features

-   **Model Inference**: Loads a pre-trained PyTorch model (`.pth` file) for inference.
-   **Per-Point Alpha Prediction**: Runs a point cloud sample through the network to predict an optimal alpha value for each point.
-   **CGAL-Free Reconstruction**: Uses the built-in `open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape` for mesh creation, removing the need for external dependencies like CGAL.
-   **Rich Visualization**: Provides an interactive Open3D window with three views for easy comparison.

---

## Prerequisites

Before you begin, ensure you have the following installed:

-   **Anaconda or Miniconda**: This is required to manage the environment and dependencies.
-   **NVIDIA GPU with CUDA Drivers**: Required for GPU acceleration. The script will fall back to CPU if a GPU is not available, but it will be much slower.
-   A trained model checkpoint file (e.g., `pointnet_alpha_v6_epoch_60.pth`).
-   The [ShapeNetCore.v2](https://www.shapenet.org/) dataset, extracted to a known directory.

---

## Installation with Conda

This guide assumes you are using Conda to manage your environment.

1.  **Clone the repository or download the script.**

2.  **Create and Activate a Conda Environment:**
    Open your terminal and create a new environment for this project. We recommend Python 3.8 or 3.9 as they have wide support across the required libraries.

    ```bash
    # Create a new conda environment named "alpha_vis" with Python 3.8
    conda create --name alpha_vis python=3.8

    # Activate the environment
    conda activate alpha_vis
    ```

3.  **Install PyTorch with CUDA:**
    It is crucial to install PyTorch using the official command from their website to ensure CUDA compatibility. **Do not use `pip install torch`**.

    -   Navigate to the [PyTorch "Get Started" Page](https://pytorch.org/get-started/locally/).
    -   Select your system configuration (e.g., Stable, Linux, Conda, Python, your CUDA version).
    -   Copy and run the generated command. For example, for CUDA 11.8, the command is:

    ```bash
    # EXAMPLE COMMAND - VERIFY ON PYTORCH WEBSITE FOR YOUR SYSTEM!
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4.  **Install PyTorch Geometric and PyTorch3D:**
    These libraries also have specific installation procedures. Follow them carefully.

    ```bash
    # Install PyTorch Geometric (this command checks for your PyTorch/CUDA version)
    conda install pyg -c pyg

    # Install PyTorch3D dependencies first, then the library itself
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
    ```

5.  **Install Remaining Libraries:**
    Finally, install the other required packages using pip within your active conda environment.

    ```bash
    pip install open3d trimesh matplotlib
    ```

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

Make sure your `alpha_vis` conda environment is active before running these commands.

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

If your model was trained with a different point density (e.g., 4096 points), you must specify it to ensure correct model input.

```bash
python visualization.py \
    --model_path ./my_4096_pt_model.pth \
    --shapenet_path ./ShapeNetCore.v2 \
    --num_points 4096 \
    --sample_index 101
```

## Troubleshooting

-   **`Error(s) in loading state_dict for ...`**: This is the most common error. It means the model architecture defined in `visualization.py` does not match the architecture that was used to train and save the `.pth` file you are loading. Ensure you are instantiating the correct model class in the `visualize_inference` function.
-   **Module Not Found / Import Errors**: If you get errors like `No module named 'torch_geometric'`, it means the installation failed or you forgot to activate your conda environment (`conda activate alpha_vis`). Re-visit the installation steps carefully.
-   **CUDA Errors**: If you encounter CUDA-related issues, it almost always means there's a mismatch between your NVIDIA driver version, your CUDA toolkit version, and the version of PyTorch you installed. The best solution is to create a fresh conda environment and carefully follow the official PyTorch installation instructions for your specific system.
