# Phantom Generation

This repository contains Python code to generate several synthetic phantoms that can be used for testing or demonstrating image processing methods from the implementation paper.

The generator is implemented in `phantom_generation.py` and produces four phantom families:

1. **Head phantoms**
   - `make_shepp_logan()` creates a resized Shepp–Logan phantom.
   - `make_head_phantom()` creates a custom head-like phantom with soft tissue, skull ring, bright inner structures, blur, and noise.
   - `make_dented_head_phantom()` creates a second custom head phantom with dents, asymmetry, and shading.

2. **Rat femur phantoms**
   - `make_rat_femur_v1()`
   - `make_rat_femur_v2()`
   - `make_rat_femur_v3()`

   These functions build bone-like structures from ellipses, polygons, and disks. They simulate outer bone shape, inner region, cortical ring, and trabecular patterns using binary masks.

3. **Metal particles phantoms**
   - `make_metal_particles_v1()`
   - `make_metal_particles_v2()`
   - `make_metal_particles_v3()`

   These functions generate porous material-like phantoms with walls, pores, and bright particle inclusions. Morphological operations such as dilation and erosion are used to create pore boundaries and structure.

4. **Dental fillings phantoms**
   - `generate_dental_phantom()`

   This function generates a head/jaw phantom with teeth and optional fillings. It uses randomized parameters with a seed for reproducibility, then applies shading and Gaussian blur.

## How the code works

The phantoms are constructed from simple geometric primitives and image operations:

- **Ellipses, polygons, and disks** are used to define object regions.
- **Binary masks** represent different anatomical or material components.
- **Intensity assignment** is used to distinguish tissue, bone, pores, particles, teeth, and fillings.
- **Gaussian smoothing** is applied to make the phantoms look less artificial.
- **Noise and shading** are added in some phantoms to create more realistic intensity variation.
- **Seeds** are used in the stochastic generators so results can be reproduced.

At the bottom of the script, the `if __name__ == "__main__":` block generates example outputs for all four families and displays them with Matplotlib.

## File overview

- `phantom_generation.py` – main phantom generator code
- `requirements.txt` – Python dependencies needed to run the script
- `README.md` – documentation and usage instructions

## Installation

It is recommended to use a virtual environment.

### 1. Create and activate a virtual environment

On Linux/macOS: 

bash
```
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the dependencies
pip install -r requirements.txt

## Requirements

The code depends on the following Python packages:

- numpy
- matplotlib
- scipy
- scikit-image

## How to run

Run the script with:

```
python phantom_generation.py

```

This will generate and display example phantoms for all four phantom families:

- 3 head phantoms
- 3 rat femur phantoms
- 3 metal particle phantoms
- 3 dental filling phantoms
