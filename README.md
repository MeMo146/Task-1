# Medical Segmentation Viewer

A Python Tkinter desktop application for medical image segmentation, visualization, and 3D modeling using TotalSegmentator.

## Features

- Load NIfTI (`.nii`, `.nii.gz`) CT images
- Segment multiple organs using TotalSegmentator
- View CT and segmentation mask side by side
- Interactive slice navigation
- 3D visualization of segmented organs

## Requirements

- Python 3.8+
- Linux or Windows OS recommended (PyVista/GUI may have limitations on Mac)

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/MeMo146/Task-1.git
   cd Task-1
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the app:
   ```
   python segmentation_app.py
   ```
2. In the GUI:
   - Click "Browse" to select a CT NIfTI file.
   - Choose an output directory.
   - Select organ for segmentation.
   - Click "Run Segmentation".
   - Navigate slices using the slider.
   - View 3D model in a popup.

**Note:** You may need a GPU for reasonable segmentation speed.

## Dependencies

See `requirements.txt` for all packages.

- `totalsegmentator`
- `nibabel`
- `numpy`
- `matplotlib`
- `scikit-image`
- `pyvista`
- `tkinter` (standard with Python)

## License

Specify your license here (e.g., MIT, Apache-2.0).
