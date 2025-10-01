# CT Scan Organ Segmentation Viewer

A modern Python desktop app for visualizing and segmenting organs in CT scans, with 2D overlays and 3D model export using TotalSegmentator.

## Features

- Load NIfTI CT (`.nii`/`.nii.gz`) files
- Segment organs: heart, liver, lungs + airway, spinal cord, rib cage
- 2D slice viewer with interactive segmentation overlays
- 3D model viewer and STL export for each organ and combined model (PyVista)
- GUI with easy workflow (Tkinter)

## Screenshots

![Rib Cage Example](assests/images/rib_cage.png)

![Spine Example](assests/images/spine.png)

![Lungs/Airway Example](assests/images/lungs.png)

## Requirements

- Python 3.8+
- Windows or Linux (PyVistaQt may not work on Mac)
- GPU recommended for faster segmentation

## Installation

```bash
git clone https://github.com/MeMo146/Task-1.git
cd Task-1
pip install -r requirements.txt
```

## Usage

```bash
python segmentation_viewer.py
```

- Select a CT file and output folder.
- Choose organ and run segmentation.
- View before/after overlays and 3D models.
- STL files for each organ and all organs combined are saved in the output directory.

## Dependencies

- totalsegmentator
- nibabel
- numpy
- matplotlib
- scikit-image
- pyvista
- pyvistaqt
- tkinter (built-in)
