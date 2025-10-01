# CT Scan Organ Segmentation Viewer

A modern Python desktop app for visualizing and segmenting organs in CT scans, with 2D overlays and 3D model export using TotalSegmentator.

## Features

- Load NIfTI CT (`.nii`/`.nii.gz`) files
- Segment organs: heart, liver, lungs + airway, spinal cord, rib cage
- 2D slice viewer with interactive segmentation overlays
- 3D model viewer and STL export for each organ and combined model (PyVista)
- GUI with easy workflow (Tkinter)

## Screenshots

![WhatsApp Image 2025-10-01 at 23 45 00_f4bfecfe](https://github.com/user-attachments/assets/a758efa1-7bb7-4547-a300-52caf713b667)



![WhatsApp Image 2025-10-01 at 23 33 39_79a64e2d](https://github.com/user-attachments/assets/3954adad-2c5a-43b3-a2b0-7a8e6e76180e)



![WhatsApp Image 2025-10-02 at 00 49 03_842a7b69](https://github.com/user-attachments/assets/b1f7660a-0544-457b-9ee2-a8e4b9887ec8)


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
