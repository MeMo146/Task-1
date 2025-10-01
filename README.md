# CT Scan Organ Segmentation Viewer

A modern Python desktop application for **visualizing and segmenting organs in CT scans**. This tool lets you load NIfTI CT files, run organ segmentation with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), interactively view 2D overlays, and explore 3D models using PyVista.

---

## Features

- **Load NIfTI CT files** (`.nii`, `.nii.gz`)
- **Segment specific organs:** lungs & airway, spinal cord (vertebrae), rib cage (individual ribs & sternum)
- **2D Viewer:** View CT slices side-by-side with segmentation overlays
- **Custom 2D Mask Colors:** Change overlay colors for each organ interactively
- **3D Model Viewer:** Visualize and interact with segmented organs in 3D (PyVista)
- **User-friendly GUI:** Built with Tkinter

---

## Screenshots

![spinal-cord](https://github.com/user-attachments/assets/7f3e0ece-0775-486a-a5a7-d211d414c41d)
![ribcage](https://github.com/user-attachments/assets/ade32a7e-4a31-4408-a5cb-4d749e4d36e3)
![lungs](https://github.com/user-attachments/assets/ef2f78bb-74eb-433c-8611-83391e7525b0)


---

## Requirements

- **Python** 3.8+
- **Operating System:** Windows or Linux (Mac not fully supported due to PyVistaQt)
- **Recommended:** GPU for faster segmentation

### Python Packages

- `totalsegmentator`
- `nibabel`
- `numpy`
- `matplotlib`
- `scikit-image`
- `pyvista`
- `pyvistaqt`
- `tkinter` (built-in)

Install all requirements with:
```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/MeMo146/Task-1.git
cd Task-1
pip install -r requirements.txt
```

---

## Usage

```bash
python ct_segmentation_viewer.py
```
**Workflow:**
1. Select a CT file (`.nii`/`.nii.gz`).
2. Select an output folder.
3. Choose the organ to segment (lungs_airway, spinal_cord, rib_cage).
4. Click **Run Organ Segmentation**.
5. Explore results in 2D and 3D.
6. Change mask overlay colors as desired.

---

## Output

- **Segmented mask files** (NIfTI) are saved in the output folder (and possibly `output_folder/segmentations/`)
- **3D models** are visualized but not automatically exported (add STL export if desired)
- **Overlay images** are not saved by default, but you can add this functionality

---

## Notes

- The app does **not** segment the liver or heart in this version.
- Mask overlays in 2D can be color-customized per organ.
- 3D colors are fixed for clarity.
- For best results, use high-quality CT scans with standard head/neck/chest/abdomen fields-of-view.

---
## Credits

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [PyVista](https://github.com/pyvista/pyvista)
- [nibabel](https://github.com/nipy/nibabel)
