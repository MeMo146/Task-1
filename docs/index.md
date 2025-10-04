# CT Scan Organ Segmentation Viewer

## Overview

**CT Scan Organ Segmentation Viewer** is a graphical user interface (GUI) application for viewing and segmenting organs in CT scan volumes. It allows users to:
- Select a CT scan (in NIfTI format)
- Run automated organ segmentation using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- Visualize the original CT scan and the segmented organs in 2D (slice-by-slice) and 3D
- Change overlay mask colors for better visualization
- Adjust the opacity of 3D models interactively

## Features

- **File Selection:** Easily select CT scan files and output folders.
- **Multiple Organ Support:** Segment lungs+airway, spinal cord+vertebrae, or rib cage (other organs can be added).
- **2D Visualization:** View CT slices with segmentation overlays.
- **3D Visualization:** Render segmented organs as interactive 3D models.
- **Customizable Colors:** Change mask overlay colors for 2D slices.
- **3D Opacity Control:** Adjust transparency of 3D models in real-time.

## Requirements

- Python 3.7+
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- nibabel
- numpy
- matplotlib
- scikit-image
- pyvista, pyvistaqt
- tkinter (included with most Python installations)

Install dependencies with:
```bash
pip install totalsegmentator nibabel numpy matplotlib scikit-image pyvista pyvistaqt
```

## Usage

1. **Run the Application**
   ```bash
   python ct_segmentation_gui.py
   ```

2. **Steps in the GUI**
   - Click **Select CT File** to choose your NIfTI CT scan (`.nii` or `.nii.gz`).
   - Click **Select Output Folder** to specify where segmentation results will be saved.
   - Choose an organ group from the dropdown (**lungs_airway**, **spinal_cord**, **rib_cage**).
   - Click **Run Organ Segmentation**.
   - Browse CT slices and see overlays.
   - Adjust mask colors (for 2D overlays) and 3D opacity as desired.
   - Use **Show 3D Models** button (automatically appears after segmentation) to see the segmentation in 3D.

## File Structure

- `main.py` — Main application script.
- `docs/` — Documentation folder (you are here!)

## Notes

- **Input:** CT scans must be in NIfTI (`.nii` or `.nii.gz`) format.
- **Segmentation:** Uses TotalSegmentator, which must be installed and properly configured.
- **Output:** Segmentation results are saved in the selected output directory.

## Troubleshooting

- If segmentation fails, check that your CT file is valid and that TotalSegmentator is installed.
- For 3D rendering, ensure your system supports OpenGL and that PyVista/PyVistaQt is installed.

## License

This project is open source. See [LICENSE](../LICENSE) for details.

---

**Author:** [youssefh4](https://github.com/youssefh4)  
**Contributions and feedback welcome!**
