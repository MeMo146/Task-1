# CT Scan Organ Segmentation Viewer (Enhanced Version)

A Python desktop application for **visualizing and segmenting organs in CT scans** with advanced 3D opacity control. Load a NIfTI CT file, segment organs using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), explore 2D overlays, customize overlay colors, and interactively adjust 3D model opacity.

---

## Features

- **Easy-to-use GUI** built with Tkinter
- **Load CT scans** (`.nii`, `.nii.gz`)
- **Segment organs:** lungs & airway, spinal cord (vertebrae), rib cage
- **2D Viewer:** CT slice and segmentation overlay (side-by-side)
- **Custom overlay colors** for each organ (2D)
- **3D Model Viewer:** Interactive 3D visualization with PyVista
- **Adjustable 3D model opacity** with a slider for better visualization

---

## Screenshots

**Lungs & Airway Segmentation**
![Lungs/Airway Example](assets/images/image1.png)

**Spinal Cord & Vertebrae Segmentation**
![Spine Example](assets/images/image2.png)

**Rib Cage Segmentation**
![Rib Cage Example](assets/images/image3.png)

---

## Quickstart

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python ct_segmentation_viewer2.py
```

### 3. Workflow

1. **Select CT File**: Load a NIfTI format CT file.
2. **Select Output Folder**: Choose where segmentations will be saved.
3. **Choose Organ**: Pick an organ group to segment.
4. **Run Organ Segmentation**: Starts segmentation and loads results.
5. **View**: Explore overlays and interact with 3D renderings.
6. **Customize Colors (2D only)**: Change overlay colors for clarity.
7. **Adjust 3D Opacity**: Use the slider to make 3D models transparent or opaque.

---

## Requirements

- Python 3.8+
- Windows or Linux (PyVistaQt may not work on Mac)
- GPU recommended for faster segmentation

**Python packages:**

- totalsegmentator
- nibabel
- numpy
- matplotlib
- scikit-image
- pyvista
- pyvistaqt

---

## Demo

You can add a GIF or video demo here:

```markdown
![App Demo](assets/demo.gif)
```
Or, for a YouTube video:

```markdown
[![Watch the demo](assets/demo_thumbnail.png)](https://youtu.be/your_video_id)
```

---

## Notes

- Only lungs/airway, spinal cord/vertebrae, and rib cage are supported in this version.
- Segmented masks are saved as NIfTI files in the output directory.
- 3D models are visualized in-app; STL export is not included by default.
- Overlay color changes only affect the 2D viewer.
- 3D opacity slider updates the transparency of all rendered meshes in real time.

---

## Credits

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [PyVista](https://github.com/pyvista/pyvista)
- [nibabel](https://github.com/nipy/nibabel)

---

## License

This project is for educational and demonstration purposes.
