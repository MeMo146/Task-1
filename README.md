# CT Scan Medical organ segmentation 

A Python desktop application for **visualizing and segmenting organs in CT scans** with advanced 3D opacity control. Load a NIfTI CT file, segment organs using [TotalSegmentator]

---

## Features

- **Easy-to-use GUI** built with Tkinter
- **Load CT scans** (`.nii`, `.nii.gz`)
- **Segment organs:** lungs & airway, spinal cord (vertebrae), rib cage
- **2D Viewer:** CT slice and segmentation overlay (side-by-side)
- **Custom overlay colors** for each organ (2D)
- **3D Model Viewer:** Interactive 3D visualization with PyVista
- **Adjustable 3D model opacity** with a slider for better visualization
- **Documentation** : For detailed usage and features, see the [Full Documentation](docs/index.md).

---

## Visual Overview


<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f1a7d3ba-dbf9-4c5a-8bb6-380c9b5d6374" width="300"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/61ec8d28-5a87-4ba3-8943-d2a2a251a05f" width="300"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/70f1846b-d8fa-4ad7-9a47-17a9e581abf9" width="300"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/21e38c39-465e-4138-87b5-75070d39ae16" width="300"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/dd04372e-cc68-48af-8db7-adce472c3763" width="300"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/71282c6f-7dce-4a04-b9dd-2f311efaf1b5" width="300"/>
    </td>
  </tr>
</table>
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
## Contributors ✨

Thanks goes to these people ([⭐](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/youssefh4"><img src="https://avatars.githubusercontent.com/u/211988432?v=4?s=100" width="100px;" alt="youssef hisham "/><br /><sub><b>youssef hisham </b></sub></a><br /><a href="#code-youssefh4" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/YoussefELRoby0"><img src="https://avatars.githubusercontent.com/u/236052660?v=4?s=100" width="100px;" alt="YoussefELRoby0"/><br /><sub><b>YoussefELRoby0</b></sub></a><br /><a href="#code-YoussefELRoby0" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/habibaabdelmeneem01-byte"><img src="https://avatars.githubusercontent.com/u/235373412?v=4?s=100" width="100px;" alt="habibaabdelmeneem01-byte"/><br /><sub><b>habibaabdelmeneem01-byte</b></sub></a><br /><a href="#data-habibaabdelmeneem01-byte" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Yassen12345678"><img src="https://avatars.githubusercontent.com/u/108278577?v=4?s=100" width="100px;" alt="Yassen12345678"/><br /><sub><b>Yassen12345678</b></sub></a><br /><a href="#data-Yassen12345678" title="Data">🔣</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->



---
## License

This project is for educational and demonstration purposes.
