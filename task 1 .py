import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from skimage import measure
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class SegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Scan Organ Segmentation Viewer")

        # --- Data holders ---
        self.input_ct_path = None
        self.output_dir = None
        self.ct_data = None
        self.masks = []  # list of (mask_data, organ_name)
        self.slice_index = 0
        self.mask_data = None

        # --- Control panel on left ---
        control_frame = tk.Frame(root)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        tk.Button(control_frame, text="Select CT File", command=self.select_ct_file).pack(pady=5, fill="x")
        tk.Button(control_frame, text="Select Output Folder", command=self.select_output_folder).pack(pady=5, fill="x")

        tk.Label(control_frame, text="Choose Organ:").pack(pady=5)
        self.organ_var = tk.StringVar(value="heart")
        organ_options = ["heart", "liver", "lungs_airway", "spinalcord"]
        self.organ_dropdown = ttk.Combobox(control_frame, textvariable=self.organ_var, values=organ_options, state="readonly")
        self.organ_dropdown.pack(pady=5, fill="x")

        tk.Button(control_frame, text="Run Organ Segmentation", command=self.run_segmentation).pack(pady=10, fill="x")

        # --- Right side: viewers ---
        right_frame = tk.Frame(root)
        right_frame.pack(side="right", fill="both", expand=True)

        # Matplotlib figures (CT + segmentation)
        self.fig, (self.ax_ct, self.ax_seg) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack()

        # Slider
        self.slider = tk.Scale(right_frame, from_=0, to=0, orient="horizontal", command=self.update_slice)
        self.slider.pack(fill="x")

    def select_ct_file(self):
        self.input_ct_path = filedialog.askopenfilename(
            title="Select CT scan file",
            filetypes=[("NIfTI files", ".nii *.nii.gz"), ("All files", ".*")]
        )
        if self.input_ct_path:
            ct_img = nib.load(self.input_ct_path)
            self.ct_data = ct_img.get_fdata()
            self.slice_index = self.ct_data.shape[2] // 2
            self.slider.config(to=self.ct_data.shape[2] - 1)
            self.update_display()

    def select_output_folder(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")

    def run_segmentation(self):
        if not self.input_ct_path or not self.output_dir:
            messagebox.showerror("Error", "Select input CT and output folder first!")
            return

        organ = self.organ_var.get()
        try:
            messagebox.showinfo("Segmentation", f"Running segmentation for {organ}...")
            totalsegmentator(self.input_ct_path, self.output_dir, task="total")

            # Segmentation results folder
            seg_dir = self.output_dir
            if os.path.isdir(os.path.join(self.output_dir, "segmentations")):
                seg_dir = os.path.join(self.output_dir, "segmentations")

            self.masks.clear()
            self.mask_data = None

            if organ == "lungs_airway":
                self.segment_lungs_airway(seg_dir)
            elif organ == "spinalcord":
                self.segment_spinalcord(seg_dir)
            else:
                self.segment_single_organ(seg_dir, organ)

            if self.mask_data is None or not np.any(self.mask_data):
                messagebox.showerror("Error", f"No mask found for {organ}")
                return

            self.update_display()
            self.show_3d_models()

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed:\n{e}")

    def segment_single_organ(self, seg_dir, organ):
        mask_path = None
        for file in os.listdir(seg_dir):
            if file.endswith(".nii.gz") and organ in file:
                mask_path = os.path.join(seg_dir, file)
                break
        if mask_path:
            mask_img = nib.load(mask_path)
            self.mask_data = mask_img.get_fdata()
            self.masks.append((self.mask_data, organ))

    def segment_lungs_airway(self, seg_dir):
        lung_keywords = ["lung_upper_lobe_left", "lung_lower_lobe_left",
                         "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
        airway_keywords = ["trachea", "bronchus"]

        combined_mask = None

        # Lungs
        for keyword in lung_keywords:
            for file in os.listdir(seg_dir):
                if file.endswith(".nii.gz") and keyword in file:
                    lung_mask = nib.load(os.path.join(seg_dir, file)).get_fdata()
                    if combined_mask is None:
                        combined_mask = np.zeros_like(lung_mask)
                    combined_mask[lung_mask > 0] = 1
                    self.masks.append((lung_mask, keyword))

        # Airway
        for keyword in airway_keywords:
            for file in os.listdir(seg_dir):
                if file.endswith(".nii.gz") and keyword in file:
                    airway_mask = nib.load(os.path.join(seg_dir, file)).get_fdata()
                    if combined_mask is None:
                        combined_mask = np.zeros_like(airway_mask)
                    combined_mask[airway_mask > 0] = 2
                    self.masks.append((airway_mask, keyword))

        if combined_mask is not None:
            self.mask_data = combined_mask

    def segment_spinalcord(self, seg_dir):
        spinal_keywords = ["vertebra", "spinalcord", "spinal_cord", "cord"]
        combined_mask = None

        for file in os.listdir(seg_dir):
            if file.endswith(".nii.gz") and any(k in file for k in spinal_keywords):
                mask_data = nib.load(os.path.join(seg_dir, file)).get_fdata()
                name = file.replace(".nii.gz", "")
                self.masks.append((mask_data, name))

                if combined_mask is None:
                    combined_mask = np.zeros_like(mask_data)
                combined_mask[mask_data > 0] = 1

        if combined_mask is not None:
            self.mask_data = combined_mask

    def show_3d_models(self):
        """Convert masks to meshes and display in PyVista BackgroundPlotter, also save STLs."""
        if not self.masks:
            messagebox.showerror("Error", "No masks loaded to visualize.")
            return

        try:
            plotter = BackgroundPlotter(window_size=(1000, 800))
            colors = ["red", "green", "blue", "yellow", "purple", "cyan", "orange", "pink"]

            combined_meshes = []

            for i, (mask_data, organ_name) in enumerate(self.masks):
                binary_mask = (mask_data > 0).astype(np.uint8)
                if not np.any(binary_mask):
                    continue

                try:
                    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
                    faces = np.c_[np.full(len(faces), 3), faces].ravel()
                    mesh = pv.PolyData(verts, faces)

                    # Add to 3D viewer
                    plotter.add_mesh(mesh, color=colors[i % len(colors)], opacity=0.7, name=organ_name)

                    # Save STL
                    stl_path = os.path.join(self.output_dir, f"{organ_name}.stl")
                    mesh.save(stl_path)
                    print(f"✅ Saved {stl_path}")

                    combined_meshes.append(mesh)

                except Exception as e:
                    print(f"❌ Could not create mesh for {organ_name}: {e}")

            # Save combined STL
            if combined_meshes:
                combined = combined_meshes[0]
                for m in combined_meshes[1:]:
                    combined = combined.merge(m)
                combined_path = os.path.join(self.output_dir, "combined_model.stl")
                combined.save(combined_path)
                print(f"✅ Saved combined model: {combined_path}")

            plotter.add_axes()
            plotter.show()

        except Exception as e:
            messagebox.showerror("3D Viewer Error", str(e))

    def update_slice(self, val):
        self.slice_index = int(val)
        self.update_display()

    def update_display(self):
        self.ax_ct.clear()
        self.ax_seg.clear()

        if self.ct_data is not None:
            self.ax_ct.imshow(self.ct_data[:, :, self.slice_index], cmap="gray")
            self.ax_ct.set_title("CT Scan")

        if self.ct_data is not None:
            self.ax_seg.imshow(self.ct_data[:, :, self.slice_index], cmap="gray")

        if self.mask_data is not None:
            if self.organ_var.get() == "lungs_airway":
                colors = ["Blues", "Oranges"]
                for i, label_val in enumerate([1, 2]):
                    mask = (self.mask_data == label_val).astype(np.uint8)
                    if mask.shape == self.ct_data.shape:
                        self.ax_seg.imshow(mask[:, :, self.slice_index], cmap=colors[i], alpha=0.5)
            else:
                if self.mask_data.shape == self.ct_data.shape:
                    self.ax_seg.imshow(self.mask_data[:, :, self.slice_index], cmap="Reds", alpha=0.5)

        self.ax_seg.set_title("Segmentation Overlay")
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationGUI(root)
    root.mainloop()
