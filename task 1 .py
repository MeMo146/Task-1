import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
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
        self.plotter = None
        self.mask_colors_2d = {}
        self.color_buttons = {}
        self.opacity_3d = 1.0  # Default fully opaque

        # --- Control panel on left ---
        control_frame = tk.Frame(root)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        tk.Button(control_frame, text="Select CT File", command=self.select_ct_file).pack(pady=5, fill="x")
        tk.Button(control_frame, text="Select Output Folder", command=self.select_output_folder).pack(pady=5, fill="x")

        tk.Label(control_frame, text="Choose Organ:").pack(pady=5)
        organ_options = ["lungs_airway", "spinal_cord", "rib_cage"]
        self.organ_var = tk.StringVar(value=organ_options[0])
        self.organ_dropdown = ttk.Combobox(control_frame, textvariable=self.organ_var, values=organ_options, state="readonly")
        self.organ_dropdown.pack(pady=5, fill="x")

        tk.Button(control_frame, text="Run Organ Segmentation", command=self.run_segmentation).pack(pady=10, fill="x")

        # Color control area (for after segmentation)
        self.color_control_frame = tk.LabelFrame(control_frame, text="Mask Colors (2D only)")
        self.color_control_frame.pack(fill="x", pady=10)

        # --- 3D Opacity slider ---
        opacity_frame = tk.LabelFrame(control_frame, text="3D Opacity")
        opacity_frame.pack(fill="x", pady=10)
        self.opacity_slider = tk.Scale(opacity_frame, from_=0, to=100, resolution=1, orient="horizontal",
                                       command=self.update_opacity_3d)
        self.opacity_slider.set(int(self.opacity_3d * 100))
        self.opacity_slider.pack(fill="x")

        # --- Right side: viewers ---
        right_frame = tk.Frame(root)
        right_frame.pack(side="right", fill="both", expand=True)

        self.fig, (self.ax_ct, self.ax_seg) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack()

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

            seg_dir = self.output_dir
            if os.path.isdir(os.path.join(self.output_dir, "segmentations")):
                seg_dir = os.path.join(self.output_dir, "segmentations")

            self.masks.clear()
            self.mask_data = None

            if organ == "lungs_airway":
                self.segment_lungs_airway(seg_dir)
            elif organ == "spinal_cord":
                self.segment_spinal_cord(seg_dir)
            elif organ == "rib_cage":
                self.segment_rib_cage(seg_dir)
            else:
                self.segment_single_organ(seg_dir, organ)

            if self.mask_data is None or not np.any(self.mask_data):
                messagebox.showerror("Error", f"No mask found for {organ}")
                return

            self.setup_mask_colors_2d()
            self.update_display()
            self.setup_color_controls()
            self.show_3d_models()

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed:\n{e}")

    def setup_mask_colors_2d(self):
        self.mask_colors_2d.clear()
        for idx, (mask_data, organ_name) in enumerate(self.masks):
            default_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            self.mask_colors_2d[organ_name] = default_colors[idx % len(default_colors)]

    def setup_color_controls(self):
        for btn in self.color_buttons.values():
            btn.destroy()
        self.color_buttons.clear()
        for organ_name in self.mask_colors_2d:
            color = self.mask_colors_2d[organ_name]
            btn = tk.Button(self.color_control_frame, text=f"Change {organ_name} color (2D only)", bg=color,
                            command=lambda o=organ_name: self.change_mask_color_2d(o))
            btn.pack(fill="x", pady=2)
            self.color_buttons[organ_name] = btn

    def change_mask_color_2d(self, organ_name):
        initial_color = self.mask_colors_2d.get(organ_name, "#ff0000")
        color_code = colorchooser.askcolor(title=f"Choose color for {organ_name} (2D)", initialcolor=initial_color)
        if color_code and color_code[1]:
            self.mask_colors_2d[organ_name] = color_code[1]
            self.color_buttons[organ_name].configure(bg=color_code[1])
            self.update_display()

    def update_opacity_3d(self, val):
        self.opacity_3d = float(val) / 100.0
        # If 3D models are visible, update their opacity
        if self.plotter is not None:
            self.show_3d_models()

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
        for keyword in lung_keywords:
            for file in os.listdir(seg_dir):
                if file.endswith(".nii.gz") and keyword in file:
                    lung_mask = nib.load(os.path.join(seg_dir, file)).get_fdata()
                    if combined_mask is None:
                        combined_mask = np.zeros_like(lung_mask)
                    combined_mask[lung_mask > 0] = 1
        for keyword in airway_keywords:
            for file in os.listdir(seg_dir):
                if file.endswith(".nii.gz") and keyword in file:
                    airway_mask = nib.load(os.path.join(seg_dir, file)).get_fdata()
                    if combined_mask is None:
                        combined_mask = np.zeros_like(airway_mask)
                    combined_mask[airway_mask > 0] = 2
        if combined_mask is not None:
            self.mask_data = combined_mask
            self.masks.append((combined_mask, "lungs_airway"))

    def segment_spinal_cord(self, seg_dir):
        vertebrae_keywords = [
            "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
            "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6", "vertebrae_T7",
            "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
            "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
            "sacrum",
            "spinal_canal", "spinal_cord"
        ]
        combined_mask = None
        label_id = 1
        for keyword in vertebrae_keywords:
            for file in os.listdir(seg_dir):
                if file.endswith(".nii.gz") and keyword in file:
                    mask = nib.load(os.path.join(seg_dir, file)).get_fdata()
                    if combined_mask is None:
                        combined_mask = np.zeros_like(mask)
                    combined_mask[mask > 0] = label_id
                    self.masks.append((mask, keyword))
                    label_id += 1
        if combined_mask is not None:
            self.mask_data = combined_mask

    def segment_rib_cage(self, seg_dir):
        rib_keywords = [f"rib_{i}" for i in range(1, 25)] + ["sternum"]
        combined_mask = None
        label_id = 1
        for keyword in rib_keywords:
            for file in os.listdir(seg_dir):
                if file.endswith(".nii.gz") and keyword in file:
                    mask = nib.load(os.path.join(seg_dir, file)).get_fdata()
                    if combined_mask is None:
                        combined_mask = np.zeros_like(mask)
                    combined_mask[mask > 0] = label_id
                    self.masks.append((mask, keyword))
                    label_id += 1
        if combined_mask is not None:
            self.mask_data = combined_mask

    def show_3d_models(self):
        if not self.masks:
            messagebox.showerror("Error", "No masks loaded to visualize.")
            return
        try:
            if self.plotter is not None:
                self.plotter.close()
            self.plotter = BackgroundPlotter(window_size=(1000, 800))
            default_colors = [
                "red", "green", "blue", "yellow", "purple", "cyan",
                "orange", "pink", "lime", "teal"
            ]
            for i, (mask_data, organ_name) in enumerate(self.masks):
                binary_mask = (mask_data > 0).astype(np.uint8)
                if not np.any(binary_mask):
                    print(f"⚠ Skipping {organ_name}: mask is empty.")
                    continue
                try:
                    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
                    faces = np.c_[np.full(len(faces), 3), faces].ravel()
                    mesh = pv.PolyData(verts, faces)
                    color = default_colors[i % len(default_colors)]
                    self.plotter.add_mesh(mesh, color=color, opacity=self.opacity_3d, name=organ_name)
                    self.plotter.add_text(organ_name, font_size=10, name=f"text_{organ_name}")
                except Exception as e:
                    print(f"❌ Could not create mesh for {organ_name}: {e}")
                    messagebox.showerror("Meshing Error", f"Could not create 3D model for {organ_name}:\n{e}")
            self.plotter.add_axes()
            self.plotter.show()
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
            organ = self.organ_var.get()
            if organ == "lungs_airway":
                for i, label_val in enumerate([1, 2]):
                    mask = (self.mask_data == label_val).astype(np.uint8)
                    if mask.shape == self.ct_data.shape:
                        organ_name = "lungs_airway"
                        color = self.mask_colors_2d.get(organ_name, ["#1f77b4", "#ff7f0e"][i])
                        from matplotlib.colors import ListedColormap
                        cmap = ListedColormap(["none", color])
                        self.ax_seg.imshow(mask[:, :, self.slice_index], cmap=cmap, alpha=0.5, vmin=0, vmax=1)
            else:
                for mask, organ_name in self.masks:
                    if mask.shape == self.ct_data.shape:
                        color = self.mask_colors_2d.get(organ_name, "#ff0000")
                        from matplotlib.colors import ListedColormap
                        cmap = ListedColormap(["none", color])
                        self.ax_seg.imshow((mask[:, :, self.slice_index] > 0).astype(np.uint8), cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        self.ax_seg.set_title("Segmentation Overlay (2D)")
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationGUI(root)
    root.mainloop()
