import tkinter as tk
from tkinter import filedialog, messagebox
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

# MONAI imports for U-Net and basic transforms
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord

# Import for GPU performance optimization
from torch.cuda.amp import autocast

# Suppress some MONAI warnings
import warnings
warnings.filterwarnings("ignore")


class SegmentationApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Medical Image Segmentation with U-Net")

        self.image_data = None
        self.segmentation = None
        self.slice_idx = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print("Enabling cuDNN benchmark mode for performance.")
            torch.backends.cudnn.benchmark = True

        # --- Initialize a Generic, OFFLINE U-Net Model ---
        # NOTE: This model is UNTRAINED. Its output will be random noise.
        # This is necessary to avoid the network download errors.
        print("Initializing a generic U-Net model (no download required)...")
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,  # background and foreground
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.model.eval()
        print("U-Net model initialized.")

        # Define a simple, offline transform pipeline
        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear", align_corners=False),
            ToTensord(keys=["image"]),
        ])

        self.build_ui()

    def build_ui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_file)
        load_btn.pack(side=tk.LEFT, padx=5)

        img_frame = tk.Frame(self.root)
        img_frame.pack()

        self.fig1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=img_frame)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10)

        self.fig2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=img_frame)
        self.canvas2.get_tk_widget().pack(side=tk.RIGHT, padx=10)

        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)

        prev_btn = tk.Button(nav_frame, text="◀ Prev", command=self.prev_slice)
        prev_btn.pack(side=tk.LEFT, padx=5)

        self.slice_label = tk.Label(nav_frame, text="Slice: 0")
        self.slice_label.pack(side=tk.LEFT, padx=10)

        next_btn = tk.Button(nav_frame, text="Next ▶", command=self.next_slice)
        next_btn.pack(side=tk.LEFT, padx=5)

        self.slice_slider = tk.Scale(
            nav_frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            command=self._slider_changed,
            length=400
        )
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        self.root.bind("<Left>", lambda event: self.prev_slice())
        self.root.bind("<Right>", lambda event: self.next_slice())
        self.root.bind("<MouseWheel>", self._on_mouse_scroll)
        self.root.bind("<Button-4>", self._on_mouse_scroll)
        self.root.bind("<Button-5>", self._on_mouse_scroll)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("NIfTI files", "*.nii *.nii.gz")]
        )
        if not file_path:
            return

        self.root.config(cursor="watch")
        self.root.update()

        try:
            # 1. Apply the simple preprocessing transforms
            data_dict = self.transforms({"image": file_path})
            image_tensor = data_dict["image"]

            # 2. Run inference on the U-Net
            with torch.no_grad():
                val_inputs = image_tensor.unsqueeze(0).to(self.device)
                if self.device.type == 'cuda':
                    with autocast():
                        output_tensor = self.model(val_inputs)
                else:
                    output_tensor = self.model(val_inputs)

            # 3. Post-process the output
            segmentation_map = torch.argmax(output_tensor, dim=1).squeeze(0).cpu().numpy()

            # The image for display is the resized one fed to the model
            self.image_data = image_tensor.squeeze(0).cpu().numpy()
            self.segmentation = segmentation_map.astype(np.uint8)

            # 4. Update the GUI
            self.slice_idx = self.image_data.shape[0] // 2 # Slice along depth (axis 0)
            self.slice_slider.config(to=self.image_data.shape[0] - 1)
            self.update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.config(cursor="")

    def update_display(self):
        if self.image_data is None:
            return

        self.ax1.clear()
        self.ax2.clear()

        # Data from transforms is (D, H, W). We show the H, W plane.
        img_slice = self.image_data[self.slice_idx, :, :]
        seg_slice = self.segmentation[self.slice_idx, :, :]

        self.ax1.imshow(img_slice, cmap="gray")
        self.ax1.set_title("Original Image (Resized)")
        self.ax1.axis("off")

        self.ax2.imshow(img_slice, cmap="gray")
        self.ax2.imshow(seg_slice, alpha=0.5, cmap="Reds", vmin=0, vmax=1)
        self.ax2.set_title("Segmented Image (Untrained)")
        self.ax2.axis("off")

        self.canvas1.draw()
        self.canvas2.draw()

        self.slice_label.config(text=f"Slice: {self.slice_idx}")
        self.slice_slider.set(self.slice_idx)

    def prev_slice(self):
        if self.image_data is None: return
        self.slice_idx = max(0, self.slice_idx - 1)
        self.update_display()

    def next_slice(self):
        if self.image_data is None: return
        self.slice_idx = min(self.image_data.shape[0] - 1, self.slice_idx + 1)
        self.update_display()

    def _slider_changed(self, val):
        self.slice_idx = int(float(val))
        self.update_display()

    def _on_mouse_scroll(self, event):
        if self.image_data is None: return
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.prev_slice()
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.next_slice()

if _name_ == "_main_":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
