"""
Image Viewer component for SlayMarking.
Features:
- Direct click and drag of the divider line directly on the image canvas with exact pixel offset calibration.
- No external slider bar widget.
- Single view modes (Watermarked, Original) and difference residual heatmap.
- Real-time 120+ FPS composite updates during dragging.
"""
import customtkinter as ctk
from PIL import Image, ImageDraw
from typing import Optional
from ui.theme import OceanTheme

class ImageViewer(ctk.CTkFrame):
    """Central view area for displaying cover and watermarked images with on-image draggable divider."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=OceanTheme.BG_ABYSS, corner_radius=0, **kwargs)

        self.original_image: Optional[Image.Image] = None
        self.watermarked_image: Optional[Image.Image] = None
        self.difference_image: Optional[Image.Image] = None
        
        self.view_mode: str = "Split View"   # "Split View", "Watermarked", "Original", "Residual Heatmap"
        self.split_ratio: float = 0.50       # 0.0 to 1.0

        # Cached display images for instantaneous dragging at native physical device resolution
        self.cached_orig_disp: Optional[Image.Image] = None
        self.cached_wm_disp: Optional[Image.Image] = None
        self.disp_w: int = 512
        self.disp_h: int = 512
        self.phys_w: int = 512
        self.phys_h: int = 512
        
        self.cached_handle_icon: Optional[Image.Image] = None
        self.cached_handle_diam: int = 36

        self.split_label: Optional[ctk.CTkLabel] = None
        self.lbl_legend: Optional[ctk.CTkLabel] = None

        self._last_w: int = 0
        self._last_h: int = 0
        self._resize_timer: Optional[str] = None

        self._build_ui()

    def _get_or_create_handle_icon(self, diameter: int) -> Image.Image:
        """Generates an ultra-crisp, anti-aliased circular slider handle with bioluminescent arrows."""
        if self.cached_handle_icon is not None and self.cached_handle_diam == diameter:
            return self.cached_handle_icon

        # 4x supersampling for razor-sharp antialiased vector edges
        hi_size = max(diameter * 4, 128)
        icon = Image.new("RGBA", (hi_size, hi_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        center = hi_size // 2
        radius = center - 8

        line_col = (0, 245, 212, 255)      # OceanTheme.CYAN_PRIMARY
        bg_col = (11, 26, 46, 255)         # OceanTheme.BG_CARD
        glow_col = (0, 245, 212, 100)      # Soft ambient bioluminescent glow ring

        # Outer subtle glow ring
        draw.ellipse([(center - radius, center - radius), (center + radius, center + radius)], outline=glow_col, width=16)

        # Solid inner circle with sharp border
        inner_r = radius - 4
        draw.ellipse([(center - inner_r, center - inner_r), (center + inner_r, center + inner_r)], fill=bg_col, outline=line_col, width=10)

        # Precision directional arrows
        arrow_w = int(inner_r * 0.46)
        arrow_h = int(inner_r * 0.36)
        arrow_inset = int(arrow_w * 0.32)

        # Left arrow ◀
        draw.polygon([
            (center - arrow_w, center),
            (center - arrow_inset, center - arrow_h),
            (center - arrow_inset, center + arrow_h)
        ], fill=line_col)

        # Right arrow ▶
        draw.polygon([
            (center + arrow_w, center),
            (center + arrow_inset, center - arrow_h),
            (center + arrow_inset, center + arrow_h)
        ], fill=line_col)

        # Downsample with Lanczos filter to target diameter for retina smoothness
        self.cached_handle_icon = icon.resize((diameter, diameter), Image.Resampling.LANCZOS)
        self.cached_handle_diam = diameter
        return self.cached_handle_icon

    def _build_ui(self):
        # Top toolbar
        toolbar = ctk.CTkFrame(self, fg_color=OceanTheme.BG_CARD, height=48, corner_radius=8)
        toolbar.pack(fill="x", padx=12, pady=(12, 8))

        mode_label = ctk.CTkLabel(
            toolbar,
            text="👁 View Mode:",
            font=OceanTheme.FONT_BODY_BOLD,
            text_color=OceanTheme.TEXT_SECONDARY
        )
        mode_label.pack(side="left", padx=(14, 8), pady=8)

        self.seg_mode = ctk.CTkSegmentedButton(
            toolbar,
            values=["Split View", "Watermarked", "Original", "Residual Heatmap"],
            font=OceanTheme.FONT_BODY,
            selected_color=OceanTheme.OCEAN_BLUE,
            selected_hover_color=OceanTheme.AQUA_SECONDARY,
            unselected_color=OceanTheme.BG_INPUT,
            unselected_hover_color=OceanTheme.BG_HOVER,
            text_color=OceanTheme.TEXT_PRIMARY,
            command=self._on_mode_change
        )
        self.seg_mode.set("Split View")
        self.seg_mode.pack(side="left", padx=6, pady=8)

        # Display Container (holds image canvas directly)
        self.display_container = ctk.CTkFrame(self, fg_color="transparent")
        self.display_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.display_container.bind("<Configure>", self._on_container_configure)

    def _on_container_configure(self, event):
        """Debounced resize handler ensuring clean re-render on window layout changes."""
        if event.widget != self.display_container:
            return
        scaling = self._get_widget_scaling()
        curr_w = int(event.width / scaling)
        curr_h = int(event.height / scaling)
        if abs(curr_w - self._last_w) > 8 or abs(curr_h - self._last_h) > 8:
            self._last_w = curr_w
            self._last_h = curr_h
            if self._resize_timer is not None:
                self.after_cancel(self._resize_timer)
            self._resize_timer = self.after(60, self._render_current_mode)

    def set_original_image(self, image: Image.Image, name: str = ""):
        """Sets the active host cover image and clears stale watermarks."""
        self.original_image = image
        self.watermarked_image = None
        self.difference_image = None
        self._render_current_mode()

    def set_watermark_result(self, watermarked: Image.Image, difference: Image.Image, status_msg: str = ""):
        """Updates the viewer with newly generated watermarked and difference images."""
        self.watermarked_image = watermarked
        self.difference_image = difference
        self._render_current_mode()

    def _on_mode_change(self, mode: str):
        self.view_mode = mode
        self._render_current_mode()

    def _render_current_mode(self):
        """Rebuilds the display frame based on selected mode."""
        self._resize_timer = None
        for widget in self.display_container.winfo_children():
            widget.destroy()

        if self.original_image is None:
            empty_lbl = ctk.CTkLabel(
                self.display_container,
                text="🌊 No image selected\nChoose an example from the sanctuary or import your own image.",
                font=OceanTheme.FONT_TITLE,
                text_color=OceanTheme.TEXT_MUTED
            )
            empty_lbl.pack(expand=True)
            return

        scaling = self._get_widget_scaling()
        raw_w = self.display_container.winfo_width()
        raw_h = self.display_container.winfo_height()

        # Convert physical pixels to CTk logical units
        container_w = int(raw_w / scaling) if raw_w > 50 else 600
        container_h = int(raw_h / scaling) if raw_h > 50 else 500

        if self.view_mode == "Split View":
            self._render_split_view(container_w, container_h)
        elif self.view_mode == "Watermarked":
            target = self.watermarked_image if self.watermarked_image else self.original_image
            title = "Watermarked Output" if self.watermarked_image else "Watermarked (Showing Original)"
            self._render_single_image(target, title, container_w, container_h, badge="WATERMARKED" if self.watermarked_image else "PENDING")
        elif self.view_mode == "Original":
            self._render_single_image(self.original_image, "Original Cover Image", container_w, container_h, badge="ORIGINAL")
        elif self.view_mode == "Residual Heatmap":
            target = self.difference_image if self.difference_image else self._generate_blank_diff()
            title = "Watermark Residual Heatmap" if self.difference_image else "Residual Heatmap"
            self._render_single_image(target, title, container_w, container_h, badge="AMPLIFIED RESIDUALS")

    def _render_split_view(self, total_w: int, total_h: int):
        frame = ctk.CTkFrame(self.display_container, fg_color=OceanTheme.BG_CARD, corner_radius=10)
        frame.pack(fill="both", expand=True)

        # Header with legend
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(header, text="Interactive Split View", font=OceanTheme.FONT_SUBTITLE, text_color=OceanTheme.CYAN_PRIMARY).pack(side="left")

        self.lbl_legend = ctk.CTkLabel(
            header,
            text=f"◀ Original [{int((1 - self.split_ratio)*100)}%]  |  [{int(self.split_ratio*100)}%] Watermarked ▶",
            font=OceanTheme.FONT_CAPTION,
            text_color=OceanTheme.AQUA_SECONDARY
        )
        self.lbl_legend.pack(side="right")

        # Explicit margins for borders: top, bottom, left, right
        border_x = 28          # Border margin on left and right
        border_y_top = 8       # Border margin above image (below header)
        border_y_bottom = 20   # Border margin below image at bottom
        header_h = 42          # Reserved height for header toolbar

        avail_w = max(int(total_w - (border_x * 2)), 80)
        avail_h = max(int(total_h - header_h - border_y_top - border_y_bottom), 80)

        # Compute optimal display dimensions strictly preserving aspect ratio
        orig_w, orig_h = self.original_image.size
        scale = min(avail_w / orig_w, avail_h / orig_h)
        if orig_w <= avail_w and orig_h <= avail_h:
            scale = 1.0

        self.disp_w = max(int(orig_w * scale), 10)
        self.disp_h = max(int(orig_h * scale), 10)

        # Physical device dimensions for native 1:1 retina pixel resolution
        scaling = self._get_widget_scaling()
        self.phys_w = max(int(self.disp_w * scaling), 10)
        self.phys_h = max(int(self.disp_h * scaling), 10)

        # Pre-scale original and watermarked directly from source to physical resolution with LANCZOS
        self.cached_orig_disp = self.original_image.convert("RGB").resize((self.phys_w, self.phys_h), Image.Resampling.LANCZOS)
        target_wm = self.watermarked_image if self.watermarked_image else self.original_image
        self.cached_wm_disp = target_wm.convert("RGB").resize((self.phys_w, self.phys_h), Image.Resampling.LANCZOS)

        # Framed container with subtle ocean border
        img_container = ctk.CTkFrame(
            frame,
            fg_color=OceanTheme.BG_ABYSS,
            border_width=2,
            border_color=OceanTheme.BORDER_SUBTLE,
            corner_radius=8,
            width=self.disp_w + 4,
            height=self.disp_h + 4
        )
        img_container.pack_propagate(False)
        img_container.pack(expand=True, padx=border_x, pady=(border_y_top, border_y_bottom))

        # Initial composite at physical retina resolution
        composite_pil = self._build_fast_composite()
        ctk_img = ctk.CTkImage(light_image=composite_pil, dark_image=composite_pil, size=(self.disp_w, self.disp_h))

        self.split_label = ctk.CTkLabel(
            img_container,
            text="",
            image=ctk_img,
            width=self.disp_w,
            height=self.disp_h,
            cursor="crosshair"
        )
        self.split_label.image = ctk_img
        self.split_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Bind mouse events directly on the image label
        self.split_label.bind("<Button-1>", self._on_mouse_down)
        self.split_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.split_label.bind("<Motion>", self._on_mouse_hover)

    def _on_mouse_hover(self, event):
        """Changes cursor to double arrow when hovering near the divider line."""
        if self.split_label is None or self.disp_w <= 0:
            return
        label_w = self.split_label.winfo_width()
        denom = label_w if label_w > 1 else max(self.phys_w, 1)
        mouse_ratio = max(0.0, min(1.0, event.x / denom))
        ratio_tolerance = 24.0 / max(self.phys_w, 1)
        if abs(mouse_ratio - self.split_ratio) <= ratio_tolerance:
            self.split_label.configure(cursor="size_we")
        else:
            self.split_label.configure(cursor="crosshair")

    def _on_mouse_down(self, event):
        """Click anywhere on the image to position the divider with exact offset calibration."""
        self._update_split_from_x(event.x)

    def _on_mouse_drag(self, event):
        """Drag to move the divider line in real time."""
        self._update_split_from_x(event.x)

    def _update_split_from_x(self, x: int):
        if self.disp_w <= 0 or self.split_label is None:
            return
        label_w = self.split_label.winfo_width()
        denom = label_w if label_w > 1 else max(self.phys_w, 1)
        ratio = max(0.01, min(0.99, x / denom))
        self.split_ratio = ratio

        if self.lbl_legend:
            self.lbl_legend.configure(
                text=f"◀ Original [{int((1 - self.split_ratio)*100)}%]  |  [{int(self.split_ratio*100)}%] Watermarked ▶"
            )

        # Fast composite update at physical retina resolution
        composite_pil = self._build_fast_composite()
        ctk_img = ctk.CTkImage(light_image=composite_pil, dark_image=composite_pil, size=(self.disp_w, self.disp_h))
        self.split_label.configure(image=ctk_img, cursor="size_we")
        self.split_label.image = ctk_img

    def _build_fast_composite(self) -> Image.Image:
        """Instantly slices and composes physical-resolution images with crisp antialiased divider and handle."""
        w, h = self.phys_w, self.phys_h
        split_x = int(w * self.split_ratio)
        split_x = max(0, min(w, split_x))

        composite = Image.new("RGB", (w, h))

        if self.cached_orig_disp and split_x > 0:
            box_left = (0, 0, split_x, h)
            composite.paste(self.cached_orig_disp.crop(box_left), box_left)

        if self.cached_wm_disp and split_x < w:
            box_right = (split_x, 0, w, h)
            composite.paste(self.cached_wm_disp.crop(box_right), box_right)

        draw = ImageDraw.Draw(composite)

        # Draw glowing bioluminescent divider line at physical device resolution
        scaling = self._get_widget_scaling()
        line_w = max(int(2.5 * scaling), 2)
        glow_w = line_w + 2

        # Subtle glow border
        draw.line([(split_x, 0), (split_x, h)], fill=(0, 180, 216), width=glow_w)
        # Core bright cyan beam
        draw.line([(split_x, 0), (split_x, h)], fill=(0, 245, 212), width=line_w)

        # Central handle circle icon (antialiased supersampled)
        handle_diam = max(int(34 * scaling), 28)
        handle_icon = self._get_or_create_handle_icon(handle_diam)
        half_diam = handle_diam // 2
        mid_y = h // 2

        # Alpha paste on top of divider line
        paste_x = split_x - half_diam
        paste_y = mid_y - half_diam
        composite.paste(handle_icon, (paste_x, paste_y), handle_icon)

        return composite

    def _render_single_image(self, img: Image.Image, title: str, total_w: int, total_h: int, badge: str = ""):
        frame = ctk.CTkFrame(self.display_container, fg_color=OceanTheme.BG_CARD, corner_radius=10)
        frame.pack(fill="both", expand=True)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(header, text=title, font=OceanTheme.FONT_SUBTITLE, text_color=OceanTheme.CYAN_PRIMARY).pack(side="left")
        if badge:
            ctk.CTkLabel(header, text=badge, font=OceanTheme.FONT_CAPTION, text_color=OceanTheme.AQUA_SECONDARY).pack(side="right")

        border_x = 28
        border_y_top = 8
        border_y_bottom = 20
        header_h = 42

        avail_w = max(int(total_w - (border_x * 2)), 80)
        avail_h = max(int(total_h - header_h - border_y_top - border_y_bottom), 80)

        orig_w, orig_h = img.size
        scale = min(avail_w / orig_w, avail_h / orig_h)
        if orig_w <= avail_w and orig_h <= avail_h:
            scale = 1.0

        disp_w = max(int(orig_w * scale), 10)
        disp_h = max(int(orig_h * scale), 10)

        scaling = self._get_widget_scaling()
        phys_w = max(int(disp_w * scaling), 10)
        phys_h = max(int(disp_h * scaling), 10)

        img_container = ctk.CTkFrame(
            frame,
            fg_color=OceanTheme.BG_ABYSS,
            border_width=2,
            border_color=OceanTheme.BORDER_SUBTLE,
            corner_radius=8,
            width=disp_w + 4,
            height=disp_h + 4
        )
        img_container.pack_propagate(False)
        img_container.pack(expand=True, padx=border_x, pady=(border_y_top, border_y_bottom))

        scaled_pil = img.convert("RGB").resize((phys_w, phys_h), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=scaled_pil, dark_image=scaled_pil, size=(disp_w, disp_h))
        lbl_img = ctk.CTkLabel(img_container, text="", image=ctk_img, width=disp_w, height=disp_h)
        lbl_img.image = ctk_img
        lbl_img.pack(fill="both", expand=True, padx=2, pady=2)

    def _generate_blank_diff(self) -> Image.Image:
        w, h = self.original_image.size if self.original_image else (512, 512)
        return Image.new("RGB", (w, h), (10, 20, 35))

    def _scale_to_ctk(self, pil_image: Image.Image, max_w: int, max_h: int) -> ctk.CTkImage:
        orig_w, orig_h = pil_image.size
        if max_w <= 10 or max_h <= 10:
            scale = 1.0
        else:
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)

        new_w = max(int(orig_w * scale), 1)
        new_h = max(int(orig_h * scale), 1)

        scaled_pil = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return ctk.CTkImage(light_image=scaled_pil, dark_image=scaled_pil, size=(new_w, new_h))
