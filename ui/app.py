"""
Main application window for SlayMarking.
Combines Sidebar, ImageViewer, and MethodPanel with cohesive deep water aesthetics.
"""
import customtkinter as ctk
from PIL import Image
from typing import Dict, Any, Optional

from ui.theme import OceanTheme
from ui.components import Sidebar, ImageViewer, MethodPanel
from watermarking.base import BaseWatermarkMethod, WatermarkResult

class SlayMarkingApp(ctk.CTk):
    """Deep Ocean Watermarking Application Window."""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("SlayMarking — Deep Ocean Image Watermarking Suite")
        self.geometry("1400x880")
        self.minsize(1080, 680)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=OceanTheme.BG_ABYSS)

        # Maximize / Fullscreen on open
        self.after(100, lambda: self._safe_maximize())

        # State
        self.current_cover_image: Optional[Image.Image] = None
        self.current_cover_name: str = ""

        self._build_layout()
        self._bind_shortcuts()

        # Automatically apply the initial watermarking method
        self.after(200, lambda: self.method_panel.trigger_apply())

    def _safe_maximize(self):
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                pass

    def _build_layout(self):
        # 3-column layout:
        # Col 0: Sidebar (Gallery & Uploads)
        # Col 1: Center Viewer (Split slider & difference display)
        # Col 2: Method Panel (Auto-applied parameters & paper info bump)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=290)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=330)

        # Center Image Viewer
        self.viewer = ImageViewer(
            self
        )
        self.viewer.grid(row=0, column=1, sticky="nsew", padx=2)

        # Right Method Control Panel (streamlined with auto-apply)
        self.method_panel = MethodPanel(
            self,
            on_apply_watermark=self._on_apply_watermark,
            width=330
        )
        self.method_panel.grid(row=0, column=2, sticky="nsew")

        # Left Sidebar (Gallery & Uploads)
        self.sidebar = Sidebar(
            self,
            on_image_selected=self._on_image_selected,
            width=290
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self.sidebar._on_import_image())

    def _on_image_selected(self, name: str, image: Image.Image):
        """Called when an image is selected from the sidebar gallery."""
        self.current_cover_image = image
        self.current_cover_name = name
        self.viewer.set_original_image(image, name)
        # Automatically apply the currently selected watermarking method to the new image
        self.method_panel.trigger_apply()

    def _on_apply_watermark(self, method: BaseWatermarkMethod, watermark_data: str, params: Dict[str, Any]):
        """Executes watermark simulation and updates display."""
        if self.current_cover_image is None:
            return

        try:
            result: WatermarkResult = method.embed(
                image=self.current_cover_image,
                watermark_data=watermark_data,
                params=params
            )

            if result.status == "error":
                self._handle_watermark_error(method, result.message)
                return

            # Update viewer with watermarked output and amplified difference residual map
            self.viewer.clear_error()
            self.viewer.set_watermark_result(
                watermarked=result.watermarked_image,
                difference=result.difference_map,
                status_msg=result.message
            )
            self.method_panel.set_status(f"✓ {method.get_metadata().short_name} applied", is_error=False)

        except Exception as e:
            self._handle_watermark_error(method, str(e))

    def _handle_watermark_error(self, method: BaseWatermarkMethod, error_msg: str):
        """Displays error notice across viewer and control panel."""
        self.viewer.show_error(error_msg, method_name=method.get_metadata().short_name)
        short_err = error_msg.splitlines()[0]
        if len(short_err) > 42:
            short_err = short_err[:39] + "..."
        self.method_panel.set_status(f"❌ {short_err}", is_error=True)
