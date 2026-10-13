"""
Sidebar component for SlayMarking.
Manages the image gallery (bundled examples + custom user uploads), deletion of custom photos,
and nautical branding.
"""
import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from typing import Callable, List, Dict, Optional
from ui.theme import OceanTheme

class Sidebar(ctk.CTkFrame):
    """Deep ocean sidebar containing app branding and image selector."""

    def __init__(
        self,
        master,
        on_image_selected: Callable[[str, Image.Image], None],
        **kwargs
    ):
        super().__init__(master, fg_color=OceanTheme.BG_CARD, corner_radius=0, **kwargs)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.on_image_selected = on_image_selected
        
        # Track loaded images: list of dict
        self.image_entries: List[Dict] = []
        self.selected_index: int = 0
        self.row_widgets: List[Dict] = []

        self._build_ui()
        self._load_example_images()

    def _build_ui(self):
        # Header / Branding
        brand_frame = ctk.CTkFrame(self, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(18, 10))

        title_label = ctk.CTkLabel(
            brand_frame,
            text="🌊 SlayMarking",
            font=OceanTheme.FONT_HERO,
            text_color=OceanTheme.CYAN_PRIMARY
        )
        title_label.pack(anchor="w")

        # Divider
        divider = ctk.CTkFrame(self, fg_color=OceanTheme.BORDER_SUBTLE, height=1)
        divider.pack(fill="x", padx=14, pady=(12, 12))

        # Actions section: Add Custom Image button
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(fill="x", padx=14, pady=(0, 10))

        add_btn = ctk.CTkButton(
            action_frame,
            text="➕ Import Custom Image",
            font=OceanTheme.FONT_BODY_BOLD,
            fg_color=OceanTheme.BG_HOVER,
            hover_color=OceanTheme.BG_ACTIVE,
            text_color=OceanTheme.CYAN_PRIMARY,
            border_width=1,
            border_color=OceanTheme.BORDER_CYAN,
            corner_radius=8,
            height=36,
            command=self._on_import_image
        )
        add_btn.pack(fill="x")

        # Gallery Title
        gallery_header = ctk.CTkLabel(
            self,
            text="IMAGE SANCTUARY",
            font=OceanTheme.FONT_CAPTION,
            text_color=OceanTheme.TEXT_MUTED
        )
        gallery_header.pack(anchor="w", padx=16, pady=(6, 4))

        # Scrollable image gallery list
        self.scroll_gallery = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=OceanTheme.BG_HOVER,
            scrollbar_button_hover_color=OceanTheme.CYAN_PRIMARY
        )
        self.scroll_gallery.pack(fill="both", expand=True, padx=8, pady=(0, 12))

    def _load_example_images(self):
        """Loads bundled example images from assets/examples."""
        example_dir = os.path.join("assets", "examples")
        if os.path.exists(example_dir):
            for filename in sorted(os.listdir(example_dir)):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    full_path = os.path.join(example_dir, filename)
                    try:
                        pil_img = Image.open(full_path)
                        self._add_image_entry(
                            name=self._format_name(filename),
                            path=full_path,
                            image=pil_img,
                            is_custom=False
                        )
                    except Exception as e:
                        print(f"Error loading {full_path}: {e}")

        if self.image_entries:
            self._select_index(0)

    def _format_name(self, filename: str) -> str:
        base = os.path.splitext(filename)[0]
        return base.replace("_", " ").title()

    def _on_import_image(self):
        """File dialog to import custom user image."""
        file_path = filedialog.askopenfilename(
            title="Select Cover Image to Watermark",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                pil_img = Image.open(file_path)
                name = self._format_name(os.path.basename(file_path))
                self._add_image_entry(
                    name=name,
                    path=file_path,
                    image=pil_img,
                    is_custom=True
                )
                # Select the newly imported custom image
                self._select_index(len(self.image_entries) - 1)
            except Exception as e:
                print(f"Failed to open custom image: {e}")

    def _add_image_entry(self, name: str, path: str, image: Image.Image, is_custom: bool):
        """Creates a gallery row for an image with delete option for custom photos."""
        idx = len(self.image_entries)
        
        # Create thumbnail
        thumb_size = (36, 36)
        thumb_pil = image.copy()
        thumb_pil.thumbnail(thumb_size)
        ctk_thumb = ctk.CTkImage(light_image=thumb_pil, dark_image=thumb_pil, size=thumb_size)

        entry = {
            "name": name,
            "path": path,
            "image": image,
            "is_custom": is_custom,
            "thumb": ctk_thumb
        }
        self.image_entries.append(entry)

        # Row container frame
        row_frame = ctk.CTkFrame(self.scroll_gallery, fg_color="transparent")
        row_frame.pack(fill="x", pady=2)

        tag = "👤 Custom" if is_custom else "🌊 Example"
        btn_text = f"{name}\n{tag} • {image.width}×{image.height}"

        btn = ctk.CTkButton(
            row_frame,
            text=btn_text,
            image=ctk_thumb,
            compound="left",
            anchor="w",
            font=OceanTheme.FONT_BODY,
            fg_color=OceanTheme.BG_CARD_ALT,
            hover_color=OceanTheme.BG_HOVER,
            text_color=OceanTheme.TEXT_PRIMARY,
            corner_radius=8,
            height=48,
            command=lambda i=idx: self._select_by_entry(entry)
        )
        if hasattr(btn, "_text_label") and btn._text_label:
            btn._text_label.configure(justify="left", anchor="w")
        btn.pack(side="left", fill="x", expand=True)

        btn_delete = None
        if is_custom:
            # Delete button (trash bin) only for custom images
            btn_delete = ctk.CTkButton(
                row_frame,
                text="🗑️",
                width=34,
                height=48,
                font=("Segoe UI Emoji", 13),
                fg_color=OceanTheme.BG_CARD_ALT,
                hover_color="#3D121A",
                text_color=OceanTheme.CORAL_ALERT,
                corner_radius=8,
                command=lambda e=entry: self._delete_image_entry(e)
            )
            btn_delete.pack(side="right", padx=(4, 0))

        self.row_widgets.append({
            "entry": entry,
            "row_frame": row_frame,
            "button": btn,
            "delete_button": btn_delete
        })

    def _select_by_entry(self, target_entry: Dict):
        for i, entry in enumerate(self.image_entries):
            if entry is target_entry:
                self._select_index(i)
                break

    def _delete_image_entry(self, entry_to_delete: Dict):
        """Deletes a custom image from the gallery."""
        if not entry_to_delete.get("is_custom", False):
            return

        # Find index
        deleted_idx = -1
        for i, e in enumerate(self.image_entries):
            if e is entry_to_delete:
                deleted_idx = i
                break

        if deleted_idx == -1:
            return

        # Destroy row widget
        for row in self.row_widgets:
            if row["entry"] is entry_to_delete:
                row["row_frame"].destroy()
                self.row_widgets.remove(row)
                break

        self.image_entries.pop(deleted_idx)

        # If deleted item was currently selected, select another image
        if self.image_entries:
            new_idx = min(deleted_idx, len(self.image_entries) - 1)
            self._select_index(new_idx)
        else:
            self.selected_index = -1

    def _select_index(self, index: int):
        if not (0 <= index < len(self.image_entries)):
            return

        self.selected_index = index
        selected_entry = self.image_entries[index]

        # Update button highlights
        for row in self.row_widgets:
            if row["entry"] is selected_entry:
                row["button"].configure(
                    fg_color=OceanTheme.BG_ACTIVE,
                    border_width=2,
                    border_color=OceanTheme.CYAN_PRIMARY,
                    text_color=OceanTheme.CYAN_PRIMARY
                )
            else:
                row["button"].configure(
                    fg_color=OceanTheme.BG_CARD_ALT,
                    border_width=0,
                    text_color=OceanTheme.TEXT_PRIMARY
                )
            if hasattr(row["button"], "_text_label") and row["button"]._text_label:
                row["button"]._text_label.configure(justify="left", anchor="w")

        # Notify parent app
        self.on_image_selected(selected_entry["name"], selected_entry["image"])

    def get_current_image(self) -> Optional[Image.Image]:
        if 0 <= self.selected_index < len(self.image_entries):
            return self.image_entries[self.selected_index]["image"]
        return None
