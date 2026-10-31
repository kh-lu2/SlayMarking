"""
Method Control Panel for SlayMarking.
Features:
- Clean non-scrollable layout (no scrollbars).
- Perfectly aligned, centered, and balanced spacing.
- Ample room for method selector and 'ℹ️ Info' modal button without text truncation.
- Dynamic parameters without text clipping.
"""
import customtkinter as ctk
from PIL import Image
from typing import Callable, Dict, Any, Optional
from ui.theme import OceanTheme
from watermarking.registry import registry
from watermarking.base import BaseWatermarkMethod, WatermarkResult, MethodMetadata

class MethodPanel(ctk.CTkFrame):
    """Deep ocean control panel for watermarking algorithms."""

    def __init__(
        self,
        master,
        on_apply_watermark: Callable[[BaseWatermarkMethod, str, Dict[str, Any]], None],
        width: int = 330,
        **kwargs
    ):
        super().__init__(master, fg_color=OceanTheme.BG_CARD, corner_radius=0, width=width, **kwargs)
        self.pack_propagate(False)
        self.grid_propagate(False)

        self.on_apply_watermark = on_apply_watermark

        self.methods = registry.list_methods()
        self.current_method_idx: int = 0
        self.param_widgets: Dict[str, Any] = {}
        self.param_values: Dict[str, Any] = {}
        self._debounce_timer = None

        self._build_ui()
        self._select_method(0, trigger_apply=False)

    def _build_ui(self):
        # Clean non-scrollable main container with compact margins
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=14, pady=14)

        # 1. Section Header: Method Selection
        sec_header = ctk.CTkLabel(
            self.container,
            text="🔬 WATERMARKING METHOD",
            font=OceanTheme.FONT_SUBTITLE,
            text_color=OceanTheme.CYAN_PRIMARY
        )
        sec_header.pack(anchor="w", pady=(0, 6))

        # Row containing Method Selector + Light bump info button
        selector_row = ctk.CTkFrame(self.container, fg_color="transparent")
        selector_row.pack(fill="x", pady=(0, 12))

        method_names = [f"{i+1}. {m.get_metadata().short_name}" for i, m in enumerate(self.methods)]
        self.opt_method = ctk.CTkOptionMenu(
            selector_row,
            values=method_names,
            font=OceanTheme.FONT_BODY_BOLD,
            fg_color=OceanTheme.BG_HOVER,
            button_color=OceanTheme.NAVY_DEEP,
            button_hover_color=OceanTheme.AQUA_SECONDARY,
            dropdown_fg_color=OceanTheme.BG_CARD_ALT,
            dropdown_hover_color=OceanTheme.BG_ACTIVE,
            dropdown_text_color=OceanTheme.TEXT_PRIMARY,
            text_color=OceanTheme.TEXT_PRIMARY,
            corner_radius=8,
            height=36,
            dynamic_resizing=False,
            command=self._on_method_dropdown_change
        )
        self.opt_method.pack(side="left", fill="x", expand=True, padx=(0, 6))

        # Light bump / pill button for paper information modal
        self.btn_info_bump = ctk.CTkButton(
            selector_row,
            text="ℹ️ Info",
            width=68,
            height=36,
            font=OceanTheme.FONT_BODY_BOLD,
            fg_color=OceanTheme.BG_CARD_ALT,
            hover_color=OceanTheme.BG_ACTIVE,
            text_color=OceanTheme.CYAN_PRIMARY,
            border_width=1,
            border_color=OceanTheme.BORDER_CYAN,
            corner_radius=8,
            command=self._open_paper_info_modal
        )
        self.btn_info_bump.pack(side="right")

        # 2. Watermark Payload Input
        sec_payload = ctk.CTkLabel(
            self.container,
            text="🔑 WATERMARK PAYLOAD / KEY",
            font=OceanTheme.FONT_SUBTITLE,
            text_color=OceanTheme.CYAN_PRIMARY
        )
        sec_payload.pack(anchor="w", pady=(0, 6))

        self.entry_payload = ctk.CTkEntry(
            self.container,
            placeholder_text="Enter secret watermark string or cryptographic key...",
            font=OceanTheme.FONT_BODY,
            fg_color=OceanTheme.BG_INPUT,
            border_color=OceanTheme.BORDER_SUBTLE,
            text_color=OceanTheme.TEXT_PRIMARY,
            corner_radius=6,
            height=36
        )
        self.entry_payload.insert(0, "© SLAYMARKING 2026 / ABYSS")
        self.entry_payload.pack(fill="x", pady=(0, 16))
        self.entry_payload.bind("<KeyRelease>", lambda e: self._schedule_auto_apply())

        # 3. Dynamic Algorithm Parameters
        sec_params = ctk.CTkLabel(
            self.container,
            text="⚙ ALGORITHM PARAMETERS",
            font=OceanTheme.FONT_SUBTITLE,
            text_color=OceanTheme.CYAN_PRIMARY
        )
        sec_params.pack(anchor="w", pady=(0, 8))

        self.params_container = ctk.CTkFrame(self.container, fg_color="transparent")
        self.params_container.pack(fill="x", pady=(0, 12))

        # Subtle auto-apply status indicator
        self.lbl_auto_status = ctk.CTkLabel(
            self.container,
            text="⚡ Watermark applied automatically",
            font=OceanTheme.FONT_CAPTION,
            text_color=OceanTheme.SEAFOAM
        )
        self.lbl_auto_status.pack(anchor="w", pady=(4, 0))

    def _on_method_dropdown_change(self, choice_text: str):
        idx_str = choice_text.split(".")[0].strip()
        idx = int(idx_str) - 1
        self._select_method(idx, trigger_apply=True)

    def _select_method(self, index: int, trigger_apply: bool = True):
        self.current_method_idx = index
        method = self.methods[index]
        self._build_parameters_ui(method)
        if trigger_apply:
            self.trigger_apply()

    def _build_parameters_ui(self, method: BaseWatermarkMethod):
        for widget in self.params_container.winfo_children():
            widget.destroy()

        self.param_widgets.clear()
        self.param_values.clear()

        params = method.get_parameters()
        for p in params:
            p_frame = ctk.CTkFrame(self.params_container, fg_color=OceanTheme.BG_CARD_ALT, corner_radius=8)
            p_frame.pack(fill="x", pady=5)

            hdr = ctk.CTkFrame(p_frame, fg_color="transparent")
            hdr.pack(fill="x", padx=10, pady=(8, 2))

            ctk.CTkLabel(hdr, text=p.label, font=OceanTheme.FONT_BODY_BOLD, text_color=OceanTheme.TEXT_PRIMARY).pack(side="left")
            val_lbl = ctk.CTkLabel(hdr, text=str(p.default), font=OceanTheme.FONT_CAPTION, text_color=OceanTheme.AQUA_SECONDARY)
            val_lbl.pack(side="right")

            if p.description:
                desc_lbl = ctk.CTkLabel(
                    p_frame,
                    text=p.description,
                    font=OceanTheme.FONT_CAPTION,
                    text_color=OceanTheme.TEXT_MUTED,
                    wraplength=275,
                    justify="left"
                )
                desc_lbl.pack(anchor="w", padx=10, pady=(0, 4))

            if p.param_type in ("float", "int"):
                min_v = p.min_value if p.min_value is not None else 0.0
                max_v = p.max_value if p.max_value is not None else 1.0

                def make_slider_cb(p_name=p.name, p_type=p.param_type, lbl=val_lbl):
                    return lambda val: self._on_slider_changed(p_name, p_type, val, lbl)

                slider = ctk.CTkSlider(
                    p_frame,
                    from_=min_v,
                    to=max_v,
                    number_of_steps=int((max_v - min_v) / (p.step or 0.01)) if p.step else 50,
                    progress_color=OceanTheme.CYAN_PRIMARY,
                    button_color=OceanTheme.CYAN_PRIMARY,
                    button_hover_color=OceanTheme.AQUA_SECONDARY,
                    command=make_slider_cb()
                )
                slider.set(p.default)
                slider.pack(fill="x", padx=10, pady=(0, 10))

                self.param_widgets[p.name] = slider
                self.param_values[p.name] = p.default

            elif p.param_type == "choice" and p.choices:
                def make_choice_cb(p_name=p.name, lbl=val_lbl):
                    return lambda val: self._on_choice_changed(p_name, val, lbl)

                opt = ctk.CTkOptionMenu(
                    p_frame,
                    values=p.choices,
                    font=OceanTheme.FONT_BODY,
                    fg_color=OceanTheme.BG_INPUT,
                    button_color=OceanTheme.NAVY_DEEP,
                    button_hover_color=OceanTheme.AQUA_SECONDARY,
                    dropdown_fg_color=OceanTheme.BG_CARD,
                    corner_radius=6,
                    height=30,
                    dynamic_resizing=False,
                    command=make_choice_cb()
                )
                opt.set(str(p.default))
                opt.pack(fill="x", padx=10, pady=(0, 10))

                self.param_widgets[p.name] = opt
                self.param_values[p.name] = p.default

            elif p.param_type == "bool":
                def make_bool_cb(p_name=p.name, lbl=val_lbl):
                    return lambda: self._on_bool_changed(p_name, lbl)

                chk = ctk.CTkCheckBox(
                    p_frame,
                    text="Enabled",
                    font=OceanTheme.FONT_BODY,
                    checkbox_height=20,
                    checkbox_width=20,
                    checkmark_color=OceanTheme.TEXT_DARK,
                    fg_color=OceanTheme.CYAN_PRIMARY,
                    hover_color=OceanTheme.CYAN_HOVER,
                    command=make_bool_cb()
                )
                if p.default:
                    chk.select()
                else:
                    chk.deselect()
                chk.pack(anchor="w", padx=10, pady=(0, 10))

                self.param_widgets[p.name] = chk
                self.param_values[p.name] = p.default

    def _on_slider_changed(self, p_name: str, p_type: str, val: float, lbl: ctk.CTkLabel):
        if p_type == "int":
            int_val = int(round(val))
            self.param_values[p_name] = int_val
            lbl.configure(text=str(int_val))
        else:
            float_val = round(val, 3)
            self.param_values[p_name] = float_val
            lbl.configure(text=str(float_val))
        self._schedule_auto_apply()

    def _on_choice_changed(self, p_name: str, val: str, lbl: ctk.CTkLabel):
        self.param_values[p_name] = val
        lbl.configure(text=val)
        self.trigger_apply()

    def _on_bool_changed(self, p_name: str, lbl: ctk.CTkLabel):
        chk = self.param_widgets.get(p_name)
        val = bool(chk.get()) if chk else False
        self.param_values[p_name] = val
        lbl.configure(text="True" if val else "False")
        self.trigger_apply()

    def _schedule_auto_apply(self):
        """Debounces slider movements and keystrokes to smoothly reapply."""
        if self._debounce_timer is not None:
            self.after_cancel(self._debounce_timer)
        self._debounce_timer = self.after(150, self.trigger_apply)

    def set_status(self, text: str, is_error: bool = False):
        """Updates the status label with success or error styling."""
        color = OceanTheme.CORAL_ALERT if is_error else OceanTheme.SEAFOAM
        self.lbl_auto_status.configure(text=text, text_color=color)

    def trigger_apply(self):
        """Immediately applies watermark."""
        self._debounce_timer = None
        method = self.methods[self.current_method_idx]
        payload = self.entry_payload.get().strip() or "DEFAULT_WATERMARK"
        self.lbl_auto_status.configure(text="⚡ Applying watermark...", text_color=OceanTheme.AQUA_SECONDARY)
        self.on_apply_watermark(method, payload, self.param_values)

    def _open_paper_info_modal(self):
        """Opens an in-app overlay modal card directly inside the main application window."""
        method = self.methods[self.current_method_idx]
        meta: MethodMetadata = method.get_metadata()

        root = self.winfo_toplevel()

        # If an overlay already exists, remove it first
        if hasattr(self, "_active_info_overlay") and self._active_info_overlay is not None:
            try:
                self._active_info_overlay.destroy()
            except Exception:
                pass

        # In-app backdrop overlay covering the application
        overlay = ctk.CTkFrame(root, fg_color="#040A16", corner_radius=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._active_info_overlay = overlay

        # Centered in-app modal card
        modal_card = ctk.CTkFrame(
            overlay,
            fg_color=OceanTheme.BG_CARD,
            border_width=2,
            border_color=OceanTheme.BORDER_CYAN,
            corner_radius=12,
            width=640,
            height=480
        )
        modal_card.place(relx=0.5, rely=0.5, anchor="center")
        modal_card.pack_propagate(False)

        def close_overlay(e=None):
            if hasattr(self, "_active_info_overlay") and self._active_info_overlay:
                self._active_info_overlay.destroy()
                self._active_info_overlay = None

        # Bind escape key and backdrop click to dismiss
        root.bind("<Escape>", close_overlay)
        overlay.bind("<Button-1>", lambda e: close_overlay() if e.widget == overlay else None)

        content = ctk.CTkFrame(modal_card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=24, pady=20)

        # Top row: Domain badge + ✕ button
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 6))

        badge = ctk.CTkLabel(
            top_row,
            text=f"🔬 DOMAIN: {meta.domain.upper()}",
            font=OceanTheme.FONT_BODY_BOLD,
            text_color=OceanTheme.CYAN_PRIMARY
        )
        badge.pack(side="left")

        btn_x = ctk.CTkButton(
            top_row,
            text="✕",
            width=32,
            height=32,
            font=("Segoe UI", 13, "bold"),
            fg_color=OceanTheme.BG_CARD_ALT,
            hover_color=OceanTheme.CORAL_ALERT,
            text_color=OceanTheme.TEXT_SECONDARY,
            corner_radius=6,
            command=close_overlay
        )
        btn_x.pack(side="right")

        # Paper title
        title_lbl = ctk.CTkLabel(
            content,
            text=meta.paper_title,
            font=OceanTheme.FONT_TITLE,
            text_color=OceanTheme.TEXT_PRIMARY,
            wraplength=580,
            justify="left"
        )
        title_lbl.pack(anchor="w", pady=(0, 6))

        # Citation info
        cite_lbl = ctk.CTkLabel(
            content,
            text=f"Authors: {meta.authors}\nPublished: {meta.year} • {meta.venue}",
            font=OceanTheme.FONT_BODY,
            text_color=OceanTheme.AQUA_SECONDARY,
            wraplength=580,
            justify="left"
        )
        cite_lbl.pack(anchor="w", pady=(0, 10))

        # Abstract box
        abs_frame = ctk.CTkFrame(content, fg_color=OceanTheme.BG_CARD_ALT, corner_radius=8)
        abs_frame.pack(fill="both", expand=True, pady=(0, 4))

        ctk.CTkLabel(
            abs_frame,
            text="PAPER ABSTRACT / SUMMARY",
            font=OceanTheme.FONT_CAPTION,
            text_color=OceanTheme.TEXT_MUTED
        ).pack(anchor="w", padx=14, pady=(10, 4))

        abs_text = ctk.CTkLabel(
            abs_frame,
            text=meta.abstract or meta.description,
            font=OceanTheme.FONT_BODY,
            text_color=OceanTheme.TEXT_SECONDARY,
            wraplength=550,
            justify="left"
        )
        abs_text.pack(anchor="w", padx=14, pady=(0, 10))
