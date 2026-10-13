"""
Automated unit and integration test for SlayMarking baseline application.
Tests:
- 6 research watermarking methods
- Direct on-image draggable divider composite
- Custom image deletion
- Paper information modal
- Full UI lifecycle and auto-application
"""
import unittest
from PIL import Image
from watermarking.registry import registry
from ui.app import SlayMarkingApp

class TestSlayMarking(unittest.TestCase):

    def test_registry_has_six_methods(self):
        methods = registry.list_methods()
        self.assertEqual(len(methods), 6, "Expected exactly 6 registered watermarking methods.")

    def test_each_method_metadata(self):
        for method in registry.list_methods():
            meta = method.get_metadata()
            self.assertTrue(meta.method_id)
            self.assertTrue(meta.title)
            self.assertTrue(meta.paper_title)
            self.assertTrue(meta.authors)
            self.assertGreater(meta.year, 1990)

    def test_direct_divider_dragging(self):
        app = SlayMarkingApp()
        app.update()

        # Test updating split ratio via click/drag coordinate
        app.viewer._update_split_from_x(250)
        app.update()
        self.assertGreater(app.viewer.split_ratio, 0.0)
        self.assertLess(app.viewer.split_ratio, 1.0)

        # Test fast composite generation
        comp = app.viewer._build_fast_composite()
        self.assertEqual(comp.size, (app.viewer.phys_w, app.viewer.phys_h))

        app.destroy()

    def test_custom_image_delete(self):
        app = SlayMarkingApp()
        app.update()
        
        initial_count = len(app.sidebar.image_entries)
        custom_img = Image.new("RGB", (100, 100), (10, 50, 90))
        app.sidebar._add_image_entry("Test Custom", "test_path.png", custom_img, is_custom=True)
        self.assertEqual(len(app.sidebar.image_entries), initial_count + 1)
        
        custom_entry = app.sidebar.image_entries[-1]
        self.assertTrue(custom_entry["is_custom"])
        app.sidebar._delete_image_entry(custom_entry)
        self.assertEqual(len(app.sidebar.image_entries), initial_count)
        
        app.destroy()

    def test_paper_info_modal(self):
        app = SlayMarkingApp()
        app.update()
        
        app.method_panel._open_paper_info_modal()
        app.update()
        
        app.destroy()

    def test_ui_lifecycle_and_auto_apply(self):
        app = SlayMarkingApp()
        app.update()

        self.assertGreaterEqual(len(app.sidebar.image_entries), 1)

        for i, method in enumerate(app.method_panel.methods):
            app.method_panel._select_method(i, trigger_apply=True)
            app.update()

        for mode in ["Split View", "Watermarked", "Original", "Residual Heatmap"]:
            app.viewer._on_mode_change(mode)
            app.update()

        app.destroy()
        print("All updated baseline features verified successfully.")

if __name__ == "__main__":
    unittest.main()
