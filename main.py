"""
SlayMarking - Deep Ocean Image Watermarking Suite
Main application entry point.
"""
import sys
import os

# Ensure current working directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ui.app import SlayMarkingApp

def main():
    """Launches the SlayMarking GUI window application."""
    app = SlayMarkingApp()
    app.mainloop()

if __name__ == "__main__":
    main()
