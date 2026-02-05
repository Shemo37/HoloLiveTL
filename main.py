import sys
import os
import warnings
import tkinter as tk
from queue import Queue

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

# Environment optimizations
os.environ['TRANSFORMERS_VERBOSITY'] = 'warning'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def main():
    """Main application entry point"""
    try:
        # Import modules after path setup
        from modules import Config, TranslatorStats
        from gui import ControlGUI
        
        # Initialize global objects
        config = Config()
        stats = TranslatorStats()
        gui_queue = Queue()
        
        # Create and run GUI
        root = tk.Tk()
        app = ControlGUI(root, config, stats, gui_queue)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        
        print("🚀 Live Translator (Organized) starting...")
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_modular.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Handle PyInstaller frozen executable
    if hasattr(sys, '_MEIPASS'):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    main()
