"""
Main GUI window for Live Translator - Enhanced Version
"""
import time
import threading
from queue import Queue, Empty
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, Label, colorchooser
from datetime import datetime
import json
import os
import traceback
import sys
import io
import logging
from collections import deque

# Import our modular components
from modules.config import Config
from modules.stats import TranslatorStats
from modules.audio_utils import find_audio_device
from modules.recorder import recorder_thread
from modules.processor import processor_thread
from modules.model_utils import ensure_model_downloaded
from modules.config import MODEL_ID

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import torchaudio
except ImportError:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependency Error", "torchaudio not found. Please run 'pip install torchaudio' in your terminal.")
    root.destroy()
    raise ImportError("torchaudio module not found")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependency Error", "huggingface_hub not found. Please run 'pip install huggingface_hub' in your terminal.")
    root.destroy()
    raise ImportError("huggingface_hub module not found")

try:
    import soundcard as sc
except ImportError:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependency Error", "soundcard not found. Please run 'pip install soundcard' in your terminal.")
    root.destroy()
    raise ImportError("soundcard module not found")


class TranslationHistoryPanel:
    """Separate panel for displaying translation history"""
    def __init__(self, parent, on_copy_callback=None):
        self.parent = parent
        self.on_copy = on_copy_callback
        self.history = []
        self.window = None
        self.listbox = None
        self.is_visible = False

    def show(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            self.window.focus_force()
            return

        self.window = tk.Toplevel(self.parent)
        self.window.title("Translation History")
        self.window.geometry("500x400")
        self.window.minsize(400, 300)
        self.is_visible = True

        # Main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Header with count
        header_frame = tk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 5))

        self.count_label = tk.Label(header_frame, text=f"Total: {len(self.history)} translations",
                                     font=("Helvetica", 10))
        self.count_label.pack(side='left')

        # Buttons
        btn_frame = tk.Frame(header_frame)
        btn_frame.pack(side='right')

        tk.Button(btn_frame, text="Copy Selected", command=self._copy_selected,
                  width=12).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Copy All", command=self._copy_all,
                  width=10).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Clear", command=self._clear_history,
                  width=8).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Export", command=self._export_history,
                  width=8).pack(side='left', padx=2)

        # Listbox with scrollbar
        list_frame = tk.Frame(main_frame)
        list_frame.pack(fill='both', expand=True)

        scrollbar_y = tk.Scrollbar(list_frame, orient='vertical')
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = tk.Scrollbar(list_frame, orient='horizontal')
        scrollbar_x.pack(side='bottom', fill='x')

        self.listbox = tk.Listbox(list_frame, font=("Consolas", 10),
                                   yscrollcommand=scrollbar_y.set,
                                   xscrollcommand=scrollbar_x.set,
                                   selectmode=tk.EXTENDED)
        self.listbox.pack(side='left', fill='both', expand=True)

        scrollbar_y.config(command=self.listbox.yview)
        scrollbar_x.config(command=self.listbox.xview)

        # Bind double-click to copy
        self.listbox.bind('<Double-Button-1>', lambda e: self._copy_selected())
        self.listbox.bind('<Control-c>', lambda e: self._copy_selected())

        # Populate with existing history
        self._refresh_list()

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.is_visible = False
        if self.window:
            self.window.destroy()
            self.window = None
            self.listbox = None

    def add_translation(self, text, timestamp=None):
        if not text or not text.strip():
            return
        if timestamp is None:
            timestamp = datetime.now()
        entry = f"[{timestamp:%H:%M:%S}] {text}"
        self.history.append(entry)

        if self.listbox and self.listbox.winfo_exists():
            self.listbox.insert(tk.END, entry)
            self.listbox.see(tk.END)
            self.count_label.config(text=f"Total: {len(self.history)} translations")

    def _refresh_list(self):
        if not self.listbox:
            return
        self.listbox.delete(0, tk.END)
        for entry in self.history:
            self.listbox.insert(tk.END, entry)
        if self.history:
            self.listbox.see(tk.END)

    def _copy_selected(self):
        if not self.listbox:
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        texts = [self.listbox.get(i) for i in selection]
        text = '\n'.join(texts)
        self.parent.clipboard_clear()
        self.parent.clipboard_append(text)
        logger.info(f"Copied {len(selection)} translation(s) to clipboard")

    def _copy_all(self):
        if not self.history:
            return
        text = '\n'.join(self.history)
        self.parent.clipboard_clear()
        self.parent.clipboard_append(text)
        logger.info(f"Copied all {len(self.history)} translations to clipboard")

    def _clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all translation history?"):
            self.history.clear()
            if self.listbox:
                self.listbox.delete(0, tk.END)
                self.count_label.config(text="Total: 0 translations")
            logger.info("Translation history cleared")

    def _export_history(self):
        if not self.history:
            messagebox.showinfo("Export", "No translations to export.")
            return
        filename = f"translations_{datetime.now():%Y%m%d_%H%M%S}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.history))
            messagebox.showinfo("Export", f"Translations exported to:\n{filename}")
            logger.info(f"Exported {len(self.history)} translations to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")


class StatisticsPanel:
    """Panel for displaying real-time statistics"""
    def __init__(self, parent, stats):
        self.parent = parent
        self.stats = stats
        self.window = None
        self.update_job = None

    def show(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return

        self.window = tk.Toplevel(self.parent)
        self.window.title("Translation Statistics")
        self.window.geometry("350x300")
        self.window.resizable(False, False)

        main_frame = tk.Frame(self.window, padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)

        # Title
        tk.Label(main_frame, text="Session Statistics",
                 font=("Helvetica", 14, "bold")).pack(pady=(0, 15))

        # Stats grid
        stats_frame = tk.Frame(main_frame)
        stats_frame.pack(fill='x')

        self.stat_labels = {}
        stat_items = [
            ("Total Chunks Processed:", "chunks"),
            ("Successful Translations:", "successful"),
            ("Filtered (Hallucinations):", "filtered"),
            ("Average Confidence:", "confidence"),
            ("Session Duration:", "duration"),
            ("Translations/Minute:", "rate"),
        ]

        for i, (label_text, key) in enumerate(stat_items):
            tk.Label(stats_frame, text=label_text, font=("Helvetica", 10),
                     anchor='w').grid(row=i, column=0, sticky='w', pady=3)
            self.stat_labels[key] = tk.Label(stats_frame, text="0",
                                              font=("Helvetica", 10, "bold"),
                                              anchor='e', width=15)
            self.stat_labels[key].grid(row=i, column=1, sticky='e', pady=3)

        stats_frame.columnconfigure(1, weight=1)

        # Reset button
        tk.Button(main_frame, text="Reset Statistics", command=self._reset_stats,
                  width=15).pack(pady=(20, 0))

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_stats()

    def _update_stats(self):
        if not self.window or not self.window.winfo_exists():
            return

        try:
            self.stat_labels["chunks"].config(text=str(self.stats.total_chunks))
            self.stat_labels["successful"].config(text=str(self.stats.successful_translations))
            self.stat_labels["filtered"].config(text=str(self.stats.filtered_count))

            avg_conf = self.stats.average_confidence
            self.stat_labels["confidence"].config(text=f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")

            duration = self.stats.session_duration
            mins, secs = divmod(int(duration), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                self.stat_labels["duration"].config(text=f"{hours}h {mins}m {secs}s")
            else:
                self.stat_labels["duration"].config(text=f"{mins}m {secs}s")

            rate = self.stats.translations_per_minute
            self.stat_labels["rate"].config(text=f"{rate:.1f}")

        except Exception as e:
            logger.error(f"Error updating stats: {e}")

        self.update_job = self.window.after(1000, self._update_stats)

    def _reset_stats(self):
        self.stats.reset()
        logger.info("Statistics reset")

    def _on_close(self):
        if self.update_job:
            self.window.after_cancel(self.update_job)
        self.window.destroy()
        self.window = None


class SubtitlePositionControl:
    """Control for positioning subtitle window"""
    def __init__(self, parent, on_position_change):
        self.parent = parent
        self.on_position_change = on_position_change
        self.frame = None

    def create(self, container):
        self.frame = tk.LabelFrame(container, text="Subtitle Position", padx=10, pady=10)

        # Position presets
        preset_frame = tk.Frame(self.frame)
        preset_frame.pack(fill='x', pady=(0, 5))

        tk.Label(preset_frame, text="Quick Position:").pack(side='left')

        positions = [
            ("Top", "top"),
            ("Center", "center"),
            ("Bottom", "bottom"),
        ]

        for text, pos in positions:
            tk.Button(preset_frame, text=text, width=8,
                      command=lambda p=pos: self.on_position_change(p)).pack(side='left', padx=2)

        # Manual offset controls
        offset_frame = tk.Frame(self.frame)
        offset_frame.pack(fill='x', pady=5)

        tk.Label(offset_frame, text="Fine Tune:").pack(side='left')
        tk.Button(offset_frame, text="Up", width=6,
                  command=lambda: self.on_position_change("up")).pack(side='left', padx=2)
        tk.Button(offset_frame, text="Down", width=6,
                  command=lambda: self.on_position_change("down")).pack(side='left', padx=2)
        tk.Button(offset_frame, text="Left", width=6,
                  command=lambda: self.on_position_change("left")).pack(side='left', padx=2)
        tk.Button(offset_frame, text="Right", width=6,
                  command=lambda: self.on_position_change("right")).pack(side='left', padx=2)

        return self.frame


class ControlGUI:
    def __init__(self, root, config, stats, gui_queue):
        self.root = root
        self.config = config
        self.stats = stats
        self.gui_queue = gui_queue

        self.root.title("Live Audio Translator")
        self.root.geometry("650x900")
        self.root.resizable(True, True)
        self.root.minsize(600, 700)

        self.worker_threads = []
        self.stop_event = None
        self.subtitle_window = None
        self.subtitle_label = None
        self.subtitle_shadow_label = None
        self.background_canvas = None
        self.background_rect = None
        self.last_subtitle = ""
        self.subtitle_history = []
        self._drag_data = {"x": 0, "y": 0}
        self.device_list = []

        # New panels
        self.history_panel = TranslationHistoryPanel(self.root)
        self.stats_panel = StatisticsPanel(self.root, self.stats)

        self.log_window = None
        self.log_text_widget = None

        self.log_queue = Queue()
        self.log_buffer = deque(maxlen=1000)
        self.log_file = None
        self._patch_stdout()

        os.makedirs("presets", exist_ok=True)

        self.setup_ui()
        self._start_log_processor()
        self._bind_keyboard_shortcuts()

    def _bind_keyboard_shortcuts(self):
        """Bind global keyboard shortcuts"""
        self.root.bind('<Control-q>', lambda e: self.on_close())
        self.root.bind('<F5>', lambda e: self.start_translator())
        self.root.bind('<F6>', lambda e: self.stop_translator())
        self.root.bind('<Control-h>', lambda e: self.history_panel.show())
        self.root.bind('<Control-l>', lambda e: self.open_log_window())

    def _patch_stdout(self):
        self.log_file = open("translator_app.log", "a", encoding='utf-8', buffering=1)

        class StdoutRedirector(io.TextIOBase):
            def __init__(self, outer):
                self.outer = outer

            def write(self, s):
                if sys.__stdout__ is not None:
                    sys.__stdout__.write(s)
                if self.outer.log_file and not self.outer.log_file.closed:
                    self.outer.log_file.write(s)
                self.outer.log_buffer.append(s)
                self.outer.log_queue.put(s)

            def flush(self):
                if sys.__stdout__ is not None:
                    sys.__stdout__.flush()
                if self.outer.log_file and not self.outer.log_file.closed:
                    self.outer.log_file.flush()

        sys.stdout = StdoutRedirector(self)
        sys.stderr = sys.stdout
        print(f"\n--- Application session started at {datetime.now()} ---")

    def _start_log_processor(self):
        self.root.after(100, self._process_log_queue)

    def _process_log_queue(self):
        messages_to_process = 100
        batch = []
        for _ in range(messages_to_process):
            try:
                message = self.log_queue.get_nowait()
                batch.append(message)
            except Empty:
                break

        if batch and self.log_text_widget and self.log_text_widget.winfo_exists():
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('end', "".join(batch))
            self.log_text_widget.see('end')
            self.log_text_widget.config(state='disabled')

        self.root.after(100, self._process_log_queue)

    def setup_ui(self):
        # Create main scrollable frame
        main_canvas = tk.Canvas(self.root)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = tk.Frame(main_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Header
        header_frame = tk.Frame(self.scrollable_frame)
        header_frame.pack(pady=10, padx=20, fill='x')

        title_label = tk.Label(header_frame, text="Live Audio Translator",
                               font=("Helvetica", 18, "bold"))
        title_label.pack()

        subtitle_info = tk.Label(header_frame,
                                  text="Real-time Japanese to English translation for VTuber streams",
                                  font=("Helvetica", 9), fg="grey")
        subtitle_info.pack()

        # Audio device selection
        device_frame = tk.LabelFrame(self.scrollable_frame, text="Audio Device", padx=10, pady=10)
        device_frame.pack(pady=5, padx=20, fill='x')

        device_row = tk.Frame(device_frame)
        device_row.pack(fill='x')

        self.device_var = tk.StringVar()
        self.device_menu = tk.OptionMenu(device_row, self.device_var, "Loading...")
        self.device_menu.pack(side="left", padx=5, expand=True, fill='x')

        tk.Button(device_row, text="Refresh", command=self.refresh_devices,
                  width=8).pack(side="right", padx=5)

        self.refresh_devices()
        self.device_var.trace_add('write', self.on_device_select)

        # Status and control buttons
        status_frame = tk.Frame(self.scrollable_frame)
        status_frame.pack(pady=10, padx=20, fill='x')

        self.status_label = tk.Label(status_frame, text="Status: Ready",
                                      font=("Helvetica", 11, "bold"), fg="green")
        self.status_label.pack()

        button_frame = tk.Frame(self.scrollable_frame)
        button_frame.pack(pady=5, padx=20)

        self.download_button = tk.Button(button_frame, text="Download Model",
                                         command=self.download_model, bg="#007bff",
                                         fg="white", font=("Helvetica", 11), width=14, height=2)
        self.download_button.pack(side="left", padx=5)

        self.start_button = tk.Button(button_frame, text="Start (F5)",
                                       command=self.start_translator, bg="#28a745",
                                       fg="white", font=("Helvetica", 11), width=12, height=2)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop (F6)",
                                      command=self.stop_translator, bg="#dc3545",
                                      fg="white", font=("Helvetica", 11), width=12, height=2,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=5)

        # Quick action buttons
        quick_frame = tk.Frame(self.scrollable_frame)
        quick_frame.pack(pady=5, padx=20)

        tk.Button(quick_frame, text="History (Ctrl+H)", command=self.history_panel.show,
                  width=14).pack(side="left", padx=3)
        tk.Button(quick_frame, text="Statistics", command=self.stats_panel.show,
                  width=12).pack(side="left", padx=3)
        tk.Button(quick_frame, text="Log (Ctrl+L)", command=self.open_log_window,
                  width=12).pack(side="left", padx=3)

        # Settings container with notebook tabs
        settings_notebook = ttk.Notebook(self.scrollable_frame)
        settings_notebook.pack(pady=10, padx=20, fill='x')

        # Tab 1: Audio Settings
        audio_tab = tk.Frame(settings_notebook, padx=10, pady=10)
        settings_notebook.add(audio_tab, text="Audio")

        # Dynamic chunking settings
        dynamic_frame = tk.LabelFrame(audio_tab, text="Dynamic Chunking (Recommended)", padx=10, pady=10)
        dynamic_frame.pack(pady=5, fill="x")

        self.dynamic_chunk_var = tk.BooleanVar(value=self.config.use_dynamic_chunking)
        self.dynamic_chunk_check = tk.Checkbutton(dynamic_frame,
                                                   text="Enable Dynamic Chunks",
                                                   variable=self.dynamic_chunk_var,
                                                   font=("Helvetica", 10))
        self.dynamic_chunk_check.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))

        tk.Label(dynamic_frame, text="Silence Timeout (s):").grid(row=1, column=0, sticky="w", pady=2)
        self.dyn_silence_var = tk.StringVar(value=str(self.config.dynamic_silence_timeout))
        self.dyn_silence_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_silence_var, width=8)
        self.dyn_silence_entry.grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(dynamic_frame, text="Max Duration (s):").grid(row=1, column=2, sticky="w", pady=2, padx=(10,0))
        self.dyn_max_dur_var = tk.StringVar(value=str(self.config.dynamic_max_chunk_duration))
        self.dyn_max_dur_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_max_dur_var, width=8)
        self.dyn_max_dur_entry.grid(row=1, column=3, padx=5, sticky="w")

        tk.Label(dynamic_frame, text="Min Speech (s):").grid(row=2, column=0, sticky="w", pady=2)
        self.dyn_min_speech_var = tk.StringVar(value=str(self.config.dynamic_min_speech_duration))
        self.dyn_min_speech_entry = tk.Entry(dynamic_frame, textvariable=self.dyn_min_speech_var, width=8)
        self.dyn_min_speech_entry.grid(row=2, column=1, padx=5, sticky="w")

        # VAD settings
        vad_frame = tk.LabelFrame(audio_tab, text="Voice Activity Detection", padx=10, pady=10)
        vad_frame.pack(pady=5, fill="x")

        self.vad_var = tk.BooleanVar(value=self.config.use_vad_filter)
        self.vad_check = tk.Checkbutton(vad_frame, text="Enable VAD Filter",
                                         variable=self.vad_var, font=("Helvetica", 10))
        self.vad_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        tk.Label(vad_frame, text="Volume Threshold:").grid(row=1, column=0, sticky="w", pady=2)
        self.volume_var = tk.StringVar(value=str(self.config.volume_threshold))
        self.volume_entry = tk.Entry(vad_frame, textvariable=self.volume_var, width=8)
        self.volume_entry.grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(vad_frame, text="VAD Threshold (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.vad_threshold_var = tk.StringVar(value=str(int(self.config.vad_threshold * 100)))
        self.vad_threshold_entry = tk.Entry(vad_frame, textvariable=self.vad_threshold_var, width=8)
        self.vad_threshold_entry.grid(row=2, column=1, padx=5, sticky="w")

        # Tab 2: Appearance Settings
        appearance_tab = tk.Frame(settings_notebook, padx=10, pady=10)
        settings_notebook.add(appearance_tab, text="Appearance")

        # Font settings
        font_frame = tk.LabelFrame(appearance_tab, text="Font Settings", padx=10, pady=10)
        font_frame.pack(pady=5, fill="x")

        tk.Label(font_frame, text="Font Size:").grid(row=0, column=0, sticky="w", pady=2)
        self.font_var = tk.StringVar(value=str(self.config.font_size))
        self.font_entry = tk.Entry(font_frame, textvariable=self.font_var, width=8)
        self.font_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.font_entry.bind('<KeyRelease>', self.update_subtitle_style)

        tk.Label(font_frame, text="Font Weight:").grid(row=0, column=2, sticky="w", pady=2, padx=(10, 0))
        self.font_weight_var = tk.StringVar(value=self.config.font_weight)
        self.font_weight_menu = tk.OptionMenu(font_frame, self.font_weight_var,
                                               'normal', 'bold',
                                               command=self.on_font_weight_change)
        self.font_weight_menu.grid(row=0, column=3, padx=5, sticky="w")

        tk.Label(font_frame, text="Font Color:").grid(row=1, column=0, sticky="w", pady=2)
        self.font_color_var = tk.StringVar(value=self.config.subtitle_font_color)
        self.font_color_btn = tk.Button(font_frame, text="Pick", command=self.pick_font_color, width=6)
        self.font_color_btn.grid(row=1, column=1, padx=5, sticky="w")
        self.font_color_display = tk.Label(font_frame, text='    ',
                                            bg=self.config.subtitle_font_color,
                                            relief="solid", borderwidth=1)
        self.font_color_display.grid(row=1, column=2, padx=5, sticky="w")

        self.text_shadow_var = tk.BooleanVar(value=getattr(self.config, 'text_shadow', True))
        self.text_shadow_check = tk.Checkbutton(font_frame, text="Text Shadow",
                                                 variable=self.text_shadow_var,
                                                 command=self.on_text_shadow_change)
        self.text_shadow_check.grid(row=1, column=3, sticky="w", pady=2)

        # Background settings
        bg_frame = tk.LabelFrame(appearance_tab, text="Background Settings", padx=10, pady=10)
        bg_frame.pack(pady=5, fill="x")

        tk.Label(bg_frame, text="BG Mode:").grid(row=0, column=0, sticky="w", pady=2)
        self.bg_mode_var = tk.StringVar(value=self.config.subtitle_bg_mode)
        self.bg_mode_menu = tk.OptionMenu(bg_frame, self.bg_mode_var,
                                           'transparent', 'solid',
                                           command=self.set_bg_mode)
        self.bg_mode_menu.grid(row=0, column=1, padx=5, sticky="w")

        tk.Label(bg_frame, text="Opacity (%):").grid(row=0, column=2, sticky="w", pady=2, padx=(10,0))
        self.opacity_var = tk.StringVar(value=str(int(self.config.window_opacity * 100)))
        self.opacity_entry = tk.Entry(bg_frame, textvariable=self.opacity_var, width=8)
        self.opacity_entry.grid(row=0, column=3, padx=5, sticky="w")
        self.opacity_entry.bind('<KeyRelease>', self.on_opacity_change)

        tk.Label(bg_frame, text="BG Color:").grid(row=1, column=0, sticky="w", pady=2)
        self.bg_color_var = tk.StringVar(value=self.config.subtitle_bg_color)
        self.bg_color_btn = tk.Button(bg_frame, text="Pick", command=self.pick_bg_color, width=6)
        self.bg_color_btn.grid(row=1, column=1, padx=5, sticky="w")
        self.bg_color_display = tk.Label(bg_frame, text='    ',
                                          bg=self.config.subtitle_bg_color,
                                          relief="solid", borderwidth=1)
        self.bg_color_display.grid(row=1, column=2, padx=5, sticky="w")

        # Subtitle position controls
        self.position_control = SubtitlePositionControl(self.root, self.on_subtitle_position_change)
        position_frame = self.position_control.create(appearance_tab)
        position_frame.pack(pady=5, fill="x")

        # Tab 3: Presets
        presets_tab = tk.Frame(settings_notebook, padx=10, pady=10)
        settings_notebook.add(presets_tab, text="Presets")

        # Load preset
        load_frame = tk.LabelFrame(presets_tab, text="Load Preset", padx=10, pady=10)
        load_frame.pack(pady=5, fill="x")

        self.preset_var = tk.StringVar()
        self.preset_menu = tk.OptionMenu(load_frame, self.preset_var, "No presets found")
        self.preset_menu.pack(side="left", padx=5, expand=True, fill='x')
        self.load_preset_button = tk.Button(load_frame, text="Load", command=self.load_preset, width=8)
        self.load_preset_button.pack(side="left", padx=5)

        # Save preset
        save_frame = tk.LabelFrame(presets_tab, text="Save Preset", padx=10, pady=10)
        save_frame.pack(pady=5, fill="x")

        self.save_preset_name_var = tk.StringVar()
        self.save_preset_entry = tk.Entry(save_frame, textvariable=self.save_preset_name_var)
        self.save_preset_entry.pack(side="left", padx=5, expand=True, fill='x')
        self.save_preset_button = tk.Button(save_frame, text="Save", command=self.save_preset, width=8)
        self.save_preset_button.pack(side="left", padx=5)

        # Preset management
        mgmt_frame = tk.Frame(presets_tab)
        mgmt_frame.pack(pady=10, fill='x')

        tk.Button(mgmt_frame, text="Refresh List", command=self.refresh_preset_list,
                  width=12).pack(side="left", padx=5)
        tk.Button(mgmt_frame, text="Reset to Defaults", command=self.reset_to_defaults,
                  width=14).pack(side="left", padx=5)

        self.refresh_preset_list()

        # Keyboard shortcuts info
        shortcuts_frame = tk.LabelFrame(self.scrollable_frame, text="Keyboard Shortcuts", padx=10, pady=5)
        shortcuts_frame.pack(pady=10, padx=20, fill='x')

        shortcuts_text = (
            "F5: Start  |  F6: Stop  |  Ctrl+H: History  |  Ctrl+L: Log  |  Ctrl+Q: Quit\n"
            "Subtitle Window: Drag to move  |  Ctrl+C: Copy  |  Ctrl+S: Save  |  Esc: Stop"
        )
        tk.Label(shortcuts_frame, text=shortcuts_text, font=("Consolas", 9),
                 justify="center").pack(pady=5)

    def on_subtitle_position_change(self, position):
        """Handle subtitle window position changes"""
        if not self.subtitle_window or not self.subtitle_window.winfo_exists():
            return

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = self.subtitle_window.winfo_width()
        win_height = self.subtitle_window.winfo_height()

        current_x = self.subtitle_window.winfo_x()
        current_y = self.subtitle_window.winfo_y()

        move_step = 50  # pixels for fine-tune movement

        if position == "top":
            new_x = (screen_width - win_width) // 2
            new_y = 50
        elif position == "center":
            new_x = (screen_width - win_width) // 2
            new_y = (screen_height - win_height) // 2
        elif position == "bottom":
            new_x = (screen_width - win_width) // 2
            new_y = screen_height - win_height - 100
        elif position == "up":
            new_x = current_x
            new_y = max(0, current_y - move_step)
        elif position == "down":
            new_x = current_x
            new_y = min(screen_height - win_height, current_y + move_step)
        elif position == "left":
            new_x = max(0, current_x - move_step)
            new_y = current_y
        elif position == "right":
            new_x = min(screen_width - win_width, current_x + move_step)
            new_y = current_y
        else:
            return

        self.subtitle_window.geometry(f"+{new_x}+{new_y}")

    def reset_to_defaults(self):
        """Reset all settings to default values"""
        if not messagebox.askyesno("Reset Settings",
                                    "Are you sure you want to reset all settings to defaults?"):
            return

        # Reset config to defaults
        from modules.config import (VOLUME_THRESHOLD, USE_VAD_FILTER, VAD_THRESHOLD,
                                    DEFAULT_BG_COLOR, DEFAULT_FONT_COLOR,
                                    DEFAULT_BG_MODE, DEFAULT_WINDOW_OPACITY)

        self.config.volume_threshold = VOLUME_THRESHOLD
        self.config.use_vad_filter = USE_VAD_FILTER
        self.config.vad_threshold = VAD_THRESHOLD
        self.config.subtitle_bg_color = DEFAULT_BG_COLOR
        self.config.subtitle_font_color = DEFAULT_FONT_COLOR
        self.config.subtitle_bg_mode = DEFAULT_BG_MODE
        self.config.window_opacity = DEFAULT_WINDOW_OPACITY
        self.config.font_size = 24
        self.config.font_weight = "bold"
        self.config.text_shadow = True
        self.config.use_dynamic_chunking = True
        self.config.dynamic_silence_timeout = 1.2
        self.config.dynamic_max_chunk_duration = 15.0
        self.config.dynamic_min_speech_duration = 0.3

        # Update UI
        self.volume_var.set(str(self.config.volume_threshold))
        self.vad_var.set(self.config.use_vad_filter)
        self.vad_threshold_var.set(str(int(self.config.vad_threshold * 100)))
        self.font_var.set(str(self.config.font_size))
        self.font_weight_var.set(self.config.font_weight)
        self.opacity_var.set(str(int(self.config.window_opacity * 100)))
        self.bg_mode_var.set(self.config.subtitle_bg_mode)
        self.bg_color_display.config(bg=self.config.subtitle_bg_color)
        self.font_color_display.config(bg=self.config.subtitle_font_color)
        self.text_shadow_var.set(self.config.text_shadow)
        self.dynamic_chunk_var.set(self.config.use_dynamic_chunking)
        self.dyn_silence_var.set(str(self.config.dynamic_silence_timeout))
        self.dyn_max_dur_var.set(str(self.config.dynamic_max_chunk_duration))
        self.dyn_min_speech_var.set(str(self.config.dynamic_min_speech_duration))

        self.update_subtitle_style()
        self.config.save_config()
        logger.info("Settings reset to defaults")
        messagebox.showinfo("Reset", "All settings have been reset to defaults.")

    def refresh_devices(self):
        self.device_list = sc.all_microphones(include_loopback=True)
        device_names = [mic.name for mic in self.device_list]
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        if not device_names:
            menu.add_command(label="No devices found", state="disabled")
            self.device_var.set("No devices found")
        else:
            for name in device_names:
                menu.add_command(label=name, command=lambda v=name: self.device_var.set(v))

            if self.config.selected_audio_device and self.config.selected_audio_device in device_names:
                self.device_var.set(self.config.selected_audio_device)
            else:
                preferred_device = find_audio_device()
                if preferred_device:
                    self.device_var.set(preferred_device.name)
                elif device_names:
                    self.device_var.set(device_names[0])
                else:
                    self.device_var.set("No devices found")

    def get_selected_device_name(self):
        selected_name = self.device_var.get()
        return selected_name if selected_name != "No devices found" else None

    def stop_translator(self, event=None):
        if self.worker_threads:
            logger.info("Stopping translator...")
            if self.stop_event:
                self.stop_event.set()
            for t in self.worker_threads:
                t.join(timeout=1.0)
        self.worker_threads = []
        self.destroy_subtitle_window()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped", fg="red")
        logger.info("Translator stopped.")

    def check_gui_queue(self):
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "subtitle":
                    self.update_subtitle_text(data)
                elif msg_type == "model_loaded":
                    self.status_label.config(text="Status: Running", fg="green")
                    self.stop_button.config(state="normal")
                elif msg_type == "error":
                    self.status_label.config(text="Status: Error!", fg="red")
                    if self.subtitle_label:
                        self.update_subtitle_text(f"ERROR: {data}")
                    self.stop_translator()
                    return
        except Empty:
            pass
        finally:
            if self.worker_threads:
                self.root.after(100, self.check_gui_queue)

    def create_subtitle_window(self):
        if self.subtitle_window:
            return
        self.subtitle_window = tk.Toplevel(self.root)
        self.subtitle_window.overrideredirect(True)

        # Position at bottom center of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = 1000
        win_height = 200
        x = (screen_width - win_width) // 2
        y = screen_height - win_height - 100

        self.subtitle_window.geometry(f"{win_width}x{win_height}+{x}+{y}")
        self.subtitle_window.wm_attributes("-topmost", True)
        self.subtitle_window.config(bg='green')
        self.subtitle_window.wm_attributes("-transparentcolor", "green")

        self.background_canvas = tk.Canvas(self.subtitle_window, bg='green', highlightthickness=0)
        self.background_canvas.pack(pady=20, padx=20, expand=True, fill="both")

        self.background_rect = self.background_canvas.create_rectangle(0, 0, 0, 0, outline="", width=0)

        self.subtitle_shadow_label = tk.Label(self.background_canvas, text="",
                                               wraplength=900, justify="center")
        self.subtitle_label = tk.Label(self.background_canvas, text="Waiting for audio...",
                                        wraplength=900, justify="center")

        self.update_subtitle_style()

        # Bindings
        self.subtitle_window.bind("<Escape>", self.stop_translator)
        for widget in [self.subtitle_label, self.subtitle_shadow_label, self.background_canvas]:
            widget.bind("<ButtonPress-1>", self.start_drag)
            widget.bind("<ButtonRelease-1>", self.stop_drag)
            widget.bind("<B1-Motion>", self.do_drag)
            widget.bind("<Control-c>", self.copy_subtitle)
            widget.bind("<Control-s>", self.save_subtitle_history)

    def destroy_subtitle_window(self):
        if self.subtitle_window:
            try:
                self.subtitle_window.destroy()
            except tk.TclError:
                pass
            self.subtitle_window = None

    def update_subtitle_text(self, text):
        if not self.subtitle_label or not self.subtitle_label.winfo_exists():
            return
        if text != self.last_subtitle:
            self.last_subtitle = text
            display_text = text or "..."
            try:
                self.subtitle_label.config(text=display_text)
                if self.subtitle_shadow_label:
                    self.subtitle_shadow_label.config(text=display_text)
            except tk.TclError:
                return

            if text.strip() and "ERROR" not in text:
                timestamp = datetime.now()
                self.subtitle_history.append(f"[{timestamp:%H:%M:%S}] {text}")
                self.history_panel.add_translation(text, timestamp)

            self._update_background_size()
            self._resize_window_if_needed()

    def _update_background_size(self):
        if not self.subtitle_window or not self.background_canvas.winfo_exists():
            return
        try:
            self.subtitle_window.update_idletasks()

            label_width = self.subtitle_label.winfo_reqwidth()
            label_height = self.subtitle_label.winfo_reqheight()

            min_width = 200
            min_height = 50
            label_width = max(label_width, min_width)
            label_height = max(label_height, min_height)

            canvas_width = self.background_canvas.winfo_width()
            canvas_height = self.background_canvas.winfo_height()

            padding_x = max(25, min(40, label_width * 0.1))
            padding_y = max(15, min(25, label_height * 0.15))

            x0 = (canvas_width - label_width) / 2 - padding_x
            y0 = (canvas_height - label_height) / 2 - padding_y
            x1 = (canvas_width + label_width) / 2 + padding_x
            y1 = (canvas_height + label_height) / 2 + padding_y

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(canvas_width, x1)
            y1 = min(canvas_height, y1)

            self.background_canvas.coords(self.background_rect, x0, y0, x1, y1)

            self.subtitle_label.place(relx=0.5, rely=0.5, anchor="center")

            if self.config.text_shadow and self.subtitle_shadow_label.winfo_exists():
                shadow_offset = 2
                self.subtitle_shadow_label.place(
                    x=self.subtitle_label.winfo_x() + shadow_offset,
                    y=self.subtitle_label.winfo_y() + shadow_offset
                )
                self.background_canvas.tag_lower(self.background_rect)
                self.subtitle_shadow_label.lift()
                self.subtitle_label.lift()
            elif self.subtitle_shadow_label.winfo_exists():
                self.subtitle_shadow_label.place_forget()

        except tk.TclError as e:
            logger.debug(f"Error updating background size: {e}")

    def _resize_window_if_needed(self):
        if not self.subtitle_window or not self.subtitle_label.winfo_exists():
            return

        try:
            label_width = self.subtitle_label.winfo_reqwidth()
            label_height = self.subtitle_label.winfo_reqheight()

            padding_x = 60
            padding_y = 60

            required_width = max(800, label_width + padding_x)
            required_height = max(150, label_height + padding_y)

            current_width = self.subtitle_window.winfo_width()
            current_height = self.subtitle_window.winfo_height()

            width_diff = abs(required_width - current_width)
            height_diff = abs(required_height - current_height)

            if width_diff > 50 or height_diff > 30:
                x = self.subtitle_window.winfo_x()
                y = self.subtitle_window.winfo_y()

                self.subtitle_window.geometry(f"{required_width}x{required_height}+{x}+{y}")
                self.background_canvas.configure(width=required_width-40, height=required_height-40)

                new_wraplength = max(400, required_width - 80)
                self.subtitle_label.configure(wraplength=new_wraplength)
                if self.subtitle_shadow_label:
                    self.subtitle_shadow_label.configure(wraplength=new_wraplength)

                self.subtitle_window.update_idletasks()
                self._update_background_size()

        except tk.TclError as e:
            logger.debug(f"Error resizing window: {e}")

    def start_drag(self, event):
        self._drag_data["x"], self._drag_data["y"] = event.x, event.y

    def stop_drag(self, event):
        self._drag_data["x"], self._drag_data["y"] = 0, 0

    def do_drag(self, event):
        if self.subtitle_window:
            x = self.subtitle_window.winfo_pointerx() - self._drag_data["x"]
            y = self.subtitle_window.winfo_pointery() - self._drag_data["y"]
            self.subtitle_window.geometry(f"+{x}+{y}")

    def copy_subtitle(self, event=None):
        if self.last_subtitle:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_subtitle)
            logger.info(f"Copied subtitle to clipboard")

    def save_subtitle_history(self, event=None):
        if self.subtitle_history:
            filename = f"subtitles_{datetime.now():%Y%m%d_%H%M%S}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.subtitle_history))
                logger.info(f"Saved history to: {filename}")
                messagebox.showinfo("Saved", f"Translation history saved to:\n{filename}")
            except Exception as e:
                logger.error(f"Error saving history: {e}")
                messagebox.showerror("Error", f"Failed to save: {e}")

    def on_close(self):
        logger.info("Closing application...")

        if self.stop_event:
            self.stop_event.set()

        self.apply_and_save_settings()

        logger.info("--- Application session ended ---")
        if hasattr(sys, '__stdout__'):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        if self.log_file and not self.log_file.closed:
            self.log_file.close()
            self.log_file = None

        self.root.destroy()

    def open_log_window(self):
        if self.log_window and tk.Toplevel.winfo_exists(self.log_window):
            self.log_window.lift()
            return
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Application Log")
        self.log_window.geometry("800x500")

        # Toolbar
        toolbar = tk.Frame(self.log_window)
        toolbar.pack(fill='x', padx=5, pady=5)

        tk.Button(toolbar, text="Clear Log", command=self._clear_log_display,
                  width=10).pack(side='left', padx=2)
        tk.Button(toolbar, text="Copy All", command=self._copy_log,
                  width=10).pack(side='left', padx=2)

        # Text widget with scrollbar
        text_frame = tk.Frame(self.log_window)
        text_frame.pack(expand=True, fill='both', padx=5, pady=5)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')

        self.log_text_widget = tk.Text(text_frame, wrap='word', font=("Consolas", 10),
                                        state='disabled', yscrollcommand=scrollbar.set)
        self.log_text_widget.pack(expand=True, fill='both')
        scrollbar.config(command=self.log_text_widget.yview)

        if self.log_buffer:
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('1.0', "".join(self.log_buffer))
            self.log_text_widget.see('end')
            self.log_text_widget.config(state='disabled')

        self.log_window.protocol("WM_DELETE_WINDOW", self._on_log_close)

    def _clear_log_display(self):
        if self.log_text_widget:
            self.log_text_widget.config(state='normal')
            self.log_text_widget.delete('1.0', tk.END)
            self.log_text_widget.config(state='disabled')

    def _copy_log(self):
        if self.log_text_widget:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.log_text_widget.get('1.0', tk.END))
            logger.info("Log copied to clipboard")

    def _on_log_close(self):
        if self.log_window:
            self.log_window.destroy()
            self.log_window = None
            self.log_text_widget = None

    def pick_bg_color(self):
        color = colorchooser.askcolor(title="Pick Background Color",
                                       initialcolor=self.config.subtitle_bg_color)
        if color and color[1]:
            self.config.subtitle_bg_color = color[1]
            self.bg_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def pick_font_color(self):
        color = colorchooser.askcolor(title="Pick Font Color",
                                       initialcolor=self.config.subtitle_font_color)
        if color and color[1]:
            self.config.subtitle_font_color = color[1]
            self.font_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def apply_and_save_settings(self, save_to_disk=True):
        try:
            self.config.volume_threshold = max(0.0, float(self.volume_var.get()))
            self.config.use_vad_filter = self.vad_var.get()
            self.config.vad_threshold = max(0.0, min(1.0, float(self.vad_threshold_var.get()) / 100.0))

            self.config.use_dynamic_chunking = self.dynamic_chunk_var.get()
            self.config.dynamic_silence_timeout = max(0.1, float(self.dyn_silence_var.get()))
            self.config.dynamic_max_chunk_duration = max(1.0, float(self.dyn_max_dur_var.get()))
            self.config.dynamic_min_speech_duration = max(0.1, float(self.dyn_min_speech_var.get()))

            self.config.font_size = int(self.font_var.get())
            self.config.window_opacity = max(0.0, min(1.0, float(self.opacity_var.get()) / 100.0))
            self.config.font_weight = self.font_weight_var.get()
            self.config.text_shadow = self.text_shadow_var.get()
            self.config.subtitle_bg_mode = self.bg_mode_var.get()

            self.config.selected_audio_device = self.device_var.get()
            if save_to_disk:
                self.config.save_config()
                logger.info("Settings applied and saved.")
            else:
                logger.info("Settings applied to current session.")
            return True
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Invalid Input",
                                 f"Please ensure all numeric fields are valid numbers.\nError: {e}")
            return False

    def start_translator(self, event=None):
        if self.worker_threads:
            return
        if not self.apply_and_save_settings():
            return
        self.stats.reset()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Loading model(s)...", fg="orange")
        self.root.update_idletasks()

        self.create_subtitle_window()
        self.stop_event = threading.Event()
        audio_queue = Queue(maxsize=20)
        selected_device_name = self.get_selected_device_name()
        if selected_device_name is None:
            messagebox.showerror("Audio Error", "Could not find a valid audio device.")
            self.status_label.config(text="Status: Error!", fg="red")
            self.start_button.config(state="normal")
            return

        recorder = threading.Thread(target=recorder_thread,
                                     args=(self.stop_event, audio_queue, self.config,
                                           self.gui_queue, selected_device_name),
                                     daemon=True)
        processor = threading.Thread(target=processor_thread,
                                       args=(self.stop_event, audio_queue, self.config,
                                             self.stats, self.gui_queue),
                                       daemon=True)
        self.worker_threads = [recorder, processor]
        for t in self.worker_threads:
            t.start()
        self.check_gui_queue()

    def on_opacity_change(self, event=None):
        try:
            self.config.window_opacity = max(0.0, min(1.0, float(self.opacity_var.get()) / 100.0))
        except (ValueError, tk.TclError):
            pass
        self.update_subtitle_style()

    def on_font_weight_change(self, value=None):
        self.config.font_weight = self.font_weight_var.get()
        self.update_subtitle_style()

    def on_text_shadow_change(self):
        self.config.text_shadow = self.text_shadow_var.get()
        self.update_subtitle_style()

    def set_bg_mode(self, value=None):
        self.config.subtitle_bg_mode = self.bg_mode_var.get()
        self.update_subtitle_style()

    def update_subtitle_style(self, event=None):
        if not self.subtitle_window or not self.subtitle_label.winfo_exists():
            return
        try:
            font_size = int(self.font_var.get())
            font_weight = self.font_weight_var.get()
            font_tuple = ("Helvetica", font_size, font_weight)

            self.subtitle_label.config(font=font_tuple, fg=self.config.subtitle_font_color,
                                        bg=self.config.subtitle_bg_color)
            if self.subtitle_shadow_label and self.subtitle_shadow_label.winfo_exists():
                self.subtitle_shadow_label.config(font=font_tuple, fg='#1c1c1c',
                                                   bg=self.config.subtitle_bg_color)
            if self.background_canvas and self.background_rect:
                self.background_canvas.itemconfig(self.background_rect,
                                                   fill=self.config.subtitle_bg_color,
                                                   outline=self.config.border_color,
                                                   width=self.config.border_width)
            if self.config.subtitle_bg_mode == 'transparent':
                self.subtitle_window.wm_attributes("-alpha", self.config.window_opacity)
            else:
                self.subtitle_window.wm_attributes("-alpha", 1.0)
            self._update_background_size()
        except (ValueError, tk.TclError):
            pass

    def on_device_select(self, *args):
        self.config.selected_audio_device = self.device_var.get()

    def download_model(self):
        self.status_label.config(text="Status: Downloading model...", fg="blue")
        self.download_button.config(state="disabled")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.root.update_idletasks()

        def do_download():
            try:
                logger.info("Starting model download...")
                ensure_model_downloaded(MODEL_ID, self.config.model_cache_dir)
                logger.info("Model downloads/verifications complete.")
                self.gui_queue.put(("status_update", ("Status: All models are ready!", "green")))
            except Exception as e:
                error_msg = f"Download failed: {e}"
                logger.error(error_msg)
                traceback.print_exc()
                self.gui_queue.put(("status_update", (f"Status: {error_msg}", "red")))
            finally:
                self.gui_queue.put(("download_finished", None))

        def process_download_queue():
            try:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "status_update":
                    text, color = data
                    self.status_label.config(text=text, fg=color)
                elif msg_type == "download_finished":
                    self.download_button.config(state="normal")
                    self.start_button.config(state="normal")
                    if not self.worker_threads:
                        self.stop_button.config(state="disabled")
                    return
            except Empty:
                pass
            self.root.after(100, process_download_queue)

        threading.Thread(target=do_download, daemon=True).start()
        process_download_queue()

    def refresh_preset_list(self):
        preset_dir = "presets"
        if not os.path.exists(preset_dir):
            os.makedirs(preset_dir, exist_ok=True)

        preset_files = [f for f in os.listdir(preset_dir) if f.endswith('.json')]
        presets = [os.path.splitext(f)[0] for f in preset_files]

        menu = self.preset_menu["menu"]
        menu.delete(0, "end")

        if not presets:
            menu.add_command(label="No presets found", state="disabled")
            self.preset_var.set("No presets found")
        else:
            for preset_name in sorted(presets):
                menu.add_command(label=preset_name,
                                 command=lambda v=preset_name: self.preset_var.set(v))
            self.preset_var.set(presets[0])

    def save_preset(self):
        preset_name = self.save_preset_name_var.get().strip()
        if not preset_name:
            messagebox.showwarning("Warning", "Please enter a name for the preset.")
            return

        if not self.apply_and_save_settings(save_to_disk=False):
            messagebox.showerror("Error", "Could not save preset due to invalid settings.")
            return

        preset_data = {
            "volume_threshold": self.config.volume_threshold,
            "chunk_duration": self.config.chunk_duration,
            "language_code": self.config.language_code,
            "window_opacity": self.config.window_opacity,
            "font_size": self.config.font_size,
            "use_vad_filter": self.config.use_vad_filter,
            "vad_threshold": self.config.vad_threshold,
            "subtitle_bg_color": self.config.subtitle_bg_color,
            "subtitle_font_color": self.config.subtitle_font_color,
            "subtitle_bg_mode": self.config.subtitle_bg_mode,
            "font_weight": self.config.font_weight,
            "text_shadow": self.config.text_shadow,
            "border_width": self.config.border_width,
            "border_color": self.config.border_color,
            "output_mode": self.config.output_mode,
            "use_dynamic_chunking": self.config.use_dynamic_chunking,
            "dynamic_max_chunk_duration": self.config.dynamic_max_chunk_duration,
            "dynamic_silence_timeout": self.config.dynamic_silence_timeout,
            "dynamic_min_speech_duration": self.config.dynamic_min_speech_duration
        }

        preset_dir = "presets"
        os.makedirs(preset_dir, exist_ok=True)

        file_path = os.path.join(preset_dir, f"{preset_name}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=4)
            messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully.")
            self.refresh_preset_list()
            self.save_preset_name_var.set("")
            logger.info(f"Preset '{preset_name}' saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {e}")

    def load_preset(self):
        preset_name = self.preset_var.get()
        if not preset_name or preset_name == "No presets found":
            messagebox.showwarning("Warning", "No preset selected.")
            return

        file_path = os.path.join("presets", f"{preset_name}.json")
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Preset file not found: {file_path}")
            self.refresh_preset_list()
            return

        try:
            with open(file_path, 'r') as f:
                preset_data = json.load(f)

            for key, value in preset_data.items():
                setattr(self.config, key, value)

            self.volume_var.set(str(self.config.volume_threshold))
            self.opacity_var.set(str(int(self.config.window_opacity * 100)))
            self.font_var.set(str(self.config.font_size))
            self.font_weight_var.set(self.config.font_weight)
            self.vad_var.set(self.config.use_vad_filter)
            self.vad_threshold_var.set(str(int(self.config.vad_threshold * 100)))
            self.bg_mode_var.set(self.config.subtitle_bg_mode)
            self.bg_color_display.config(bg=self.config.subtitle_bg_color)
            self.font_color_display.config(bg=self.config.subtitle_font_color)
            self.text_shadow_var.set(self.config.text_shadow)

            if 'use_dynamic_chunking' in preset_data:
                self.dynamic_chunk_var.set(self.config.use_dynamic_chunking)
                self.dyn_silence_var.set(str(self.config.dynamic_silence_timeout))
                self.dyn_max_dur_var.set(str(self.config.dynamic_max_chunk_duration))
                self.dyn_min_speech_var.set(str(self.config.dynamic_min_speech_duration))

            self.update_subtitle_style()

            messagebox.showinfo("Success", f"Preset '{preset_name}' loaded.")
            logger.info(f"Preset '{preset_name}' loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")
