"""
Main GUI window for Live Translator - Enhanced Version with Speaker Diarization
"""
import time
import threading
from queue import Queue, Empty
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, Label, colorchooser, simpledialog
from datetime import datetime
import json
import os
import traceback
import sys
import io
import logging
from collections import deque

# Import our modular components
from modules.config import Config, SPEAKER_COLORS
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
    """Separate panel for displaying translation history with speaker info"""
    def __init__(self, parent, on_copy_callback=None):
        self.parent = parent
        self.on_copy = on_copy_callback
        self.history = []  # List of (timestamp, speaker, text, color) tuples
        self.window = None
        self.text_widget = None
        self.is_visible = False

    def show(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            self.window.focus_force()
            return

        self.window = tk.Toplevel(self.parent)
        self.window.title("Translation History")
        self.window.geometry("600x450")
        self.window.minsize(500, 350)
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

        tk.Button(btn_frame, text="Copy All", command=self._copy_all,
                  width=10).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Clear", command=self._clear_history,
                  width=8).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Export", command=self._export_history,
                  width=8).pack(side='left', padx=2)

        # Text widget with scrollbar (supports colors)
        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill='both', expand=True)

        scrollbar_y = tk.Scrollbar(text_frame, orient='vertical')
        scrollbar_y.pack(side='right', fill='y')

        self.text_widget = tk.Text(text_frame, font=("Consolas", 10),
                                    yscrollcommand=scrollbar_y.set,
                                    state='disabled', wrap='word',
                                    bg='#1e1e1e', fg='#FFFFFF')  # Dark background, white text
        self.text_widget.pack(side='left', fill='both', expand=True)
        scrollbar_y.config(command=self.text_widget.yview)

        # Configure tags for speaker colors
        for i, color in enumerate(SPEAKER_COLORS):
            self.text_widget.tag_configure(f"speaker_{i+1}", foreground=color, font=("Consolas", 10, "bold"))
        self.text_widget.tag_configure("timestamp", foreground="#888888")
        self.text_widget.tag_configure("text", foreground="#E0E0E0")  # Light gray for better readability

        # Populate with existing history
        self._refresh_list()

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.is_visible = False
        if self.window:
            self.window.destroy()
            self.window = None
            self.text_widget = None

    def add_translation(self, text, speaker=None, speaker_color=None, timestamp=None):
        if not text or not text.strip():
            return
        if timestamp is None:
            timestamp = datetime.now()

        self.history.append({
            'timestamp': timestamp,
            'speaker': speaker,
            'text': text,
            'color': speaker_color
        })

        if self.text_widget and self.text_widget.winfo_exists():
            self._add_entry_to_widget(self.history[-1])
            self.text_widget.see(tk.END)
            self.count_label.config(text=f"Total: {len(self.history)} translations")

    def _add_entry_to_widget(self, entry):
        self.text_widget.config(state='normal')

        # Add timestamp
        self.text_widget.insert(tk.END, f"[{entry['timestamp']:%H:%M:%S}] ", "timestamp")

        # Add speaker label if present
        if entry['speaker']:
            speaker_num = int(entry['speaker'].split()[-1]) if entry['speaker'].startswith('Speaker') else 1
            tag = f"speaker_{speaker_num}"
            self.text_widget.insert(tk.END, f"[{entry['speaker']}] ", tag)

        # Add text
        self.text_widget.insert(tk.END, f"{entry['text']}\n", "text")

        self.text_widget.config(state='disabled')

    def _refresh_list(self):
        if not self.text_widget:
            return
        self.text_widget.config(state='normal')
        self.text_widget.delete('1.0', tk.END)
        for entry in self.history:
            self._add_entry_to_widget(entry)
        self.text_widget.config(state='disabled')

    def _copy_all(self):
        if not self.history:
            return
        lines = []
        for entry in self.history:
            speaker_part = f"[{entry['speaker']}] " if entry['speaker'] else ""
            lines.append(f"[{entry['timestamp']:%H:%M:%S}] {speaker_part}{entry['text']}")
        text = '\n'.join(lines)
        self.parent.clipboard_clear()
        self.parent.clipboard_append(text)
        logger.info(f"Copied all {len(self.history)} translations to clipboard")

    def _clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all translation history?"):
            self.history.clear()
            if self.text_widget:
                self.text_widget.config(state='normal')
                self.text_widget.delete('1.0', tk.END)
                self.text_widget.config(state='disabled')
                self.count_label.config(text="Total: 0 translations")
            logger.info("Translation history cleared")

    def _export_history(self):
        if not self.history:
            messagebox.showinfo("Export", "No translations to export.")
            return
        filename = f"translations_{datetime.now():%Y%m%d_%H%M%S}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.history:
                    speaker_part = f"[{entry['speaker']}] " if entry['speaker'] else ""
                    f.write(f"[{entry['timestamp']:%H:%M:%S}] {speaker_part}{entry['text']}\n")
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

        tk.Label(main_frame, text="Session Statistics",
                 font=("Helvetica", 14, "bold")).pack(pady=(0, 15))

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

        preset_frame = tk.Frame(self.frame)
        preset_frame.pack(fill='x', pady=(0, 5))

        tk.Label(preset_frame, text="Quick Position:").pack(side='left')

        positions = [("Top", "top"), ("Center", "center"), ("Bottom", "bottom")]
        for text, pos in positions:
            tk.Button(preset_frame, text=text, width=8,
                      command=lambda p=pos: self.on_position_change(p)).pack(side='left', padx=2)

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
        self.root.geometry("680x950")
        self.root.resizable(True, True)
        self.root.minsize(650, 750)

        self.worker_threads = []
        self.stop_event = None
        self.subtitle_window = None
        self.subtitle_label = None
        self.subtitle_shadow_label = None
        self.speaker_label = None  # NEW: Label for speaker indicator
        self.background_canvas = None
        self.background_rect = None
        self.last_subtitle = ""
        self.last_speaker = None
        self.last_speaker_color = None
        self.subtitle_history = []
        self.subtitle_lines = []  # Store recent subtitle lines for multi-line display
        self.max_subtitle_lines = 3  # Maximum lines to show at once
        self.is_multiline_mode = False  # Track current display mode
        self.subtitle_visible = False  # Track if subtitle is currently shown
        self._drag_data = {"x": 0, "y": 0}
        self.device_list = []
        self.diarization_enabled = False

        # Panels
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
        batch = []
        for _ in range(100):
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
        # Scrollable main frame
        main_canvas = tk.Canvas(self.root)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = tk.Frame(main_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Header
        header_frame = tk.Frame(self.scrollable_frame)
        header_frame.pack(pady=10, padx=20, fill='x')

        tk.Label(header_frame, text="Live Audio Translator",
                 font=("Helvetica", 18, "bold")).pack()
        tk.Label(header_frame, text="Real-time Japanese to English translation with Speaker Diarization",
                 font=("Helvetica", 9), fg="grey").pack()

        # Audio device selection
        device_frame = tk.LabelFrame(self.scrollable_frame, text="Audio Device", padx=10, pady=10)
        device_frame.pack(pady=5, padx=20, fill='x')

        device_row = tk.Frame(device_frame)
        device_row.pack(fill='x')

        self.device_var = tk.StringVar()
        self.device_menu = tk.OptionMenu(device_row, self.device_var, "Loading...")
        self.device_menu.pack(side="left", padx=5, expand=True, fill='x')

        tk.Button(device_row, text="Refresh", command=self.refresh_devices, width=8).pack(side="right", padx=5)

        self.refresh_devices()
        self.device_var.trace_add('write', self.on_device_select)

        # Status
        status_frame = tk.Frame(self.scrollable_frame)
        status_frame.pack(pady=10, padx=20, fill='x')

        self.status_label = tk.Label(status_frame, text="Status: Ready",
                                      font=("Helvetica", 11, "bold"), fg="green")
        self.status_label.pack()

        self.diarization_status_label = tk.Label(status_frame, text="Speaker Diarization: Disabled",
                                                   font=("Helvetica", 9), fg="gray")
        self.diarization_status_label.pack()

        # Control buttons
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

        tk.Button(quick_frame, text="History (Ctrl+H)", command=self.history_panel.show, width=14).pack(side="left", padx=3)
        tk.Button(quick_frame, text="Statistics", command=self.stats_panel.show, width=12).pack(side="left", padx=3)
        tk.Button(quick_frame, text="Log (Ctrl+L)", command=self.open_log_window, width=12).pack(side="left", padx=3)

        # Settings notebook
        settings_notebook = ttk.Notebook(self.scrollable_frame)
        settings_notebook.pack(pady=10, padx=20, fill='x')

        # Tab 1: Audio Settings
        audio_tab = tk.Frame(settings_notebook, padx=10, pady=10)
        settings_notebook.add(audio_tab, text="Audio")

        # Dynamic chunking
        dynamic_frame = tk.LabelFrame(audio_tab, text="Dynamic Chunking", padx=10, pady=10)
        dynamic_frame.pack(pady=5, fill="x")

        self.dynamic_chunk_var = tk.BooleanVar(value=self.config.use_dynamic_chunking)
        tk.Checkbutton(dynamic_frame, text="Enable Dynamic Chunks (Recommended)",
                       variable=self.dynamic_chunk_var, font=("Helvetica", 10)).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(dynamic_frame, text="Silence Timeout (s):").grid(row=1, column=0, sticky="w", pady=2)
        self.dyn_silence_var = tk.StringVar(value=str(self.config.dynamic_silence_timeout))
        tk.Entry(dynamic_frame, textvariable=self.dyn_silence_var, width=8).grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(dynamic_frame, text="Max Duration (s):").grid(row=1, column=2, sticky="w", padx=(10,0))
        self.dyn_max_dur_var = tk.StringVar(value=str(self.config.dynamic_max_chunk_duration))
        tk.Entry(dynamic_frame, textvariable=self.dyn_max_dur_var, width=8).grid(row=1, column=3, padx=5, sticky="w")

        tk.Label(dynamic_frame, text="Min Speech (s):").grid(row=2, column=0, sticky="w", pady=2)
        self.dyn_min_speech_var = tk.StringVar(value=str(self.config.dynamic_min_speech_duration))
        tk.Entry(dynamic_frame, textvariable=self.dyn_min_speech_var, width=8).grid(row=2, column=1, padx=5, sticky="w")

        # VAD settings
        vad_frame = tk.LabelFrame(audio_tab, text="Voice Activity Detection", padx=10, pady=10)
        vad_frame.pack(pady=5, fill="x")

        self.vad_var = tk.BooleanVar(value=self.config.use_vad_filter)
        tk.Checkbutton(vad_frame, text="Enable VAD Filter", variable=self.vad_var, font=("Helvetica", 10)).grid(row=0, column=0, columnspan=2, sticky="w")

        tk.Label(vad_frame, text="Volume Threshold:").grid(row=1, column=0, sticky="w", pady=2)
        self.volume_var = tk.StringVar(value=str(self.config.volume_threshold))
        tk.Entry(vad_frame, textvariable=self.volume_var, width=8).grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(vad_frame, text="VAD Threshold (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.vad_threshold_var = tk.StringVar(value=str(int(self.config.vad_threshold * 100)))
        tk.Entry(vad_frame, textvariable=self.vad_threshold_var, width=8).grid(row=2, column=1, padx=5, sticky="w")

        # Tab 2: Speaker Diarization (NEW)
        diarization_tab = tk.Frame(settings_notebook, padx=10, pady=10)
        settings_notebook.add(diarization_tab, text="Speakers")

        # Enable diarization
        enable_frame = tk.LabelFrame(diarization_tab, text="Speaker Diarization", padx=10, pady=10)
        enable_frame.pack(pady=5, fill="x")

        self.diarization_var = tk.BooleanVar(value=self.config.use_speaker_diarization)
        tk.Checkbutton(enable_frame, text="Enable Speaker Diarization",
                       variable=self.diarization_var, font=("Helvetica", 10, "bold")).pack(anchor='w')

        tk.Label(enable_frame, text="Identifies different speakers in the audio stream",
                 font=("Helvetica", 9), fg="gray").pack(anchor='w', pady=(0, 5))

        tk.Label(enable_frame, text="Note: Requires ~4GB additional VRAM and HuggingFace token",
                 font=("Helvetica", 9), fg="orange").pack(anchor='w')

        # HuggingFace token
        token_frame = tk.LabelFrame(diarization_tab, text="HuggingFace Token", padx=10, pady=10)
        token_frame.pack(pady=5, fill="x")

        tk.Label(token_frame, text="Required for pyannote speaker diarization model",
                 font=("Helvetica", 9), fg="gray").pack(anchor='w')

        token_row = tk.Frame(token_frame)
        token_row.pack(fill='x', pady=5)

        self.hf_token_var = tk.StringVar(value=self.config.hf_token or "")
        self.hf_token_entry = tk.Entry(token_row, textvariable=self.hf_token_var, width=40, show="*")
        self.hf_token_entry.pack(side="left", padx=5, expand=True, fill='x')

        self.show_token_var = tk.BooleanVar(value=False)
        tk.Checkbutton(token_row, text="Show", variable=self.show_token_var,
                       command=self._toggle_token_visibility).pack(side="left", padx=5)

        tk.Label(token_frame, text="Get token at: https://huggingface.co/settings/tokens",
                 font=("Helvetica", 8), fg="blue", cursor="hand2").pack(anchor='w')

        # Speaker settings
        speaker_settings_frame = tk.LabelFrame(diarization_tab, text="Speaker Settings", padx=10, pady=10)
        speaker_settings_frame.pack(pady=5, fill="x")

        tk.Label(speaker_settings_frame, text="Min Speakers:").grid(row=0, column=0, sticky="w", pady=2)
        self.min_speakers_var = tk.StringVar(value=str(self.config.min_speakers))
        tk.Entry(speaker_settings_frame, textvariable=self.min_speakers_var, width=5).grid(row=0, column=1, padx=5, sticky="w")

        tk.Label(speaker_settings_frame, text="Max Speakers:").grid(row=0, column=2, sticky="w", pady=2, padx=(10,0))
        self.max_speakers_var = tk.StringVar(value=str(self.config.max_speakers))
        tk.Entry(speaker_settings_frame, textvariable=self.max_speakers_var, width=5).grid(row=0, column=3, padx=5, sticky="w")

        self.show_speaker_colors_var = tk.BooleanVar(value=self.config.show_speaker_colors)
        tk.Checkbutton(speaker_settings_frame, text="Show speaker colors in subtitle",
                       variable=self.show_speaker_colors_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=5)

        # Speaker color preview
        color_frame = tk.Frame(speaker_settings_frame)
        color_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=5)

        tk.Label(color_frame, text="Speaker Colors:").pack(side="left")
        for i, color in enumerate(SPEAKER_COLORS[:5]):
            lbl = tk.Label(color_frame, text=f" {i+1} ", bg=color, fg="white", font=("Helvetica", 9, "bold"))
            lbl.pack(side="left", padx=2)

        # Tab 3: Appearance
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

        tk.Label(font_frame, text="Font Weight:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.font_weight_var = tk.StringVar(value=self.config.font_weight)
        tk.OptionMenu(font_frame, self.font_weight_var, 'normal', 'bold',
                      command=self.on_font_weight_change).grid(row=0, column=3, padx=5, sticky="w")

        tk.Label(font_frame, text="Font Color:").grid(row=1, column=0, sticky="w", pady=2)
        self.font_color_btn = tk.Button(font_frame, text="Pick", command=self.pick_font_color, width=6)
        self.font_color_btn.grid(row=1, column=1, padx=5, sticky="w")
        self.font_color_display = tk.Label(font_frame, text='    ', bg=self.config.subtitle_font_color, relief="solid", borderwidth=1)
        self.font_color_display.grid(row=1, column=2, padx=5, sticky="w")

        self.text_shadow_var = tk.BooleanVar(value=getattr(self.config, 'text_shadow', True))
        tk.Checkbutton(font_frame, text="Text Shadow", variable=self.text_shadow_var,
                       command=self.on_text_shadow_change).grid(row=1, column=3, sticky="w")

        # Background settings
        bg_frame = tk.LabelFrame(appearance_tab, text="Background Settings", padx=10, pady=10)
        bg_frame.pack(pady=5, fill="x")

        tk.Label(bg_frame, text="BG Mode:").grid(row=0, column=0, sticky="w", pady=2)
        self.bg_mode_var = tk.StringVar(value=self.config.subtitle_bg_mode)
        tk.OptionMenu(bg_frame, self.bg_mode_var, 'transparent', 'solid',
                      command=self.set_bg_mode).grid(row=0, column=1, padx=5, sticky="w")

        tk.Label(bg_frame, text="Opacity (%):").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.opacity_var = tk.StringVar(value=str(int(self.config.window_opacity * 100)))
        self.opacity_entry = tk.Entry(bg_frame, textvariable=self.opacity_var, width=8)
        self.opacity_entry.grid(row=0, column=3, padx=5, sticky="w")
        self.opacity_entry.bind('<KeyRelease>', self.on_opacity_change)

        tk.Label(bg_frame, text="BG Color:").grid(row=1, column=0, sticky="w", pady=2)
        self.bg_color_btn = tk.Button(bg_frame, text="Pick", command=self.pick_bg_color, width=6)
        self.bg_color_btn.grid(row=1, column=1, padx=5, sticky="w")
        self.bg_color_display = tk.Label(bg_frame, text='    ', bg=self.config.subtitle_bg_color, relief="solid", borderwidth=1)
        self.bg_color_display.grid(row=1, column=2, padx=5, sticky="w")

        # Position controls
        self.position_control = SubtitlePositionControl(self.root, self.on_subtitle_position_change)
        self.position_control.create(appearance_tab).pack(pady=5, fill="x")

        # Tab 4: Presets
        presets_tab = tk.Frame(settings_notebook, padx=10, pady=10)
        settings_notebook.add(presets_tab, text="Presets")

        load_frame = tk.LabelFrame(presets_tab, text="Load Preset", padx=10, pady=10)
        load_frame.pack(pady=5, fill="x")

        self.preset_var = tk.StringVar()
        self.preset_menu = tk.OptionMenu(load_frame, self.preset_var, "No presets found")
        self.preset_menu.pack(side="left", padx=5, expand=True, fill='x')
        tk.Button(load_frame, text="Load", command=self.load_preset, width=8).pack(side="left", padx=5)

        save_frame = tk.LabelFrame(presets_tab, text="Save Preset", padx=10, pady=10)
        save_frame.pack(pady=5, fill="x")

        self.save_preset_name_var = tk.StringVar()
        tk.Entry(save_frame, textvariable=self.save_preset_name_var).pack(side="left", padx=5, expand=True, fill='x')
        tk.Button(save_frame, text="Save", command=self.save_preset, width=8).pack(side="left", padx=5)

        mgmt_frame = tk.Frame(presets_tab)
        mgmt_frame.pack(pady=10, fill='x')
        tk.Button(mgmt_frame, text="Refresh List", command=self.refresh_preset_list, width=12).pack(side="left", padx=5)
        tk.Button(mgmt_frame, text="Reset to Defaults", command=self.reset_to_defaults, width=14).pack(side="left", padx=5)

        self.refresh_preset_list()

        # Shortcuts info
        shortcuts_frame = tk.LabelFrame(self.scrollable_frame, text="Keyboard Shortcuts", padx=10, pady=5)
        shortcuts_frame.pack(pady=10, padx=20, fill='x')

        tk.Label(shortcuts_frame, text="F5: Start  |  F6: Stop  |  Ctrl+H: History  |  Ctrl+L: Log  |  Ctrl+Q: Quit\n"
                                        "Subtitle Window: Drag to move  |  Ctrl+C: Copy  |  Ctrl+S: Save  |  Esc: Stop",
                 font=("Consolas", 9), justify="center").pack(pady=5)

    def _toggle_token_visibility(self):
        if self.show_token_var.get():
            self.hf_token_entry.config(show="")
        else:
            self.hf_token_entry.config(show="*")

    def on_subtitle_position_change(self, position):
        if not self.subtitle_window or not self.subtitle_window.winfo_exists():
            return

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width = self.subtitle_window.winfo_width()
        win_height = self.subtitle_window.winfo_height()

        current_x = self.subtitle_window.winfo_x()
        current_y = self.subtitle_window.winfo_y()
        move_step = 50

        if position == "top":
            new_x, new_y = (screen_width - win_width) // 2, 50
        elif position == "center":
            new_x, new_y = (screen_width - win_width) // 2, (screen_height - win_height) // 2
        elif position == "bottom":
            new_x, new_y = (screen_width - win_width) // 2, screen_height - win_height - 100
        elif position == "up":
            new_x, new_y = current_x, max(0, current_y - move_step)
        elif position == "down":
            new_x, new_y = current_x, min(screen_height - win_height, current_y + move_step)
        elif position == "left":
            new_x, new_y = max(0, current_x - move_step), current_y
        elif position == "right":
            new_x, new_y = min(screen_width - win_width, current_x + move_step), current_y
        else:
            return

        self.subtitle_window.geometry(f"+{new_x}+{new_y}")

    def reset_to_defaults(self):
        if not messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            return

        from modules.config import (VOLUME_THRESHOLD, USE_VAD_FILTER, VAD_THRESHOLD,
                                    DEFAULT_BG_COLOR, DEFAULT_FONT_COLOR,
                                    DEFAULT_BG_MODE, DEFAULT_WINDOW_OPACITY,
                                    DEFAULT_USE_DIARIZATION, DEFAULT_MIN_SPEAKERS, DEFAULT_MAX_SPEAKERS)

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
        self.config.use_speaker_diarization = DEFAULT_USE_DIARIZATION
        self.config.min_speakers = DEFAULT_MIN_SPEAKERS
        self.config.max_speakers = DEFAULT_MAX_SPEAKERS

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
        self.diarization_var.set(self.config.use_speaker_diarization)
        self.min_speakers_var.set(str(self.config.min_speakers))
        self.max_speakers_var.set(str(self.config.max_speakers))

        self.update_subtitle_style()
        self.config.save_config()
        messagebox.showinfo("Reset", "All settings reset to defaults.")

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
        self.diarization_status_label.config(text="Speaker Diarization: Disabled", fg="gray")

    def check_gui_queue(self):
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "subtitle":
                    self.update_subtitle_text(data)
                elif msg_type == "model_loaded":
                    self.status_label.config(text="Status: Running", fg="green")
                    self.stop_button.config(state="normal")
                elif msg_type == "diarization_status":
                    if data:
                        self.diarization_status_label.config(text="Speaker Diarization: ENABLED", fg="green")
                        self.diarization_enabled = True
                    else:
                        self.diarization_status_label.config(text="Speaker Diarization: Disabled", fg="gray")
                        self.diarization_enabled = False
                elif msg_type == "error":
                    self.status_label.config(text="Status: Error!", fg="red")
                    if self.subtitle_label:
                        self.update_subtitle_text({"text": f"ERROR: {data}", "display_text": f"ERROR: {data}"})
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

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        win_width, win_height = 1000, 200
        x = (screen_width - win_width) // 2
        y = screen_height - win_height - 100

        self.subtitle_window.geometry(f"{win_width}x{win_height}+{x}+{y}")
        self.subtitle_window.wm_attributes("-topmost", True)
        self.subtitle_window.config(bg='green')
        self.subtitle_window.wm_attributes("-transparentcolor", "green")

        self.background_canvas = tk.Canvas(self.subtitle_window, bg='green', highlightthickness=0)
        self.background_canvas.pack(pady=20, padx=20, expand=True, fill="both")

        self.background_rect = self.background_canvas.create_rectangle(0, 0, 0, 0, outline="", width=0)

        # Get font settings
        font_size = self.config.font_size
        font_weight = self.config.font_weight
        font_tuple = ("Helvetica", font_size, font_weight)

        # Speaker label (positioned above subtitle) - kept for compatibility
        self.speaker_label = tk.Label(self.background_canvas, text="", font=("Helvetica", 12, "bold"),
                                       fg=self.config.subtitle_font_color,
                                       bg=self.config.subtitle_bg_color)

        # Single-line subtitle labels (default mode) - start empty, no "Waiting for audio"
        self.subtitle_shadow_label = tk.Label(self.background_canvas, text="",
                                               font=font_tuple,
                                               fg='#1c1c1c',
                                               bg=self.config.subtitle_bg_color,
                                               wraplength=900, justify="center")
        self.subtitle_label = tk.Label(self.background_canvas, text="",
                                        font=font_tuple,
                                        fg=self.config.subtitle_font_color,
                                        bg=self.config.subtitle_bg_color,
                                        wraplength=900, justify="center")
        # Don't place yet - only show when there's actual text
        self.subtitle_visible = False

        # Multi-line frame (hidden by default, used when diarization is active)
        self.subtitle_frame = tk.Frame(self.background_canvas, bg=self.config.subtitle_bg_color)
        # Don't place it yet - only show when multi-line mode is active

        # Multi-line labels (for showing recent translations)
        self.multi_line_labels = []
        self.subtitle_lines = []  # Reset on window creation
        self.is_multiline_mode = False  # Track current display mode

        # Apply initial style
        self._apply_subtitle_style()

        self.subtitle_window.bind("<Escape>", self.stop_translator)
        for widget in [self.subtitle_label, self.subtitle_shadow_label, self.background_canvas, self.speaker_label, self.subtitle_frame]:
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

    def update_subtitle_text(self, data):
        """Update subtitle with speaker info support - shows multiple lines for different speakers"""
        if not self.subtitle_label or not self.subtitle_label.winfo_exists():
            return

        # Handle both string and dict data formats
        if isinstance(data, str):
            text = data
            display_text = data
            speaker = None
            speaker_color = None
            confidence = None
        else:
            text = data.get('text', '')
            display_text = data.get('display_text', text)
            speaker = data.get('speaker')
            speaker_color = data.get('speaker_color')
            confidence = data.get('confidence')

        if text == self.last_subtitle and speaker == self.last_speaker:
            return

        self.last_subtitle = text
        self.last_speaker = speaker
        self.last_speaker_color = speaker_color

        try:
            # Check if we have speaker diarization enabled and should show multi-line
            use_multiline = (self.diarization_enabled and
                           self.config.use_speaker_diarization and
                           speaker is not None)

            if use_multiline:
                # Switch to multi-line mode
                if not self.is_multiline_mode:
                    self._switch_to_multiline_mode()

                # Add new line to subtitle_lines
                new_line = {
                    'text': text,
                    'speaker': speaker,
                    'color': speaker_color or self.config.subtitle_font_color
                }
                self.subtitle_lines.append(new_line)

                # Keep only the last N lines
                if len(self.subtitle_lines) > self.max_subtitle_lines:
                    self.subtitle_lines = self.subtitle_lines[-self.max_subtitle_lines:]

                # Update multi-line display
                self._update_multiline_subtitle()
            else:
                # Switch to single-line mode
                if self.is_multiline_mode:
                    self._switch_to_singleline_mode()

                # Only show if there's actual text
                if text and text.strip():
                    # Show the subtitle label if not visible
                    if not self.subtitle_visible:
                        self.subtitle_label.place(relx=0.5, rely=0.5, anchor="center")
                        self.subtitle_visible = True

                    # Update speaker label
                    if speaker and self.config.show_speaker_colors and self.speaker_label:
                        self.speaker_label.config(text=speaker, fg=speaker_color or "#FFFFFF",
                                                   bg=self.config.subtitle_bg_color)
                        self.speaker_label.place(relx=0.5, rely=0.3, anchor="center")
                    elif self.speaker_label:
                        self.speaker_label.place_forget()

                    # Update main subtitle
                    self.subtitle_label.config(text=text)
                    if self.subtitle_shadow_label and self.config.text_shadow:
                        self.subtitle_shadow_label.config(text=text)
                        self.subtitle_shadow_label.place(relx=0.5, rely=0.5, anchor="center", x=2, y=2)
                        self.subtitle_label.lift()

        except tk.TclError:
            return

        if text.strip() and "ERROR" not in text:
            timestamp = datetime.now()
            self.subtitle_history.append(f"[{timestamp:%H:%M:%S}] {display_text}")
            self.history_panel.add_translation(text, speaker, speaker_color, timestamp)

        self._update_background_size()
        self._resize_window_if_needed()

    def _switch_to_multiline_mode(self):
        """Switch from single-line to multi-line subtitle display"""
        self.is_multiline_mode = True

        # Hide single-line elements
        self.subtitle_label.place_forget()
        self.subtitle_shadow_label.place_forget()
        self.speaker_label.place_forget()

        # Show multi-line frame
        self.subtitle_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Resize window to accommodate multi-line content
        if self.subtitle_window:
            x, y = self.subtitle_window.winfo_x(), self.subtitle_window.winfo_y()
            self.subtitle_window.geometry(f"1000x250+{x}+{y}")
            self.background_canvas.configure(width=960, height=210)

    def _switch_to_singleline_mode(self):
        """Switch from multi-line to single-line subtitle display"""
        self.is_multiline_mode = False

        # Clear multi-line labels
        for lbl in self.multi_line_labels:
            try:
                lbl.destroy()
            except:
                pass
        self.multi_line_labels = []
        self.subtitle_lines = []

        # Hide multi-line frame
        self.subtitle_frame.place_forget()

        # Only show single-line subtitle if there's text
        if self.last_subtitle and self.last_subtitle.strip():
            self.subtitle_label.place(relx=0.5, rely=0.5, anchor="center")
            self.subtitle_visible = True
        else:
            self.subtitle_visible = False

    def _update_multiline_subtitle(self):
        """Update the multi-line subtitle display"""
        if not self.subtitle_frame or not self.subtitle_frame.winfo_exists():
            return

        # Clear existing multi-line labels
        for lbl in self.multi_line_labels:
            try:
                lbl.destroy()
            except:
                pass
        self.multi_line_labels = []

        font_size = self.config.font_size
        font_weight = self.config.font_weight

        # Create labels for each line (older lines more faded)
        for i, line_data in enumerate(self.subtitle_lines):
            # Calculate opacity based on position (newer = brighter)
            is_newest = (i == len(self.subtitle_lines) - 1)

            # Create frame for this line
            line_frame = tk.Frame(self.subtitle_frame, bg=self.config.subtitle_bg_color)
            line_frame.pack(pady=5, fill='x', padx=10)

            # Speaker label
            speaker_text = f"[{line_data['speaker']}]" if line_data['speaker'] else ""
            if speaker_text:
                speaker_lbl = tk.Label(line_frame, text=speaker_text,
                                        font=("Helvetica", font_size - 4, "bold"),
                                        fg=line_data['color'],
                                        bg=self.config.subtitle_bg_color)
                speaker_lbl.pack(side='left', padx=(0, 10))
                self.multi_line_labels.append(speaker_lbl)

            # Text label - newer lines are brighter
            if is_newest:
                text_color = self.config.subtitle_font_color
                text_font = ("Helvetica", font_size, font_weight)
            else:
                # Fade older lines slightly
                text_color = "#AAAAAA"  # Dimmer color for older lines
                text_font = ("Helvetica", font_size - 2, "normal")

            text_lbl = tk.Label(line_frame, text=line_data['text'],
                                 font=text_font,
                                 fg=text_color,
                                 bg=self.config.subtitle_bg_color,
                                 wraplength=800, justify="left")
            text_lbl.pack(side='left', fill='x', expand=True)

            self.multi_line_labels.append(line_frame)
            self.multi_line_labels.append(text_lbl)

        # Force update and resize after adding content
        self.subtitle_window.update_idletasks()

    def _update_background_size(self):
        if not self.subtitle_window or not self.background_canvas.winfo_exists():
            return
        try:
            self.subtitle_window.update_idletasks()

            # Get dimensions based on current mode
            if self.is_multiline_mode and self.subtitle_frame.winfo_exists():
                # Multi-line mode: use subtitle_frame dimensions
                label_width = self.subtitle_frame.winfo_reqwidth()
                label_height = self.subtitle_frame.winfo_reqheight()
            else:
                # Single-line mode: use subtitle_label dimensions
                label_width = self.subtitle_label.winfo_reqwidth()
                label_height = self.subtitle_label.winfo_reqheight()

            label_width = max(label_width, 200)
            label_height = max(label_height, 50)

            canvas_width = self.background_canvas.winfo_width()
            canvas_height = self.background_canvas.winfo_height()

            padding_x = max(25, min(40, label_width * 0.1))
            padding_y = max(15, min(25, label_height * 0.15))

            x0 = (canvas_width - label_width) / 2 - padding_x
            y0 = (canvas_height - label_height) / 2 - padding_y
            x1 = (canvas_width + label_width) / 2 + padding_x
            y1 = (canvas_height + label_height) / 2 + padding_y

            self.background_canvas.coords(self.background_rect, max(0, x0), max(0, y0),
                                           min(canvas_width, x1), min(canvas_height, y1))

            # Only reposition single-line label if not in multiline mode
            if not self.is_multiline_mode:
                self.subtitle_label.place(relx=0.5, rely=0.55, anchor="center")

                if self.config.text_shadow and self.subtitle_shadow_label.winfo_exists():
                    self.subtitle_shadow_label.place(x=self.subtitle_label.winfo_x() + 2,
                                                      y=self.subtitle_label.winfo_y() + 2)
                    self.background_canvas.tag_lower(self.background_rect)
                    self.subtitle_shadow_label.lift()
                    self.subtitle_label.lift()
                    if self.speaker_label:
                        self.speaker_label.lift()
                elif self.subtitle_shadow_label.winfo_exists():
                    self.subtitle_shadow_label.place_forget()
            else:
                # Multi-line mode: ensure frame is on top
                self.background_canvas.tag_lower(self.background_rect)
                self.subtitle_frame.lift()

        except tk.TclError:
            pass

    def _resize_window_if_needed(self):
        if not self.subtitle_window or not self.subtitle_label.winfo_exists():
            return
        try:
            # Get dimensions based on current mode
            if self.is_multiline_mode and self.subtitle_frame.winfo_exists():
                # Multi-line mode: use subtitle_frame dimensions
                self.subtitle_window.update_idletasks()
                label_width = self.subtitle_frame.winfo_reqwidth()
                label_height = self.subtitle_frame.winfo_reqheight()
                # Add extra height for multi-line display
                required_height = max(200, label_height + 100)
            else:
                # Single-line mode: use subtitle_label dimensions
                label_width = self.subtitle_label.winfo_reqwidth()
                label_height = self.subtitle_label.winfo_reqheight()
                required_height = max(150, label_height + 80)

            required_width = max(800, label_width + 100)

            current_width = self.subtitle_window.winfo_width()
            current_height = self.subtitle_window.winfo_height()

            if abs(required_width - current_width) > 50 or abs(required_height - current_height) > 30:
                x, y = self.subtitle_window.winfo_x(), self.subtitle_window.winfo_y()
                self.subtitle_window.geometry(f"{required_width}x{required_height}+{x}+{y}")
                self.background_canvas.configure(width=required_width-40, height=required_height-40)
                new_wraplength = max(400, required_width - 80)
                self.subtitle_label.configure(wraplength=new_wraplength)
                if self.subtitle_shadow_label:
                    self.subtitle_shadow_label.configure(wraplength=new_wraplength)
                self.subtitle_window.update_idletasks()
                self._update_background_size()
        except tk.TclError:
            pass

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

    def save_subtitle_history(self, event=None):
        if self.subtitle_history:
            filename = f"subtitles_{datetime.now():%Y%m%d_%H%M%S}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.subtitle_history))
                messagebox.showinfo("Saved", f"History saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

    def on_close(self):
        if self.stop_event:
            self.stop_event.set()
        self.apply_and_save_settings()
        if hasattr(sys, '__stdout__'):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        self.root.destroy()

    def open_log_window(self):
        if self.log_window and tk.Toplevel.winfo_exists(self.log_window):
            self.log_window.lift()
            return
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Application Log")
        self.log_window.geometry("800x500")

        toolbar = tk.Frame(self.log_window)
        toolbar.pack(fill='x', padx=5, pady=5)
        tk.Button(toolbar, text="Clear", command=lambda: self.log_text_widget.delete('1.0', tk.END) if self.log_text_widget else None, width=10).pack(side='left', padx=2)
        tk.Button(toolbar, text="Copy All", command=lambda: (self.root.clipboard_clear(), self.root.clipboard_append(self.log_text_widget.get('1.0', tk.END))) if self.log_text_widget else None, width=10).pack(side='left', padx=2)

        text_frame = tk.Frame(self.log_window)
        text_frame.pack(expand=True, fill='both', padx=5, pady=5)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')

        self.log_text_widget = tk.Text(text_frame, wrap='word', font=("Consolas", 10), state='disabled', yscrollcommand=scrollbar.set)
        self.log_text_widget.pack(expand=True, fill='both')
        scrollbar.config(command=self.log_text_widget.yview)

        if self.log_buffer:
            self.log_text_widget.config(state='normal')
            self.log_text_widget.insert('1.0', "".join(self.log_buffer))
            self.log_text_widget.see('end')
            self.log_text_widget.config(state='disabled')

        self.log_window.protocol("WM_DELETE_WINDOW", lambda: (self.log_window.destroy(), setattr(self, 'log_window', None), setattr(self, 'log_text_widget', None)))

    def pick_bg_color(self):
        color = colorchooser.askcolor(initialcolor=self.config.subtitle_bg_color)
        if color and color[1]:
            self.config.subtitle_bg_color = color[1]
            self.bg_color_display.config(bg=color[1])
            self.update_subtitle_style()

    def pick_font_color(self):
        color = colorchooser.askcolor(initialcolor=self.config.subtitle_font_color)
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

            # Diarization settings
            self.config.use_speaker_diarization = self.diarization_var.get()
            self.config.min_speakers = max(1, min(10, int(self.min_speakers_var.get())))
            self.config.max_speakers = max(1, min(10, int(self.max_speakers_var.get())))
            self.config.show_speaker_colors = self.show_speaker_colors_var.get()

            hf_token = self.hf_token_var.get().strip()
            if hf_token:
                self.config.hf_token = hf_token

            if save_to_disk:
                self.config.save_config()
            return True
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Invalid Input", f"Please check all fields.\nError: {e}")
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

        if self.config.use_speaker_diarization:
            self.diarization_status_label.config(text="Speaker Diarization: Loading...", fg="orange")
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
                                     args=(self.stop_event, audio_queue, self.config, self.gui_queue, selected_device_name),
                                     daemon=True)
        processor = threading.Thread(target=processor_thread,
                                       args=(self.stop_event, audio_queue, self.config, self.stats, self.gui_queue),
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
            # Update config from UI variables
            self.config.font_size = int(self.font_var.get())
            self.config.font_weight = self.font_weight_var.get()
            self._apply_subtitle_style()
        except (ValueError, tk.TclError):
            pass

    def _apply_subtitle_style(self):
        """Apply current style settings to subtitle window"""
        if not self.subtitle_window or not self.subtitle_label.winfo_exists():
            return
        try:
            font_size = self.config.font_size
            font_weight = self.config.font_weight
            font_tuple = ("Helvetica", font_size, font_weight)

            # Update main subtitle label
            self.subtitle_label.config(
                font=font_tuple,
                fg=self.config.subtitle_font_color,
                bg=self.config.subtitle_bg_color
            )

            # Update shadow label
            if self.subtitle_shadow_label and self.subtitle_shadow_label.winfo_exists():
                self.subtitle_shadow_label.config(
                    font=font_tuple,
                    fg='#1c1c1c',
                    bg=self.config.subtitle_bg_color
                )

            # Update speaker label
            if self.speaker_label and self.speaker_label.winfo_exists():
                self.speaker_label.config(
                    fg=self.config.subtitle_font_color,
                    bg=self.config.subtitle_bg_color
                )

            # Update multi-line frame
            if hasattr(self, 'subtitle_frame') and self.subtitle_frame and self.subtitle_frame.winfo_exists():
                self.subtitle_frame.config(bg=self.config.subtitle_bg_color)

            # Update background
            if self.background_canvas and self.background_rect:
                self.background_canvas.itemconfig(
                    self.background_rect,
                    fill=self.config.subtitle_bg_color,
                    outline=self.config.border_color,
                    width=self.config.border_width
                )

            # Update window transparency
            if self.config.subtitle_bg_mode == 'transparent':
                self.subtitle_window.wm_attributes("-alpha", self.config.window_opacity)
            else:
                self.subtitle_window.wm_attributes("-alpha", 1.0)

            self._update_background_size()
        except (ValueError, tk.TclError) as e:
            logger.debug(f"Error applying subtitle style: {e}")

    def on_device_select(self, *args):
        self.config.selected_audio_device = self.device_var.get()

    def download_model(self):
        self.status_label.config(text="Status: Downloading model...", fg="blue")
        self.download_button.config(state="disabled")
        self.start_button.config(state="disabled")
        self.root.update_idletasks()

        def do_download():
            try:
                ensure_model_downloaded(MODEL_ID, self.config.model_cache_dir)
                self.gui_queue.put(("status_update", ("Status: Models ready!", "green")))
            except Exception as e:
                self.gui_queue.put(("status_update", (f"Status: Download failed", "red")))
            finally:
                self.gui_queue.put(("download_finished", None))

        def process_queue():
            try:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "status_update":
                    self.status_label.config(text=data[0], fg=data[1])
                elif msg_type == "download_finished":
                    self.download_button.config(state="normal")
                    self.start_button.config(state="normal")
                    return
            except Empty:
                pass
            self.root.after(100, process_queue)

        threading.Thread(target=do_download, daemon=True).start()
        process_queue()

    def refresh_preset_list(self):
        preset_dir = "presets"
        os.makedirs(preset_dir, exist_ok=True)
        preset_files = [f for f in os.listdir(preset_dir) if f.endswith('.json')]
        presets = [os.path.splitext(f)[0] for f in preset_files]

        menu = self.preset_menu["menu"]
        menu.delete(0, "end")

        if not presets:
            menu.add_command(label="No presets found", state="disabled")
            self.preset_var.set("No presets found")
        else:
            for name in sorted(presets):
                menu.add_command(label=name, command=lambda v=name: self.preset_var.set(v))
            self.preset_var.set(presets[0])

    def save_preset(self):
        preset_name = self.save_preset_name_var.get().strip()
        if not preset_name:
            messagebox.showwarning("Warning", "Enter a preset name.")
            return

        if not self.apply_and_save_settings(save_to_disk=False):
            return

        preset_data = self.config.to_dict()
        # Don't save sensitive data in presets
        preset_data.pop('hf_token', None)

        file_path = os.path.join("presets", f"{preset_name}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=4)
            messagebox.showinfo("Success", f"Preset '{preset_name}' saved.")
            self.refresh_preset_list()
            self.save_preset_name_var.set("")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def load_preset(self):
        preset_name = self.preset_var.get()
        if not preset_name or preset_name == "No presets found":
            messagebox.showwarning("Warning", "No preset selected.")
            return

        file_path = os.path.join("presets", f"{preset_name}.json")
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Preset file not found.")
            self.refresh_preset_list()
            return

        try:
            with open(file_path, 'r') as f:
                preset_data = json.load(f)

            for key, value in preset_data.items():
                if key != 'hf_token':  # Don't overwrite token from preset
                    setattr(self.config, key, value)

            # Update UI
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
            self.dynamic_chunk_var.set(self.config.use_dynamic_chunking)
            self.dyn_silence_var.set(str(self.config.dynamic_silence_timeout))
            self.dyn_max_dur_var.set(str(self.config.dynamic_max_chunk_duration))
            self.dyn_min_speech_var.set(str(self.config.dynamic_min_speech_duration))
            self.diarization_var.set(getattr(self.config, 'use_speaker_diarization', False))
            self.min_speakers_var.set(str(getattr(self.config, 'min_speakers', 1)))
            self.max_speakers_var.set(str(getattr(self.config, 'max_speakers', 5)))

            self.update_subtitle_style()
            messagebox.showinfo("Success", f"Preset '{preset_name}' loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
