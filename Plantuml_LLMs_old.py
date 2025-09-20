#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Plantuml_LLMs.py
Tkinter GUI with floating image, mutually exclusive dataset checkboxes,
progress bar, and live script output in GUI window.
"""

import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import threading
import sys
import os
from pathlib import Path

# ============================================================
# Base directory = where this file lives
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# Helper: Run external Python script with debugging (UTF-8 forced)
# ============================================================
def run_script(script_path):
    def task():
        abs_path = BASE_DIR / script_path
        abs_path_str = str(abs_path)

        # Log debug info
        log_text.insert(tk.END, f"\n{'='*60}\n")
        log_text.insert(tk.END, f"[🔍 DEBUG] Starting script execution\n")
        log_text.insert(tk.END, f"[📄 SCRIPT] {script_path}\n")
        log_text.insert(tk.END, f"[🎯 FULL_PATH] {abs_path_str}\n")
        log_text.insert(tk.END, f"[🐍 PYTHON] {sys.executable}\n")

        if not abs_path.exists():
            log_text.insert(tk.END, f"[✗ ERROR] Script not found: {abs_path_str}\n")
            log_text.see(tk.END)
            return

        # Special case: show folder contents for first script
        if "01_Scoring_prompts_using_huggingface.py" in script_path:
            parent_dir = abs_path.parent
            log_text.insert(tk.END, f"\n[📂 LISTING FOLDER] {parent_dir}\n")
            try:
                for item in sorted(parent_dir.iterdir()):
                    if item.is_dir():
                        log_text.insert(tk.END, f"  📁 {item.name}/\n")
                    else:
                        log_text.insert(tk.END, f"  📄 {item.name}\n")
            except Exception as e:
                log_text.insert(tk.END, f"[✗ ERROR] Could not list folder: {e}\n")
            log_text.see(tk.END)

        log_text.insert(tk.END, f"[✓ FOUND] Script exists, starting execution...\n")
        log_text.see(tk.END)

        progress_bar.start(10)
        run_btn_state("disabled")

        try:
            cmd = [sys.executable, abs_path_str]
            log_text.insert(tk.END, f"[⚡ COMMAND] {' '.join(cmd)}\n")
            log_text.see(tk.END)

            # ✅ Force UTF-8 environment
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            log_text.insert(tk.END, f"[▶ RUNNING] Process started (PID: {process.pid})\n")
            log_text.see(tk.END)

            # Live read output
            while True:
                if process.poll() is not None:
                    break
                stdout_line = process.stdout.readline()
                if stdout_line:
                    log_text.insert(tk.END, f"[OUT] {stdout_line}")
                    log_text.see(tk.END)
                    root.update_idletasks()

            stdout, stderr = process.communicate()
            if stdout:
                log_text.insert(tk.END, f"[OUT] {stdout}\n")
            if stderr:
                log_text.insert(tk.END, f"[ERR] {stderr}\n")

            return_code = process.returncode
            log_text.insert(tk.END, f"[📊 RESULT] Return code: {return_code}\n")
            if return_code == 0:
                log_text.insert(tk.END, f"[✓ SUCCESS] {script_path} completed successfully\n")
            else:
                log_text.insert(tk.END, f"[✗ FAILED] {script_path} failed\n")

        except Exception as e:
            log_text.insert(tk.END, f"[✗ ERROR] {e}\n")

        finally:
            progress_bar.stop()
            run_btn_state("normal")
            log_text.insert(tk.END, f"[🏁 FINISHED] Script execution completed\n")
            log_text.insert(tk.END, "=" * 60 + "\n")
            log_text.see(tk.END)

    threading.Thread(target=task, daemon=True).start()

def run_btn_state(state):
    """Enable/disable all pipeline buttons while running."""
    for child in process_frame.winfo_children():
        if isinstance(child, tk.Button):
            child.config(state=state)
    btn_analysis.config(state=state)
    btn_run_all.config(state=state)

# ============================================================
# File loader functions
# ============================================================
def load_user_stories():
    filename = filedialog.askopenfilename(
        title="Select User Stories File",
        filetypes=[("Data Files", "*.json *.csv *.xlsx"), ("All Files", "*.*")]
    )
    if filename:
        log_text.insert(tk.END, f"\n[📂 User Stories Loaded] {filename}\n")
        log_text.see(tk.END)

def load_user_llm():
    filename = filedialog.askopenfilename(
        title="Select User LLM Replies File",
        filetypes=[("Data Files", "*.json *.csv *.xlsx"), ("All Files", "*.*")]
    )
    if filename:
        log_text.insert(tk.END, f"\n[📂 User LLM Replies Loaded] {filename}\n")
        log_text.see(tk.END)

def clear_log():
    log_text.delete(1.0, tk.END)
    log_text.insert(tk.END, "[🎯 READY] PlantUML LLMs Pipeline GUI - Log cleared\n")
    log_text.insert(tk.END, f"[📁 BASE DIR] {BASE_DIR}\n")

def toggle_datasets(source):
    if source == "user":
        if use_user_var.get():
            use_current_var.set(False)
            btn_user_stories.config(state="normal")
            btn_user_llm.config(state="normal")
        else:
            use_current_var.set(True)
            btn_user_stories.config(state="disabled")
            btn_user_llm.config(state="disabled")
    elif source == "current":
        if use_current_var.get():
            use_user_var.set(False)
            btn_user_stories.config(state="disabled")
            btn_user_llm.config(state="disabled")
        else:
            use_user_var.set(True)
            btn_user_stories.config(state="normal")
            btn_user_llm.config(state="normal")

# ============================================================
# GUI setup
# ============================================================
root = tk.Tk()
root.title("An Open Source Pipeline for Evaluating Large Language Models in UML Activity Diagram Generation via Python")

root.geometry("1200x800")

# --- Title
#title_label = tk.Label(root, text="Open Source Stories / PlantUML LLMs Replies",
                       #font=("Arial", 16, "bold"))
#title_label.pack(pady=10, anchor="w", padx=10)

# --- Floating Image in Top-Right
try:
    image_obj = tk.PhotoImage(file=str(BASE_DIR / "Image.png"))
    img_label = tk.Label(root, image=image_obj, borderwidth=0)
    img_label.place(relx=1.0, y=10, anchor="ne")
except Exception:
    img_label = tk.Label(root, text="[Image Not Found]", font=("Arial", 10, "italic"))
    img_label.place(relx=1.0, y=10, anchor="ne")

# --- Dataset selection
dataset_frame = tk.Frame(root)
dataset_frame.pack(pady=5, anchor="w", padx=20)

use_current_var = tk.BooleanVar(value=True)
use_user_var = tk.BooleanVar(value=False)

tk.Checkbutton(dataset_frame, text="Use of current Datasets",
               variable=use_current_var, font=("Arial", 11, "bold"),
               command=lambda: toggle_datasets("current")).grid(row=1, column=0, sticky="w", padx=5, pady=2)

tk.Checkbutton(dataset_frame, text="Use of User Datasets",
               variable=use_user_var, font=("Arial", 11, "bold"),
               command=lambda: toggle_datasets("user")).grid(row=2, column=0, sticky="w", padx=5, pady=2)
btn_frame = tk.Frame(dataset_frame)
btn_frame.grid(row=0, column=1, rowspan=2, padx=20)

btn_user_stories = tk.Button(btn_frame, text="Loading User Stories", width=25,
                             state="disabled", command=load_user_stories)
btn_user_stories.pack(pady=3)

btn_user_llm = tk.Button(btn_frame, text="Loading User LLM replies", width=25,
                         state="disabled", command=load_user_llm)
btn_user_llm.pack(pady=3)

# --- Main horizontal container
main_container = tk.Frame(root)
main_container.pack(fill="both", expand=True, padx=10, pady=10)

# --- Left side (pipeline + controls)
left_frame = tk.Frame(main_container)
left_frame.pack(side="left", fill="y", padx=5, pady=5)

process_frame = tk.LabelFrame(left_frame, text="Pipeline Processes", padx=10, pady=10,
                              font=("Arial", 12, "bold"))
process_frame.pack(padx=10, pady=15, fill="x")

steps = [
    ("1. Assessing row prompt complexity", "src/01_Scoring_prompts_using_huggingface.py"),
    ("2. Using NLP to convert simple prompts", "src/02_Huggingfacs_Transformer_Standard_prompt.py"),
    ("3. Syntactic evaluation", "src/03_Syntatic_Elemnt_Score.py"),
    ("4. Stories similarity", "src/04_PlantUML_similarities.py"),
    ("5. Prompt ↔ Code correspond", "src/05_Symantic_prompt_and_code_coresponding_huggingface.py"),
    ("6. Image code to render", "src/06_Image_plantuml_render_to_jpg.py"),
    ("7. Select optimal prompts for human assessing", "src/07_Select_optimal_prompts_for_human_assessing.py"),
    ("8. Group merge: Image + Human assessing", "src/08_Group_MergeImage_for_selected_prompt_Humanassessing.py"),
]

for i, (label, script) in enumerate(steps):
    b = tk.Button(process_frame, text=label, anchor="w", width=65,
                  command=lambda s=script: run_script(s))
    b.grid(row=i, column=0, sticky="w", pady=2)

# Controls under process buttons
control_frame = tk.Frame(left_frame)
control_frame.pack(fill="x", pady=10)

btn_run_all = tk.Button(control_frame, text="🚀 Run All Pipeline",
                        width=20, height=2, relief="groove",
                        font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                        command=lambda: [run_script(s) for _, s in steps])
btn_run_all.pack(side="left", padx=5)

btn_analysis = tk.Button(control_frame, text="📊 Analysis",
                         width=20, height=2, relief="groove",
                         font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
                         command=lambda: run_script("src/analyze_results.py"))
btn_analysis.pack(side="left", padx=5)

progress_frame = tk.Frame(left_frame)
progress_frame.pack(fill="x", padx=20, pady=5)

progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
progress_bar.pack(fill="x")

# --- Right side (log)
right_frame = tk.Frame(main_container)
right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

log_frame = tk.LabelFrame(right_frame, text="Execution Log & Script Output", padx=5, pady=5)
log_frame.pack(fill="both", expand=True)

log_text = tk.Text(log_frame, wrap="word", bg="black", fg="white",
                   height=15, font=("Consolas", 9))
scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
log_text.configure(yscrollcommand=scrollbar.set)

log_text.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

log_text.tag_config("error", foreground="red")

log_text.insert(tk.END, "[🎯 READY] PlantUML LLMs Pipeline GUI - Ready to execute scripts\n")
log_text.insert(tk.END, f"[📁 BASE DIR] {BASE_DIR}\n")
log_text.insert(tk.END, f"[🐍 PYTHON] {sys.executable}\n")

root.mainloop()
