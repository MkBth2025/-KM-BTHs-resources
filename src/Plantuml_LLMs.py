#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Plantuml_LLMs.py
Tkinter GUI with floating image, mutually exclusive dataset checkboxes,
progress bar, live script output, and safe file loaders with warnings + backups.
"""


import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import threading
import sys
import os
import shutil
from pathlib import Path
import shlex


# ============================================================
# Base directory = where this file lives
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Enhanced: Run script with command line arguments support (Fixed)
# ============================================================
def run_script_with_args(command_string):
    """Run script that may include command line arguments - Fixed version."""
    def task():
        log_text.insert(tk.END, f"\n{'='*60}\n")
        log_text.insert(tk.END, f"[🔍 DEBUG] Starting script execution with arguments\n")
        log_text.insert(tk.END, f"[📄 COMMAND] {command_string}\n")
        log_text.insert(tk.END, f"[🐍 PYTHON] {sys.executable}\n")
        log_text.insert(tk.END, f"[📁 WORKING_DIR] {BASE_DIR}\n")

        progress_bar.start(10)
        run_btn_state("disabled")

        try:
            # Method 1: Use shell=True to mimic VSCode terminal behavior exactly
            if sys.platform.startswith('win'):
                # Windows: Use shell=True for better compatibility
                cmd = command_string
                shell_mode = True
            else:
                # Linux/macOS: Parse command properly
                cmd_parts = shlex.split(command_string)
                if cmd_parts[0] == "python":
                    cmd_parts[0] = sys.executable
                cmd = cmd_parts
                shell_mode = False
            
            log_text.insert(tk.END, f"[⚡ PARSED_COMMAND] {cmd}\n")
            log_text.insert(tk.END, f"[🖥️ SHELL_MODE] {shell_mode}\n")

            # ✅ Force UTF-8 environment and add Python path
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONPATH"] = str(BASE_DIR)
            
            # Add current directory to PATH for better module resolution
            if "PATH" in env:
                env["PATH"] = str(BASE_DIR) + os.pathsep + env["PATH"]
            else:
                env["PATH"] = str(BASE_DIR)

            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                shell=shell_mode
            )

            log_text.insert(tk.END, f"[▶ RUNNING] PID: {process.pid}\n")
            log_text.see(tk.END)

            # Real-time output capture
            while True:
                if process.poll() is not None:
                    break
                stdout_line = process.stdout.readline()
                if stdout_line:
                    log_text.insert(tk.END, f"[OUT] {stdout_line}")
                    log_text.see(tk.END)
                    root.update_idletasks()

            # Get remaining output
            stdout, stderr = process.communicate()
            if stdout:
                log_text.insert(tk.END, f"[OUT] {stdout}\n")
            if stderr:
                log_text.insert(tk.END, f"[ERR] {stderr}\n")

            if process.returncode == 0:
                log_text.insert(tk.END, f"[✓ SUCCESS] {command_string}\n")
                log_text.insert(tk.END, f"[🎉 COMPLETED] Finished running of selected program: {command_string}\n")
            else:
                log_text.insert(tk.END, f"[✗ FAILED] Return code: {process.returncode}\n")
                log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {command_string} (with errors)\n")

        except Exception as e:
            log_text.insert(tk.END, f"[✗ ERROR] Exception: {str(e)}\n")
            log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {command_string} (with exception)\n")

        finally:
            progress_bar.stop()
            run_btn_state("normal")
            log_text.insert(tk.END, f"[🏁 FINISHED] Script execution completed\n")
            log_text.insert(tk.END, "=" * 60 + "\n")
            log_text.see(tk.END)

    threading.Thread(target=task, daemon=True).start()


# ============================================================
# Alternative: Direct execution method (if above doesn't work)
# ============================================================
def run_analysis_direct():
    """Direct execution of Analysis.py with arguments - Alternative method."""
    def task():
        log_text.insert(tk.END, f"\n{'='*60}\n")
        log_text.insert(tk.END, f"[🔍 DEBUG] Direct Analysis.py execution\n")
        
        script_path = BASE_DIR / "src" / "Analysis.py"
        log_text.insert(tk.END, f"[📄 SCRIPT_PATH] {script_path}\n")
        log_text.insert(tk.END, f"[📁 WORKING_DIR] {BASE_DIR}\n")

        if not script_path.exists():
            log_text.insert(tk.END, f"[✗ ERROR] Analysis.py not found at: {script_path}\n")
            log_text.see(tk.END)
            return

        progress_bar.start(10)
        run_btn_state("disabled")

        try:
            # Build command with explicit arguments
            cmd = [
                sys.executable,
                str(script_path),
                "--stat-yaml", "stat.yaml",
                "--report-dir", "report",
                "--outdir", "report/Analysis"
            ]
            
            log_text.insert(tk.END, f"[⚡ COMMAND] {' '.join(cmd)}\n")

            # ✅ Enhanced environment setup
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONPATH"] = str(BASE_DIR)
            
            # Ensure required directories exist
            report_dir = BASE_DIR / "report"
            analysis_dir = BASE_DIR / "report" / "Analysis"
            report_dir.mkdir(exist_ok=True)
            analysis_dir.mkdir(exist_ok=True)
            
            log_text.insert(tk.END, f"[📂 CREATED] report directories\n")

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

            log_text.insert(tk.END, f"[▶ RUNNING] PID: {process.pid}\n")
            log_text.see(tk.END)

            # Real-time output
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

            if process.returncode == 0:
                log_text.insert(tk.END, f"[✓ SUCCESS] Analysis.py completed successfully\n")
                log_text.insert(tk.END, f"[🎉 COMPLETED] Finished running of selected program: Analysis.py\n")
            else:
                log_text.insert(tk.END, f"[✗ FAILED] Analysis.py failed with return code: {process.returncode}\n")
                log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: Analysis.py (with errors)\n")

        except Exception as e:
            log_text.insert(tk.END, f"[✗ ERROR] Exception: {str(e)}\n")
            log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: Analysis.py (with exception)\n")

        finally:
            progress_bar.stop()
            run_btn_state("normal")
            log_text.insert(tk.END, f"[🏁 FINISHED] Analysis execution completed\n")
            log_text.insert(tk.END, "=" * 60 + "\n")
            log_text.see(tk.END)

    threading.Thread(target=task, daemon=True).start()


# ============================================================
# Original run_script function (unchanged)
# ============================================================
def run_script(script_path):
    def task():
        abs_path = BASE_DIR / script_path
        abs_path_str = str(abs_path)

        log_text.insert(tk.END, f"\n{'='*60}\n")
        log_text.insert(tk.END, f"[🔍 DEBUG] Starting script execution\n")
        log_text.insert(tk.END, f"[📄 SCRIPT] {script_path}\n")
        log_text.insert(tk.END, f"[🎯 FULL_PATH] {abs_path_str}\n")
        log_text.insert(tk.END, f"[🐍 PYTHON] {sys.executable}\n")

        if not abs_path.exists():
            log_text.insert(tk.END, f"[✗ ERROR] Script not found: {abs_path_str}\n")
            log_text.see(tk.END)
            return

        progress_bar.start(10)
        run_btn_state("disabled")

        try:
            cmd = [sys.executable, abs_path_str]
            log_text.insert(tk.END, f"[⚡ COMMAND] {' '.join(cmd)}\n")

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

            log_text.insert(tk.END, f"[▶ RUNNING] PID: {process.pid}\n")
            log_text.see(tk.END)

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

            if process.returncode == 0:
                log_text.insert(tk.END, f"[✓ SUCCESS] {script_path}\n")
                log_text.insert(tk.END, f"[🎉 COMPLETED] Finished running of selected program: {script_path}\n")
            else:
                log_text.insert(tk.END, f"[✗ FAILED] {script_path}\n")
                log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {script_path} (with errors)\n")

        except Exception as e:
            log_text.insert(tk.END, f"[✗ ERROR] {e}\n")
            log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {script_path} (with exception)\n")

        finally:
            progress_bar.stop()
            run_btn_state("normal")
            log_text.insert(tk.END, f"[🏁 FINISHED] Script execution completed\n")
            log_text.insert(tk.END, "=" * 60 + "\n")
            log_text.see(tk.END)

    threading.Thread(target=task, daemon=True).start()


def run_btn_state(state):
    for child in process_frame.winfo_children():
        if isinstance(child, tk.Button):
            child.config(state=state)
    btn_analysis.config(state=state)
    btn_run_all.config(state=state)


# ============================================================
# Synchronous versions for pipeline execution
# ============================================================
def run_script_with_args_sync(command_string):
    """Synchronous version - Fixed for command line arguments."""
    try:
        log_text.insert(tk.END, f"[🔄 SYNC] Executing: {command_string}\n")
        
        # Use shell=True for Windows compatibility
        if sys.platform.startswith('win'):
            cmd = command_string
            shell_mode = True
        else:
            cmd_parts = shlex.split(command_string)
            if cmd_parts[0] == "python":
                cmd_parts[0] = sys.executable
            cmd = cmd_parts
            shell_mode = False
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(BASE_DIR)
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            shell=shell_mode
        )
        
        stdout, stderr = process.communicate()
        
        if stdout:
            log_text.insert(tk.END, f"[OUT] {stdout}\n")
        if stderr:
            log_text.insert(tk.END, f"[ERR] {stderr}\n")
        
        if process.returncode == 0:
            log_text.insert(tk.END, f"[✓ SUCCESS] {command_string}\n")
            log_text.insert(tk.END, f"[🎉 COMPLETED] Finished running of selected program: {command_string}\n")
            return True
        else:
            log_text.insert(tk.END, f"[✗ FAILED] {command_string}\n")
            log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {command_string} (with errors)\n")
            return False
            
    except Exception as e:
        log_text.insert(tk.END, f"[✗ ERROR] {e}\n")
        log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {command_string} (with exception)\n")
        return False


def run_script_sync(script_path):
    """Synchronous version of run_script for sequential execution."""
    abs_path = BASE_DIR / script_path
    abs_path_str = str(abs_path)
    
    if not abs_path.exists():
        log_text.insert(tk.END, f"[✗ ERROR] Script not found: {abs_path_str}\n")
        return False
    
    try:
        cmd = [sys.executable, abs_path_str]
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        stdout, stderr = process.communicate()
        
        if stdout:
            log_text.insert(tk.END, f"[OUT] {stdout}\n")
        if stderr:
            log_text.insert(tk.END, f"[ERR] {stderr}\n")
        
        if process.returncode == 0:
            log_text.insert(tk.END, f"[✓ SUCCESS] {script_path}\n")
            log_text.insert(tk.END, f"[🎉 COMPLETED] Finished running of selected program: {script_path}\n")
            return True
        else:
            log_text.insert(tk.END, f"[✗ FAILED] {script_path}\n")
            log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {script_path} (with errors)\n")
            return False
            
    except Exception as e:
        log_text.insert(tk.END, f"[✗ ERROR] {e}\n")
        log_text.insert(tk.END, f"[⚠️ COMPLETED] Finished running of selected program: {script_path} (with exception)\n")
        return False


# ============================================================
# Enhanced: Run All Pipeline
# ============================================================
def run_all_pipeline():
    """Enhanced run all function that executes scripts sequentially with completion messages."""
    def run_all_task():
        steps = [
            ("1. Assessing row prompt complexity", "src/01_Scoring_prompts_using_huggingface.py"),
            ("2. Using NLP to convert simple prompts", "src/02_Huggingfacs_Transformer_Standard_prompt.py"),
            ("3. Syntactic evaluation", "src/03_Syntatic_Elemnt_Score.py"),
            ("4. Stories similarity", "src/04_PlantUML_similarities.py"),
            ("5. Prompt ↔ Code correspond", "src/05_Symantic_prompt_and_code_coresponding_huggingface.py"),
            ("6. Statistical Evaluation of Computational Methods", "python src/stat.py --stat-yaml stat.yaml --report-dir report --outdir report/Stat"),
            ("7. Image code to render", "src/06_Image_plantuml_render_to_jpg.py"),
            ("8. Select optimal prompts for human assessing", "src/07_Select_optimal_prompts_for_human_assessing.py"),
            ("9. Group merge: Image + Human assessing", "src/08_Group_MergeImage_for_selected_prompt_Humanassessing.py"),
        ]
        
        log_text.insert(tk.END, f"\n{'='*60}\n")
        log_text.insert(tk.END, f"[🚀 PIPELINE] Starting full pipeline execution\n")
        log_text.insert(tk.END, f"[📊 TOTAL STEPS] {len(steps)} scripts to execute\n")
        log_text.insert(tk.END, f"{'='*60}\n")
        
        for i, (label, script) in enumerate(steps, 1):
            log_text.insert(tk.END, f"\n[📋 STEP {i}/{len(steps)}] {label}\n")
            log_text.see(tk.END)
            
            if script.startswith("python "):
                run_script_with_args_sync(script)
            else:
                run_script_sync(script)
            
            log_text.insert(tk.END, f"[✅ STEP {i} COMPLETE] {label}\n")
            log_text.see(tk.END)
        
        log_text.insert(tk.END, f"\n{'='*60}\n")
        log_text.insert(tk.END, f"[🎊 PIPELINE COMPLETE] All {len(steps)} steps finished\n")
        log_text.insert(tk.END, f"{'='*60}\n")
        log_text.see(tk.END)
    
    threading.Thread(target=run_all_task, daemon=True).start()


# ============================================================
# File loader functions (unchanged)
# ============================================================
def show_warning_and_load(filetypes, allowed_files):
    def on_continue():
        data_dir = BASE_DIR / "Data"
        backup_dir = data_dir / "backup"
        data_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup current Data/
        for item in data_dir.glob("*"):
            if item.is_file():
                shutil.copy(item, backup_dir / item.name)

        filenames = filedialog.askopenfilenames(
            title="Select File(s)",
            filetypes=filetypes,
            initialdir=str(BASE_DIR)
        )
        if not filenames:
            warning.destroy()
            return

        for filename in filenames:
            src = Path(filename)
            if src.name not in allowed_files:
                messagebox.showerror("Invalid file",
                                     f"Only {', '.join(allowed_files)} allowed.\nYou selected: {src.name}")
                continue
            shutil.copy(src, data_dir / src.name)
            log_text.insert(tk.END, f"\n[📂 COPIED] {src.name} → {data_dir}\n")
            log_text.see(tk.END)

        warning.destroy()

    def on_cancel():
        warning.destroy()

    warning = tk.Toplevel(root)
    warning.title("⚠️ Warning")
    warning.geometry("420x180")
    warning.grab_set()

    label = tk.Label(warning,
                     text="Please read readme.md/rtf,\nAfter that, if you are sure click on Continue",
                     font=("Arial", 11), wraplength=400, justify="center")
    label.pack(pady=20)

    btn_frame = tk.Frame(warning)
    btn_frame.pack(pady=10)

    btn_c = tk.Button(btn_frame, text="Continue", bg="#4CAF50", fg="white",
                      font=("Arial", 10, "bold"), width=12,
                      command=on_continue)
    btn_c.grid(row=0, column=0, padx=10)

    btn_x = tk.Button(btn_frame, text="Cancel", bg="#F44336", fg="white",
                      font=("Arial", 10, "bold"), width=12,
                      command=on_cancel)
    btn_x.grid(row=0, column=1, padx=10)


def load_user_stories():
    show_warning_and_load(
        filetypes=[("CSV Files", "*.csv")],
        allowed_files=["row_promt.csv"]
    )


def load_user_llm():
    show_warning_and_load(
        filetypes=[("Excel Files", "*.xlsx")],
        allowed_files=["test_dataset1.xlsx", "test_dataset2.xlsx"]
    )


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

# Start maximized
try:
    root.state('zoomed')  # Windows
except:
    root.attributes('-zoomed', True)  # Linux/macOS fallback

# --- Floating Image
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
btn_frame.grid(row=1, column=1, rowspan=2, padx=20)

btn_user_stories = tk.Button(btn_frame, text="Loading User Stories", width=25,
                             state="disabled", command=load_user_stories)
btn_user_stories.pack(pady=3)

btn_user_llm = tk.Button(btn_frame, text="Loading User LLM replies", width=25,
                         state="disabled", command=load_user_llm)
btn_user_llm.pack(pady=3)

# --- Main horizontal container
main_container = tk.Frame(root)
main_container.pack(fill="both", expand=True, padx=10, pady=10)

# Left side
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
    ("6. Statistical Evaluation of Computational Methods", "python src/stat.py --stat-yaml stat.yaml --report-dir report --outdir report/Stat"),
    ("7. Image code to render", "src/06_Image_plantuml_render_to_jpg.py"),
    ("8. Select optimal prompts for human assessing", "src/07_Select_optimal_prompts_for_human_assessing.py"),
    ("9. Group merge: Image + Human assessing", "src/08_Group_MergeImage_for_selected_prompt_Humanassessing.py"),
]

for i, (label, script) in enumerate(steps):
    if script.startswith("python "):
        b = tk.Button(process_frame, text=label, anchor="w", width=65,
                      command=lambda s=script: run_script_with_args(s))
    else:
        b = tk.Button(process_frame, text=label, anchor="w", width=65,
                      command=lambda s=script: run_script(s))
    b.grid(row=i, column=0, sticky="w", pady=2)

# Controls
control_frame = tk.Frame(left_frame)
control_frame.pack(fill="x", pady=10)

btn_run_all = tk.Button(control_frame, text="🚀 Run All Pipeline",
                        width=20, height=2, relief="groove",
                        font=("Arial", 10, "bold"), bg="#4CAF50", fg="white",
                        command=run_all_pipeline)
btn_run_all.pack(side="left", padx=5)

# Two options for Analysis button - choose one:
# Option 1: Fixed command parsing
btn_analysis = tk.Button(control_frame, text="📊 Analysis",
                         width=20, height=2, relief="groove",
                         font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
                         command=lambda: run_script_with_args("python src/Analysis.py --stat-yaml stat.yaml --report-dir report --outdir report/Analysis"))

# Option 2: Direct execution (use this if Option 1 still fails)
# btn_analysis = tk.Button(control_frame, text="📊 Analysis",
#                          width=20, height=2, relief="groove",
#                          font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
#                          command=run_analysis_direct)

btn_analysis.pack(side="left", padx=5)

btn_clear_log = tk.Button(control_frame, text="🗑️ Clear Log",
                         width=15, height=2, relief="groove",
                         font=("Arial", 10, "bold"), bg="#FF9800", fg="white",
                         command=clear_log)
btn_clear_log.pack(side="left", padx=5)

progress_frame = tk.Frame(left_frame)
progress_frame.pack(fill="x", padx=20, pady=5)

progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
progress_bar.pack(fill="x")

# Right side (log)
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
