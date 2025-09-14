import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import warnings

# Default configuration
DEFAULT_EXCEL = "data/test_dataset1.xlsx"
DEFAULT_OUTPUT_DIR = r"report/jpg"
LOG_NAME = "Plot_Genaration_Error_log.csv"
ERROR_LOG_NAME = "Plot_Genaration_Error_log.csv"

# PlantUML block extraction pattern
BLOCK_RE = re.compile(r"(?is)@startuml\b.*?@enduml\b")

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)

def create_column_folder(base_dir: Path, column_name: str) -> Path:
    """Create a folder for the column and return its path"""
    # Clean column name for use as folder name
    safe_column_name = safe_name_part(column_name, 50)
    column_folder = base_dir / safe_column_name
    ensure_dir(column_folder)
    return column_folder

def guess_excel_engine(excel_path: Path) -> str:
    """Determine appropriate engine for reading Excel file"""
    suffix = excel_path.suffix.lower()
    if suffix == ".xls":
        return "xlrd"
    elif suffix in [".xlsx", ".xlsm"]:
        return "openpyxl"
    else:
        return "openpyxl"  # default

def read_excel_sheets(excel_path: Path, use_columns: Optional[List[str]]) -> List[Tuple[str, "pandas.DataFrame"]]:
    """Read all sheets from Excel file"""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    
    engine = guess_excel_engine(excel_path)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xls = pd.ExcelFile(excel_path, engine=engine)
    except ImportError as e:
        if engine == "xlrd":
            raise RuntimeError("Reading .xls files requires xlrd: pip install xlrd")
        elif engine == "openpyxl":
            raise RuntimeError("Reading .xlsx/.xlsm files requires openpyxl: pip install openpyxl")
        else:
            raise RuntimeError(f"Error reading Excel file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error opening Excel file: {e}")
    
    sheets = []
    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name=sheet_name, dtype=str)
            
            if use_columns:
                missing = [c for c in use_columns if c not in df.columns]
                if missing:
                    print(f"[WARNING] Sheet '{sheet_name}' missing columns: {missing}", file=sys.stderr)
                
                available_columns = [c for c in use_columns if c in df.columns]
                if not available_columns:
                    print(f"[WARNING] Sheet '{sheet_name}' has no requested columns", file=sys.stderr)
                    continue
                df = df[available_columns]
            
            # Convert NaN values to empty strings
            df = df.fillna("")
            sheets.append((sheet_name, df))
            print(f"[INFO] Sheet '{sheet_name}' loaded - {len(df)} rows")
            
        except Exception as e:
            print(f"[ERROR] Cannot read sheet '{sheet_name}': {e}", file=sys.stderr)
            continue
    
    return sheets

def infer_activity_syntax(code: str) -> str:
    """Detect PlantUML Activity syntax type"""
    if not code:
        return "unknown"
    
    code_lower = "\n" + code.lower() + "\n"
    legacy_score = 0
    modern_score = 0
    
    # Legacy Activity indicators
    if re.search(r"(?m)^\s*start\s*$", code_lower): 
        legacy_score += 2
    if re.search(r"(?m)^\s*stop\s*$", code_lower): 
        legacy_score += 2
    if re.search(r":[^:\n]+;\s*(\n|$)", code_lower): 
        legacy_score += 1
    if "partition " in code_lower: 
        legacy_score += 1
    
    # Modern Activity indicators
    if re.search(r"\bif\s*\([^)]+?\)\s*then\b", code_lower) and "endif" in code_lower: 
        modern_score += 2
    if "repeat" in code_lower and ("repeat while" in code_lower or "while" in code_lower): 
        modern_score += 2
    if "fork" in code_lower and ("end fork" in code_lower or "endfork" in code_lower): 
        modern_score += 2
    if "switch" in code_lower and "endswitch" in code_lower: 
        modern_score += 2
    
    if modern_score > legacy_score and modern_score > 0:
        return "activity-modern"
    elif legacy_score > modern_score and legacy_score > 0:
        return "activity-legacy"
    else:
        return "unknown"

def possible_vscode_extension_dirs() -> List[Path]:
    """Find possible VS Code extension directories"""
    paths = []
    userprofile = os.environ.get("USERPROFILE")
    
    if userprofile:
        # Standard VS Code paths
        vscode_paths = [
            Path(userprofile) / ".vscode" / "extensions",
            Path(userprofile) / ".vscode-insiders" / "extensions",
            Path(userprofile) / "AppData" / "Roaming" / "Code" / "extensions",
            Path(userprofile) / "AppData" / "Local" / "Programs" / "Microsoft VS Code" / "resources" / "app" / "extensions"
        ]
        paths.extend([p for p in vscode_paths if p.exists()])
    
    # Custom path from environment variable
    custom = os.environ.get("VSCODE_EXTENSIONS_DIR")
    if custom and Path(custom).exists():
        paths.append(Path(custom))
    
    return paths

def find_jar_in_dir(root: Path) -> List[Path]:
    """Search for JAR files in directory"""
    jars = []
    try:
        for pattern in ["plantuml*.jar", "PlantUML*.jar", "*plantuml*.jar"]:
            jars.extend(root.rglob(pattern))
        # Remove duplicates and filter valid files
        jars = list(set(jar for jar in jars if jar.is_file()))
    except (PermissionError, OSError):
        pass
    return jars

def discover_vscode_plantuml_jars() -> List[Path]:
    """Search for PlantUML JAR files in VS Code extensions"""
    jars = []
    for ext_dir in possible_vscode_extension_dirs():
        jars.extend(find_jar_in_dir(ext_dir))
    
    # Sort by modification time (newest first)
    jars.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jars

def find_plantuml_cmd(jar_override: Optional[str] = None) -> Tuple[List[str], str]:
    """Find appropriate PlantUML command"""
    
    # 1. User-specified path
    if jar_override:
        jar_path = Path(jar_override)
        if jar_path.exists():
            return (["java", "-Djava.awt.headless=true", "-jar", str(jar_path)], f"jar:{jar_path.name}")
        else:
            print(f"[WARNING] Specified JAR file not found: {jar_override}", file=sys.stderr)
    
    # 2. Environment variable
    jar_env = os.environ.get("PLANTUML_JAR", "").strip('"')
    if jar_env:
        jp = Path(jar_env)
        if jp.exists():
            return (["java", "-Djava.awt.headless=true", "-jar", str(jp)], f"jar:{jp.name}")
    
    # 3. JAR file next to script
    script_dir = Path(__file__).resolve().parent
    local_jar = script_dir / "plantuml.jar"
    if local_jar.exists():
        return (["java", "-Djava.awt.headless=true", "-jar", str(local_jar)], f"jar:{local_jar.name}")
    
    # 4. Search in VS Code extensions
    vscode_jars = discover_vscode_plantuml_jars()
    if vscode_jars:
        jar = vscode_jars[0]  # newest
        return (["java", "-Djava.awt.headless=true", "-jar", str(jar)], f"jar:{jar.name}")
    
    # 5. Command line tool
    return (["plantuml"], "cli")

def detect_engine_version(jar_override: Optional[str] = None) -> str:
    """Detect PlantUML version"""
    cmd, label = find_plantuml_cmd(jar_override)
    try:
        result = subprocess.run(
            cmd + ["-version"], 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=10
        )
        output = result.stdout or result.stderr or ""
        
        # Search for version
        version_match = re.search(r"PlantUML version\s+([^\r\n]+)", output)
        if version_match:
            return version_match.group(1).strip()
        
        return f"unknown ({label})"
    except subprocess.TimeoutExpired:
        return f"timeout ({label})"
    except Exception as e:
        return f"error ({label}): {str(e)[:50]}"

def extract_jpeg_from_buffer(buf: bytes) -> Optional[bytes]:
    """Extract JPEG from output buffer"""
    if not buf:
        return None
    
    # Search for Start of Image (SOI)
    soi_index = buf.find(b"\xFF\xD8\xFF")
    if soi_index == -1:
        return None
    
    # Search for End of Image (EOI)
    eoi_index = buf.rfind(b"\xFF\xD9")
    if eoi_index != -1 and eoi_index > soi_index:
        return buf[soi_index:eoi_index + 2]
    
    # If no EOI found, return from SOI to end of buffer
    return buf[soi_index:]

def extract_png_from_buffer(buf: bytes) -> Optional[bytes]:
    """Extract PNG from output buffer"""
    if not buf:
        return None
    
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    png_signature = b'\x89PNG\r\n\x1a\n'
    png_index = buf.find(png_signature)
    
    if png_index == -1:
        return None
    
    # PNG files end with IEND chunk
    iend_marker = b'IEND\xaeB`\x82'
    iend_index = buf.find(iend_marker, png_index)
    
    if iend_index != -1:
        return buf[png_index:iend_index + 8]  # Include IEND marker
    
    # If no IEND found, return from PNG signature to end
    return buf[png_index:]

def convert_png_to_jpeg(png_data: bytes) -> bytes:
    """Convert PNG data to JPEG format"""
    try:
        from PIL import Image
        import io
        
        # Load PNG from bytes
        png_image = Image.open(io.BytesIO(png_data))
        
        # Convert to RGB (in case it has transparency)
        if png_image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            rgb_image = Image.new('RGB', png_image.size, (255, 255, 255))
            if png_image.mode == 'P':
                png_image = png_image.convert('RGBA')
            rgb_image.paste(png_image, mask=png_image.split()[-1] if png_image.mode in ('RGBA', 'LA') else None)
            png_image = rgb_image
        elif png_image.mode != 'RGB':
            png_image = png_image.convert('RGB')
        
        # Save as JPEG
        jpg_buffer = io.BytesIO()
        png_image.save(jpg_buffer, format='JPEG', quality=90, optimize=True)
        return jpg_buffer.getvalue()
        
    except ImportError:
        raise RuntimeError(
            "PNG to JPEG conversion requires Pillow. Install with: pip install Pillow"
        )
    except Exception as convert_error:
        raise RuntimeError(f"Failed to convert PNG to JPEG: {convert_error}")

def render_plantuml_to_jpg(diagram_code: str, timeout_sec: int = 120, jar_override: Optional[str] = None) -> bytes:
    """Render PlantUML code to JPG format with PNG fallback"""
    cmd, engine_info = find_plantuml_cmd(jar_override)
    
    # Try JPG first
    jpg_cmd = cmd + [
        "-pipe", 
        "-tjpg", 
        "-charset", "UTF-8",
        "-Djava.awt.headless=true"
    ]
    
    try:
        # First attempt: JPG
        process = subprocess.run(
            jpg_cmd,
            input=diagram_code.encode("utf-8"),
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        
        stdout_data = process.stdout or b""
        
        # Check for JPEG signature
        jpeg_data = extract_jpeg_from_buffer(stdout_data)
        if jpeg_data and len(jpeg_data) > 100:
            return jpeg_data
        
        # If JPG failed, try PNG and convert
        print(f"[INFO] JPG generation failed, trying PNG format...")
        
        png_cmd = cmd + [
            "-pipe", 
            "-tpng", 
            "-charset", "UTF-8",
            "-Djava.awt.headless=true"
        ]
        
        process = subprocess.run(
            png_cmd,
            input=diagram_code.encode("utf-8"),
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        
        stdout_data = process.stdout or b""
        
        # Check for PNG signature and convert to JPEG
        png_data = extract_png_from_buffer(stdout_data)
        if png_data and len(png_data) > 100:
            try:
                jpeg_data = convert_png_to_jpeg(png_data)
                if jpeg_data and len(jpeg_data) > 100:
                    return jpeg_data
            except Exception as e:
                print(f"[WARNING] PNG to JPEG conversion failed: {e}")
                # Fall back to returning PNG data
                return png_data
        
    except FileNotFoundError:
        raise RuntimeError(
            "PlantUML not found. If using VS Code extension, "
            "ensure Java is installed (java -version)."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"PlantUML render timeout after {timeout_sec} seconds.")
    
    # If both attempts failed, provide detailed error
    stderr_data = process.stderr or b""
    
    def clean_text(text: bytes, limit: int = 500) -> str:
        if not text:
            return ""
        decoded = text.decode("utf-8", "replace").strip()
        decoded = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]+", " ", decoded)
        return (decoded[:limit] + " ...<truncated>") if len(decoded) > limit else decoded
    
    error_msg = "PlantUML could not generate valid image."
    
    stderr_text = clean_text(stderr_data)
    if stderr_text:
        error_msg += f" Error: {stderr_text}"
    
    if process.returncode != 0:
        error_msg += f" Exit code: {process.returncode}"
    
    raise RuntimeError(error_msg)

def normalize_blocks(text: str) -> List[str]:
    """Extract and normalize PlantUML blocks"""
    if not text or not isinstance(text, str):
        return []
    
    # Search for complete blocks
    blocks = [match.group(0) for match in BLOCK_RE.finditer(text)]
    
    if blocks:
        return blocks
    
    # If no complete block found, wrap text in block
    stripped = text.strip()
    if stripped:
        # Check if already has PlantUML tags
        if not stripped.lower().startswith('@startuml') and not stripped.lower().endswith('@enduml'):
            return [f"@startuml\n{stripped}\n@enduml\n"]
        else:
            return [stripped]
    
    return []

def safe_name_part(text: str, max_length: int = 50) -> str:
    """Generate safe filename"""
    if not text:
        return "unknown"
    
    # Remove invalid characters and replace with underscore
    safe_text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    # Remove multiple underscores
    safe_text = re.sub(r"_+", "_", safe_text)
    # Trim and remove leading/trailing underscores
    safe_text = safe_text[:max_length].strip("_")
    
    return safe_text if safe_text else "unknown"

def generate_unique_name(sheet_name: str, row_index: int, col_index: int, diagram_index: int, code: str, column_name: str) -> str:
    """Generate unique filename using column name and row number"""
    # Don't include column name in filename since it's now in the folder name
    # Generate hash from code for uniqueness
    code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()[:6]
    
    # Create filename: Row[X]_[optional_diagram_index]_[hash]
    if diagram_index > 1:
        name = f"Row{row_index}_D{diagram_index}_{code_hash}"
    else:
        name = f"Row{row_index}_{code_hash}"
    
    return name

def write_log_entry(writer, **kwargs):
    """Write entry to log file"""
    row = [
        kwargs.get('timestamp', ''),
        kwargs.get('excel_file', ''),
        kwargs.get('sheet', ''),
        kwargs.get('row_index', ''),
        kwargs.get('col_index', ''),
        kwargs.get('column_name', ''),
        kwargs.get('diagram_index', ''),
        kwargs.get('status', ''),
        kwargs.get('error', ''),
        kwargs.get('engine_version', ''),
        kwargs.get('inferred_syntax', ''),
        kwargs.get('output_path', ''),
        kwargs.get('output_format', ''),
    ]
    writer.writerow(row)

def process_excel(
    excel_path: Path,
    output_dir: Path,
    use_columns: Optional[List[str]] = None,
    jar_override: Optional[str] = None,
) -> None:
    """Main Excel processing function"""
    
    print(f"[START] Processing file: {excel_path}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Create main output directory
    ensure_dir(output_dir)
    
    # Log file paths (in main directory)
    log_path = output_dir / LOG_NAME
    error_log_path = output_dir / ERROR_LOG_NAME
    
    # Detect PlantUML version
    engine_version = detect_engine_version(jar_override)
    print(f"[INFO] PlantUML engine: {engine_version}")
    
    # Processing statistics
    stats = {
        'total_cells': 0,
        'cells_with_content': 0,
        'total_diagrams': 0,
        'successful_renders': 0,
        'failed_renders': 0,
        'jpeg_outputs': 0,
        'png_outputs': 0,
        'column_folders_created': set()
    }
    
    try:
        # Read Excel sheets
        sheets = read_excel_sheets(excel_path, use_columns)
        
        if not sheets:
            raise RuntimeError("No processable sheets found in Excel file")
        
        # Open log files
        with open(log_path, "w", newline="", encoding="utf-8") as log_file, \
             open(error_log_path, "w", newline="", encoding="utf-8") as error_file:
            
            # CSV writers
            log_writer = csv.writer(log_file)
            error_writer = csv.writer(error_file)
            
            # Log file headers
            headers = [
                "timestamp_iso", "excel_file", "sheet", "row_index_1based", 
                "col_index_1based", "column_name", "diagram_index_in_cell_1based",
                "status", "error", "engine_version", "inferred_syntax", "output_path", "output_format"
            ]
            log_writer.writerow(headers)
            error_writer.writerow(headers + ["error_details"])
            
            # Process each sheet
            for sheet_name, dataframe in sheets:
                print(f"[PROCESSING] Sheet: {sheet_name} ({len(dataframe)} rows)")
                
                # Process each cell
                for row_idx in range(len(dataframe)):
                    row_number = row_idx + 1
                    
                    for col_idx, column_name in enumerate(dataframe.columns):
                        col_number = col_idx + 1
                        stats['total_cells'] += 1
                        
                        # Get cell content
                        cell_content = dataframe.iat[row_idx, col_idx]
                        
                        # Validate content
                        if not isinstance(cell_content, str) or not cell_content.strip():
                            continue
                        
                        stats['cells_with_content'] += 1
                        content = cell_content.strip()
                        
                        # Extract PlantUML blocks
                        blocks = normalize_blocks(content)
                        if not blocks:
                            continue
                        
                        # Create column folder if not exists
                        column_folder = create_column_folder(output_dir, column_name)
                        stats['column_folders_created'].add(column_name)
                        
                        # Process each block
                        for diagram_idx, block in enumerate(blocks, start=1):
                            stats['total_diagrams'] += 1
                            timestamp = dt.datetime.now().isoformat(timespec="seconds")
                            
                            # Detect syntax type
                            syntax_type = infer_activity_syntax(block)
                            
                            # Generate filename (without column name since it's in folder name)
                            base_name = generate_unique_name(
                                sheet_name, row_number, col_number, diagram_idx, block, column_name
                            )
                            # Save file in column-specific folder
                            output_file_path = column_folder / f"{base_name}.jpg"
                            
                            # Prepare log data
                            log_data = {
                                'timestamp': timestamp,
                                'excel_file': str(excel_path),
                                'sheet': sheet_name,
                                'row_index': row_number,
                                'col_index': col_number,
                                'column_name': column_name,
                                'diagram_index': diagram_idx,
                                'engine_version': engine_version,
                                'inferred_syntax': syntax_type,
                                'output_format': 'JPEG'
                            }
                            
                            try:
                                # Render diagram
                                image_data = render_plantuml_to_jpg(block, jar_override=jar_override)
                                
                                # Determine actual format of output
                                if image_data.startswith(b'\xFF\xD8\xFF'):
                                    actual_format = 'JPEG'
                                    stats['jpeg_outputs'] += 1
                                elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                                    actual_format = 'PNG'
                                    stats['png_outputs'] += 1
                                    # Change file extension for PNG
                                    output_file_path = column_folder / f"{base_name}.png"
                                else:
                                    actual_format = 'Unknown'
                                
                                # Save file
                                with open(output_file_path, "wb") as img_file:
                                    img_file.write(image_data)
                                
                                # Log success
                                log_data.update({
                                    'status': 'success',
                                    'error': '',
                                    'output_path': str(output_file_path),
                                    'output_format': actual_format
                                })
                                write_log_entry(log_writer, **log_data)
                                stats['successful_renders'] += 1
                                
                                print(f"[SUCCESS] {column_name}/{output_file_path.name} ({actual_format})")
                                
                            except Exception as render_error:
                                # Log error
                                error_message = str(render_error)[:500]  # Limit error length
                                
                                log_data.update({
                                    'status': 'error',
                                    'error': error_message,
                                    'output_path': '',
                                    'output_format': 'N/A'
                                })
                                write_log_entry(log_writer, **log_data)
                                
                                # Log to error file with more details
                                error_data = list(log_data.values()) + [str(render_error)]
                                error_writer.writerow(error_data)
                                
                                stats['failed_renders'] += 1
                                print(f"[ERROR] {column_name}/{base_name}: {error_message}")
    
    except Exception as e:
        print(f"[FATAL ERROR] {e}", file=sys.stderr)
        raise
    
    finally:
        # Final report
        print(f"\n[SUMMARY]")
        print(f"Total cells: {stats['total_cells']}")
        print(f"Cells with content: {stats['cells_with_content']}")
        print(f"Total diagrams: {stats['total_diagrams']}")
        print(f"Successful renders: {stats['successful_renders']}")
        print(f"  - JPEG outputs: {stats['jpeg_outputs']}")
        print(f"  - PNG outputs: {stats['png_outputs']}")
        print(f"Failed renders: {stats['failed_renders']}")
        print(f"Column folders created: {len(stats['column_folders_created'])}")
        print(f"  - Folders: {', '.join(sorted(stats['column_folders_created']))}")
        print(f"Success rate: {(stats['successful_renders'] / max(stats['total_diagrams'], 1)) * 100:.1f}%")
        print(f"\n[DONE] Image files saved to column folders in: {output_dir}")
        print(f"[LOG] Main log: {log_path}")
        print(f"[LOG] Error log: {error_log_path}")

def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert PlantUML diagrams from Excel file to images with column-based folder organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s --excel data.xlsx --output ./output
  %(prog)s --excel data.xlsx --columns "UML_Code" "Diagram" --jar plantuml.jar
  %(prog)s --excel data.xlsx --output ./diagrams --columns "Code"
  
File organization:
  report/jpg/
  ├── ColumnName1/
  │   ├── Row1_abc123.jpg
  │   ├── Row2_def456.jpg
  │   └── ...
  ├── ColumnName2/
  │   ├── Row1_ghi789.jpg
  │   └── ...
  ├── log.csv
  └── error_log.csv
        """
    )
    
    parser.add_argument(
        "--excel", 
        default=DEFAULT_EXCEL,
        help=f"Path to Excel file (.xls/.xlsx) - default: {DEFAULT_EXCEL}"
    )
    
    parser.add_argument(
        "--output", 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder for image files and logs - default: {DEFAULT_OUTPUT_DIR}"
    )
    
    parser.add_argument(
        "--columns", 
        nargs="*", 
        default=None,
        help="List of column names to process (optional)"
    )
    
    parser.add_argument(
        "--jar", 
        default=None,
        help="Explicit path to plantuml.jar file (overrides auto-discovery)"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="PlantUML Excel Processor v2.3"
    )
    
    return parser.parse_args(argv)

def main():
    """Main program function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Convert paths
        excel_path = Path(args.excel).resolve()
        output_dir = Path(args.output)
        
        # Check if Excel file exists
        if not excel_path.exists():
            print(f"[ERROR] Excel file not found: {excel_path}", file=sys.stderr)
            sys.exit(2)
        
        # Check file format
        if excel_path.suffix.lower() not in ['.xls', '.xlsx', '.xlsm']:
            print(f"[ERROR] Unsupported file format: {excel_path.suffix}", file=sys.stderr)
            print("Supported formats: .xls, .xlsx, .xlsm", file=sys.stderr)
            sys.exit(2)
        
        # Start processing
        process_excel(excel_path, output_dir, args.columns, args.jar)
        
    except KeyboardInterrupt:
        print("\n[CANCELLED] Processing stopped by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
