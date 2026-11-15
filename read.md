# NeuroCoreg Studio — Installation & Prerequisites Guide

*(This file mirrors [`README.md`](./README.md) for convenience.)*

**Single-Subject Multimodal Brain Image Visualization, Skull-Stripping, DKT Parcellation & 3D Coregistration Tool**  
Built on [`@niivue/niivue`](https://github.com/niivue/niivue) (`niivue/mono`) and Python (`nibabel`, `scipy`, `numpy`, `pydicom`), with seamless integration for **FSL** (`flirt`, `bet`), **FastSurfer / FreeSurfer** (`aparc.DKTatlas+aseg`), and **`dcm2niix`**.

---

## System Requirements & Recommended Prerequisites

### 1. Required Prerequisites (All Operating Systems)
| Component | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | **3.9 – 3.12+** (with `venv` & `pip`) | Runs the local HTTP & neuroimaging API server (`server.py`) and pure-Python SciPy/NiBabel pipelines |
| **Python Packages** | `numpy>=1.22`, `scipy>=1.9`, `nibabel>=4.0`, `pydicom>=2.3` | Automatically installed in user space by `start.sh` / `start.bat` (or via `pip install -r requirements.txt`) |
| **Modern Web Browser** | Chrome 113+, Edge 113+, Firefox 115+, or Safari 17+ | WebGL2 / WebGPU hardware-accelerated 2D multiplanar & 3D volume rendering |

### 2. Recommended Optional Neuroimaging Tools
> **Note:** NeuroCoreg Studio includes built-in **pure-Python (`SciPy` / `NiBabel` / `pydicom`) fallbacks** for every pipeline. Installing the tools below enables native CLI acceleration and deep-learning parcellation:

| Tool | Purpose in NeuroCoreg Studio | Built-in Fallback if Not Installed |
| :--- | :--- | :--- |
| **FSL (`bet` & `flirt`)** | Fast CLI skull-stripping (`bet`) and 6/9/12-DOF registration (`flirt`) | Pure-Python SciPy 3D morphological brain extraction & Powell NMI/NCC registration |
| **FastSurfer (`2.x` / Docker)** | Deep-learning 104-region DKT cortical/subcortical parcellation (`FastSurferVINN`) | MNI152 DKT Atlas 12-DOF registration (`FLIRT` or `SciPy`) + cached sample parcellation |
| **`dcm2niix`** | High-speed C++ DICOM-to-NIfTI series conversion | In-browser WebAssembly (`@niivue/nv-ext-dcm2niix`) + pure-Python `pydicom` converter |

---

## Installation Guide by Operating System

### macOS (Apple Silicon `M1/M2/M3/M4` & Intel)

1. **Install Python 3 & `dcm2niix` (via Homebrew)**:
   ```bash
   brew install python dcm2niix
   ```
2. **(Recommended) Install FSL (`bet` & `flirt`)**:
   Download `fslinstaller.py` from [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/macos) and run:
   ```bash
   python3 fslinstaller.py --dest "$HOME/fsl"
   ```
3. **(Recommended) Install FastSurfer**:
   Extract FastSurfer into `/Applications/FastSurfer*` or `~/FastSurfer*`, or pull the Docker image:
   ```bash
   docker pull deepmi/fastsurfer:latest
   ```
4. **Launch NeuroCoreg Studio**:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
   Open **http://127.0.0.1:8765** in your browser.

---

### Linux (Ubuntu / Debian, Fedora / RHEL, Arch)

1. **Install Python 3, `venv`, and `dcm2niix`**:
   - **Ubuntu / Debian**:
     ```bash
     sudo apt update && sudo apt install -y python3 python3-venv python3-pip dcm2niix git
     ```
   - **Fedora / RHEL**:
     ```bash
     sudo dnf install -y python3 python3-pip dcm2niix git
     ```
   - **Arch Linux**:
     ```bash
     sudo pacman -S python python-pip dcm2niix git
     ```
2. **(Recommended) Install FSL (`bet` & `flirt`)**:
   Download `fslinstaller.py` from [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/linux) and run:
   ```bash
   python3 fslinstaller.py --dest "$HOME/fsl"
   ```
3. **(Recommended) Install FastSurfer**:
   ```bash
   docker pull deepmi/fastsurfer:latest
   ```
4. **Launch NeuroCoreg Studio**:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
   Open **http://127.0.0.1:8765** in your browser.

---

### Windows (Windows 10 / 11 — Native & WSL2)

#### Option A: Native Windows 10 / 11 (No Administrator Privileges Required)
1. **Install Python 3.10+**:
   Download the Windows 64-bit installer from [python.org/downloads/windows](https://www.python.org/downloads/windows/) (check **"Add python.exe to PATH"**) or run:
   ```cmd
   winget install -e --id Python.Python.3.12 --scope user
   ```
2. **(Optional) Install `dcm2niix` & FastSurfer**:
   - Download `dcm2niix_win.zip` from [github.com/rordenlab/dcm2niix/releases](https://github.com/rordenlab/dcm2niix/releases) and place `dcm2niix.exe` on your `PATH`.
   - For FastSurfer, install Docker Desktop and run `docker pull deepmi/fastsurfer:latest`.
3. **Launch NeuroCoreg Studio**:
   Double-click **`start.bat`** or run:
   ```cmd
   start.bat
   ```
   Open **http://127.0.0.1:8765** in your browser.

#### Option B: Windows Subsystem for Linux (WSL2 — Recommended for FSL on Windows)
```powershell
wsl --install -d Ubuntu
```
Inside Ubuntu WSL2, follow the **Linux** steps above (`./start.sh`) and open **http://127.0.0.1:8765** in your Windows browser.
