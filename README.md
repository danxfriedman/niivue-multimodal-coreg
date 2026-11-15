# NeuroCoreg Studio

**Single-Subject Multimodal Brain Image Visualization, Skull-Stripping, DKT Parcellation & 3D Coregistration Tool**  
Built on [`@niivue/niivue`](https://github.com/niivue/niivue) (`niivue/mono`) and Python (`nibabel`, `scipy`, `numpy`, `pydicom`), with seamless integration for **FSL** (`flirt`, `bet`), **FastSurfer / FreeSurfer** (`aparc.DKTatlas+aseg`), and **`dcm2niix`**.

---

## Key Features

- **Multimodal NIfTI, FreeSurfer & DICOM Workspace**:
  - Load `.nii`, `.nii.gz`, FreeSurfer `.mgz` / `.mgh`, and raw DICOM series folders (`.dcm`).
  - Assign **Reference (`REF`)**, **Moving (`MOV`)**, and live-linked **Child (`CHILD`)** volumes (e.g., SEEG electrode contact masks or ROI overlays that automatically inherit Moving transforms).
- **Interactive & Automated 3D Coregistration**:
  - Manual 9-DOF sliders (translation, rotation, isotropic/anisotropic scaling) with live 4×4 World RAS & FSL FLIRT `.mat` matrix editing and **Undo / Clear Transform**.
  - Automated multimodal registration via **SciPy Powell (NMI / NCC)** or **FSL FLIRT** (6-DOF Rigid, 9-DOF Traditional, 12-DOF Affine) with pure-Python fallbacks on all operating systems.
  - Real-time alignment verification tools: **Blink Comparator (`Space`)**, **Cross-Fade Blend**, **3D Sobel Edge Contours**, **3D Checkerboard Fusion**, and **3D Red/Cyan Anaglyph**, plus live **NMI**, **|NCC|**, and **3D Edge Overlap** metrics.
- **Brain Extraction (Skull-Stripping)**:
  - One-click **FSL BET** skull-stripping with adjustable fractional intensity threshold (`-f`), plus an automatic pure-Python 3D morphological skull-stripping fallback when FSL is not installed.
- **FastSurfer / FreeSurfer DKT Anatomical Atlas & Hover Inspector**:
  - Segment T1w structural MRIs into the 104-region **Desikan–Killiany–Tourville (`aparc.DKTatlas+aseg`)** cortical and subcortical parcellation using **FastSurferVINN**, **Full FastSurfer**, or **Template Registration**.
  - Automatic brain-masking and 3D-safe `FreeSurferColorLUT` rendering (areas outside the brain are set to `0` and 100% transparent so they never obscure the 3D volume render), with dedicated opacity and windowing (`cal_min` / `cal_max`) controls and live cursor-hover region readout.
- **Project Save / Open & Network Share Support**:
  - Save and reopen complete patient workspaces (NIfTI volumes, transforms, layer roles, opacity/windowing settings, and metrics) to local folders or mounted network drives (`SMB`, `NFS`, `CIFS`, Windows UNC paths `\\server\share`, macOS `/Volumes/...`, Linux `/mnt/...`).
  - Runs entirely in user space with **zero administrative (`sudo` / Administrator) privileges required**.

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

#### Step 1: Install Python 3 & Recommended Tools (Homebrew)
If you do not have [Homebrew](https://brew.sh) installed, install it first or use the official Python installer from [python.org](https://www.python.org/downloads/macos/):
```bash
# Install Python 3 and dcm2niix
brew install python dcm2niix
```

#### Step 2: (Recommended) Install FSL (`bet` & `flirt`)
Download and run the official FSL installer script from [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/macos):
```bash
python3 fslinstaller.py --dest "$HOME/fsl"
```
*(Installing into `$HOME/fsl` requires no `sudo`/admin privileges; NeuroCoreg Studio automatically detects both `$HOME/fsl` and `/usr/local/fsl`.)*

#### Step 3: (Recommended) Install FastSurfer
- **Native macOS bundle**: Extract FastSurfer into `/Applications/FastSurfer*` or `~/FastSurfer*` (with `macos_setup_fastsurfer.sh`). `start.sh` automatically discovers and sources it.
- **Or via Docker Desktop**:
  ```bash
  docker pull deepmi/fastsurfer:latest
  ```

#### Step 4: Launch NeuroCoreg Studio
```bash
git clone https://github.com/dfriedman36/niivue-multimodal-coreg.git
cd niivue-multimodal-coreg
chmod +x start.sh
./start.sh
```
Then open **http://127.0.0.1:8765** in your browser.

---

### Linux (Ubuntu / Debian, Fedora / RHEL, Arch)

#### Step 1: Install Python 3, `venv`, and `dcm2niix`
- **Ubuntu / Debian**:
  ```bash
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip dcm2niix git
  ```
- **Fedora / RHEL / Rocky Linux**:
  ```bash
  sudo dnf install -y python3 python3-pip dcm2niix git
  ```
- **Arch Linux**:
  ```bash
  sudo pacman -S python python-pip dcm2niix git
  ```
*(If you do not have `sudo` access on a shared workstation, you can use Miniconda/Miniforge in `$HOME/miniforge3`—no root access is needed.)*

#### Step 2: (Recommended) Install FSL (`bet` & `flirt`)
Download `fslinstaller.py` from [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/linux) and install into your home directory or `/usr/local/fsl`:
```bash
python3 fslinstaller.py --dest "$HOME/fsl"
```

#### Step 3: (Recommended) Install FastSurfer
- **Via Docker**:
  ```bash
  docker pull deepmi/fastsurfer:latest
  ```
- **Or Native Conda/Pip Installation**: Clone [DeepMI/FastSurfer](https://github.com/DeepMI/FastSurfer) into `~/FastSurfer` or `/opt/fastsurfer` and set `export FASTSURFER_HOME="$HOME/FastSurfer"`.

#### Step 4: Launch NeuroCoreg Studio
```bash
git clone https://github.com/dfriedman36/niivue-multimodal-coreg.git
cd niivue-multimodal-coreg
chmod +x start.sh
./start.sh
```
Then open **http://127.0.0.1:8765** in your browser.

---

### Windows (Windows 10 / 11 — Native & WSL2)

#### Option A: Native Windows 10 / 11 (No Administrator Privileges Required)

1. **Install Python 3.10+**:
   - Download the **Windows Installer (64-bit)** from [python.org/downloads/windows](https://www.python.org/downloads/windows/) (check **"Add python.exe to PATH"**; standard per-user installation requires no Administrator elevation), **or** run via `winget`:
     ```cmd
     winget install -e --id Python.Python.3.12 --scope user
     ```
2. **(Optional) Install `dcm2niix`**:
   - Download the latest Windows release (`dcm2niix_win.zip`) from [github.com/rordenlab/dcm2niix/releases](https://github.com/rordenlab/dcm2niix/releases) and place `dcm2niix.exe` in the project folder or on your user `PATH`.
3. **(Optional) Install FastSurfer via Docker Desktop**:
   ```cmd
   docker pull deepmi/fastsurfer:latest
   ```
4. **Launch NeuroCoreg Studio**:
   - Double-click **`start.bat`** in File Explorer, or run in Command Prompt / PowerShell:
     ```cmd
     git clone https://github.com/dfriedman36/niivue-multimodal-coreg.git
     cd niivue-multimodal-coreg
     start.bat
     ```
   - `start.bat` automatically creates a user-space virtual environment in `%LOCALAPPDATA%\neurocoreg-venv`, installs `requirements.txt`, and starts the server on **http://127.0.0.1:8765**.

#### Option B: Windows Subsystem for Linux (WSL2 — Recommended if using FSL on Windows)
FSL runs natively inside WSL2 Ubuntu on Windows 10/11:
```bash
wsl --install -d Ubuntu
```
Inside the Ubuntu WSL2 terminal, follow the **Linux (Ubuntu / Debian)** instructions above (`./start.sh`), and access **http://127.0.0.1:8765** directly from your Windows browser.

---

## Manual Python Virtual Environment Setup (All OS)

If you prefer to create and manage the Python environment manually instead of using `start.sh` / `start.bat`:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 server.py
```

```cmd
:: Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python server.py
```

### Custom Port or Custom Python Interpreter
- **Custom Port** (default `8765`):
  ```bash
  PORT=9000 ./start.sh
  ```
- **Custom Python Binary**:
  ```bash
  NEUROCOREG_PYTHON=/path/to/python3 ./start.sh
  ```

---

## Project Structure

```text
niivue-multimodal-coreg/
├── index.html               # Main workstation UI & modals
├── styles.css               # Responsive dark-mode neuroimaging UI styling
├── app.js                   # Frontend NiiVue 3D coregistration, DKT LUT & workspace engine
├── server.py                # Cross-platform Python HTTP & neuroimaging API server
├── generate_sample_data.py  # Generates synthetic sample multimodal NIfTI & DICOM datasets
├── requirements.txt         # Python dependencies (numpy, scipy, nibabel, pydicom)
├── start.sh                 # macOS / Linux zero-admin launcher script
├── start.bat                # Windows zero-admin launcher script
└── sample_data/
    ├── dkt_lut.json         # FreeSurfer / FastSurfer DKT lookup table
    ├── nifti/               # Sample T1w, misaligned CT, SEEG mask, FLAIR, DTI FA, DKT & MNI templates
    └── dicom_t2_flair/      # Sample 91-slice axial T2-FLAIR DICOM series
```
