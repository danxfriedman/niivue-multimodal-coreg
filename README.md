# NeuroCoreg Studio

**Single-Subject Multimodal Brain Visualization, 2D Zoom/Pan, Drawing/VOI Studio, Multi-Atlas Parcellation (DKT, Destrieux, THOMAS / HIPS-THOMAS, Tian, FSL), Skull-Stripping & 3D Coregistration**  
Built on [`@niivue/niivue`](https://github.com/niivue/niivue) (`niivue/mono`) and Python (`nibabel`, `scipy`, `numpy`, `pydicom`), with seamless integration for **HIPS-THOMAS** (`anagrammarian/sthomas`), **FSL** (`bet`, `flirt`, `fnirt`, `applywarp`, `invwarp`, `$FSLDIR/data/atlases`), **FastSurfer / FreeSurfer**, and **`dcm2niix`**.

---

## Key Features

- **Multimodal NIfTI, FreeSurfer & DICOM Workspace**:
  - Load `.nii`, `.nii.gz`, FreeSurfer `.mgz` / `.mgh`, and raw DICOM series folders (`.dcm`).
  - Assign **Reference (`REF`)**, **Moving (`MOV`)**, and live-linked **Child (`CHILD`)** volumes (e.g., SEEG electrode contact masks or ROI overlays that automatically inherit Moving transforms).
- **2D Slice Zoom & Pan + Non-Blocking Pop-Out Windows**:
  - Scroll-wheel / pinch zoom (`1.0×–10.0×`) and Space-drag / middle-click / trackpad pan on 2D orthogonal views, with a floating zoom/reset HUD pill (`Ctrl/Cmd + 0` reset, `F` fit).
  - Non-blocking, draggable, minimizable floating pop-out tool windows so the 2D/3D brain canvas stays interactive while editing ROIs or browsing atlases.
- **Interactive & Automated 3D Coregistration**:
  - Manual 9-DOF sliders (translation, rotation, isotropic/anisotropic scaling) with live 4×4 World RAS & FSL FLIRT `.mat` matrix editing and **Undo / Clear Transform**.
  - Automated multimodal registration via **SciPy Powell (NMI / NCC)** or **FSL FLIRT** (6-DOF Rigid, 9-DOF Traditional, 12-DOF Affine) with pure-Python fallbacks on all operating systems.
  - Real-time alignment verification tools: **Blink Comparator (`Space`)**, **Cross-Fade Blend**, **3D Sobel Edge Contours**, **3D Checkerboard Fusion**, and **3D Red/Cyan Anaglyph**, plus live **NMI**, **|NCC|**, and **3D Edge Overlap** metrics.
- **Drawing & VOI Studio (`D`)**:
  - Create multiple discrete Drawing Layers with 2D/3D circular brush (`1–5` voxels), closed filled polygon pen, eraser, 2D flood fill, and 2D/3D intensity region-grow (`T`/`O` overwrite guard, sparse undo/redo stack).
  - 3D morphological operations (fill holes, keep largest island, dilate, erode, linear slice interpolation) and per-layer voxel/volume statistics.
  - Export and import MRIcroGL `.voi` masks, `.nii` / `.nii.gz` labels, and companion label metadata via native browser file pickers (`Chrome` / `Edge`), OS save dialogs, or browser downloads.
- **Extensible Multi-Atlas Registry & Floating Brain Atlas Studio (`A`)**:
  - **Cortical & Whole-Brain**: 104-region **DKT (`aparc.DKTatlas+aseg`)** (`FastSurferVINN --seg_only`) and 148-region **Destrieux (`aparc.a2009s+aseg`)** (`FastSurfer --fsaparc`, with FreeSurfer license validation/picker), or cached MNI152 FNIRT/FLIRT template registration fallbacks.
  - **THOMAS Thalamic Nuclei (28 Bilateral Nuclei)**: Automatically defaults to containerized **HIPS-THOMAS (`anagrammarian/sthomas`)** when Docker/Podman or native `hipsthomas.sh` is available for maximal native subject accuracy — supporting **Standard T1w (`-t1`)** vs **WMn / FGATIR** contrast selection, automatic **`-dm`** detection for skull-stripped inputs, strict 28-nuclei thalamic filtering, async background execution with live stage/log polling + **Cancel** (`docker rm -f`), optional custom Docker image override, per-T1w SHA-256 caching (**Force Recompute** bypass), and automatic/selectable **FNIRT nonlinear** & **FLIRT 12-DOF** backups.
  - **Tian Melbourne Subcortex (Scales I–IV)**: Bundled 1 mm MNI152 subcortical parcellations (16, 32, 50, and 54 bilateral ROIs), warped to subject space with per-T1w FNIRT/FLIRT warp caching.
  - **Local FSL Atlases & Custom Importers**: Auto-discovers `$FSLDIR/data/atlases` (Harvard-Oxford Cortical/Subcortical, JHU White-Matter, Cerebellum, STN, etc.), imports custom subject or MNI152 atlas volumes with multi-format LUTs (`.json`, `.csv`, `.tsv`, `.txt`, `.ctbl`, FSL `.xml`), and merges pre-computed HIPS-THOMAS files or folders.
  - **Interactive Inspection & Editing**: Stacked multi-atlas cursor/crosshair readouts across all loaded atlases, **Filled / +Edge / Edge** overlay styles (`intent_code = 1002`), ROI search, `L/R` filtering, per-ROI visibility checkboxes, click-to-jump centroids, and one-click **Copy Selected ROI(s) to Drawing Layer (`.voi`)**.
- **Brain Extraction (Skull-Stripping)**:
  - One-click **FSL BET** skull-stripping with adjustable fractional intensity threshold (`-f`), plus an automatic pure-Python 3D morphological skull-stripping fallback when FSL is not installed.
- **Project Save / Open & Network Share Support**:
  - Save and reopen complete patient workspaces (NIfTI volumes, transforms, layer roles, drawing layers `drawings/*.voi`, loaded multi-atlases, custom LUTs, ROI isolation sets, overlay styles, opacity/windowing settings, and metrics) to local folders or mounted network drives (`SMB`, `NFS`, `CIFS`, Windows UNC paths `\\server\share`, macOS `/Volumes/...`, Linux `/mnt/...`).
  - Runs entirely in user space with **zero administrative (`sudo` / Administrator) privileges required** (aside from standard Docker installation if using containerized tools).

---

## System Requirements & Prerequisites

### 1. Required Prerequisites (All Operating Systems)
| Component | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | **3.9 – 3.12+** (with `venv` & `pip`) | Runs the local HTTP & neuroimaging API server (`server.py`) and pure-Python SciPy/NiBabel pipelines |
| **Python Packages** | `numpy>=1.22.0`, `scipy>=1.9.0`, `nibabel>=4.0.0`, `pydicom>=2.3.0` | Listed in [`requirements.txt`](./requirements.txt); automatically installed in user space by `start.sh` / `start.bat` |
| **Modern Web Browser** | Chrome 113+, Edge 113+, Firefox 115+, or Safari 17+ | WebGL2 / WebGPU 2D & 3D rendering (`Chrome`/`Edge` recommended for native file save pickers; Firefox/Safari fall back to OS dialogs or browser downloads) |

> **No new Python pip dependencies** were added beyond [`requirements.txt`](./requirements.txt). Atlas downloading ([`scripts/fetch_phase3_atlases.py`](./scripts/fetch_phase3_atlases.py)) is only needed if regenerating bundled MNI152 assets (already included under `sample_data/atlases/`).

### 2. Optional Neuroimaging Packages & Containers
> **Note:** NeuroCoreg Studio works out-of-the-box using built-in **pure-Python (`SciPy` / `NiBabel` / `pydicom`) fallbacks** and bundled MNI152 templates (`sample_data/nifti/` and `sample_data/atlases/`). Installing the optional tools below unlocks native/nonlinear acceleration and deep-learning / multi-atlas segmentation:

| Optional Tool / Package | Purpose in NeuroCoreg Studio | Prerequisites / Notes | Built-in Fallback if Not Installed |
| :--- | :--- | :--- | :--- |
| **HIPS-THOMAS Container (`anagrammarian/sthomas`)** *(or native `hipsthomas.sh`)* | **Default** high-accuracy subject-space parcellation of the 28 bilateral **THOMAS thalamic nuclei** (`-t1`, native WMn/FGATIR, auto-`-dm`, async progress & Cancel, per-T1w cache) | Requires **Docker Desktop / Docker Engine** (or **Podman**) running with `$HOME` file sharing enabled (~5–15 GB image + RAM); on Apple Silicon Macs (`arm64`), enable Docker Desktop Rosetta/amd64 emulation if pulling an `amd64` image | Bundled 1 mm MNI152 THOMAS atlas warped via **FSL FNIRT / FLIRT** (or SciPy 9-DOF), plus manual HIPS-THOMAS NIfTI/folder import |
| **Docker Desktop / Docker Engine** *(or Podman)* | Container runtime required only if running containerized **HIPS-THOMAS** or containerized **FastSurfer** | Docker daemon must be running when launching `server.py` (user-level permission to access Docker socket, plus file sharing for `$HOME/.cache`) | Native binaries or pure-Python / MNI152 template registration |
| **FSL (`bet`, `flirt`, `fnirt`, `applywarp`, `invwarp`)** | Fast skull-stripping (`bet`), linear registration (`flirt`), **nonlinear MNI152 → subject atlas warping** (`fnirt`/`applywarp`/`invwarp`), and auto-discovery of `$FSLDIR/data/atlases` | Standard user-space FSL installation (`~/fsl` or `/usr/local/fsl`); Linux or WSL2 on Windows | Pure-Python SciPy 3D morphological BET & 9-DOF affine registration |
| **FastSurfer (`2.x` / Docker)** + **FreeSurfer License** | Deep-learning 104-region DKT (`--seg_only`) and 148-region Destrieux (`--fsaparc`) cortical parcellations | **Destrieux (`--fsaparc`)** requires a valid FreeSurfer `license.txt` (selectable via the UI picker or standard paths) | MNI152 DKT & Destrieux template registration (`FNIRT`, `FLIRT`, or `SciPy`) |
| **`dcm2niix`** | High-speed C++ DICOM-to-NIfTI series conversion | Optional CLI binary on `PATH` | In-browser WebAssembly (`@niivue/nv-ext-dcm2niix`) + pure-Python `pydicom` converter |

---

## Installation Guide by Operating System

### macOS (Apple Silicon `M1/M2/M3/M4` & Intel)

#### Step 1: Install Python 3 & `dcm2niix` (Homebrew)
If you do not have [Homebrew](https://brew.sh) installed, install it first or use the official Python installer from [python.org](https://www.python.org/downloads/macos/):
```bash
# Install Python 3 and dcm2niix
brew install python dcm2niix
```

#### Step 2: (Optional, Recommended) Install FSL (`bet`, `flirt`, `fnirt`, Standard Atlases)
Download and run the official FSL installer script from [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/macos):
```bash
python3 fslinstaller.py --dest "$HOME/fsl"
```
*(Installing into `$HOME/fsl` requires no `sudo`/admin privileges; NeuroCoreg Studio automatically detects both `$HOME/fsl` and `/usr/local/fsl` along with FNIRT binaries and `$FSLDIR/data/atlases`.)*

#### Step 3: (Optional, Recommended for THOMAS) Install HIPS-THOMAS Docker Container
1. Install and start [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) (on Apple Silicon Macs, keep **"Use Rosetta for x86_64/amd64 emulation"** enabled in Docker Desktop Settings if needed for `linux/amd64` container images).
2. Pull the HIPS-THOMAS image:
   ```bash
   docker pull anagrammarian/sthomas:latest
   ```
   *(NeuroCoreg Studio automatically detects `anagrammarian/sthomas`, legacy `anagrammarian/thomasmerged`, or `thalamicseg/hipsthomasdocker`, and lets you specify any custom local tag in Brain Atlas Studio.)*

#### Step 4: (Optional) Install FastSurfer
- **Native macOS bundle**: Extract FastSurfer into `/Applications/FastSurfer*` or `~/FastSurfer*` (with `macos_setup_fastsurfer.sh`). `start.sh` automatically discovers and sources it.
- **Or via Docker Desktop**:
  ```bash
  docker pull deepmi/fastsurfer:latest
  ```

#### Step 5: Launch NeuroCoreg Studio
```bash
git clone https://github.com/danxfriedman/niivue-multimodal-coreg.git
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

#### Step 2: (Optional, Recommended) Install FSL (`bet`, `flirt`, `fnirt`, Standard Atlases)
Download `fslinstaller.py` from [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/linux) and install into your home directory or `/usr/local/fsl`:
```bash
python3 fslinstaller.py --dest "$HOME/fsl"
```

#### Step 3: (Optional, Recommended for THOMAS) Install HIPS-THOMAS Docker Container
Install [Docker Engine](https://docs.docker.com/engine/install/) (or Podman) and add your user to the `docker` group (or configure rootless Docker/Podman), then pull the image:
```bash
docker pull anagrammarian/sthomas:latest
```
*(Or if `hipsthomas.sh` is installed natively on your `PATH`, NeuroCoreg Studio detects and runs it directly without Docker.)*

#### Step 4: (Optional) Install FastSurfer
- **Via Docker**:
  ```bash
  docker pull deepmi/fastsurfer:latest
  ```
- **Or Native Conda/Pip Installation**: Clone [DeepMI/FastSurfer](https://github.com/DeepMI/FastSurfer) into `~/FastSurfer` or `/opt/fastsurfer` and set `export FASTSURFER_HOME="$HOME/FastSurfer"`.

#### Step 5: Launch NeuroCoreg Studio
```bash
git clone https://github.com/danxfriedman/niivue-multimodal-coreg.git
cd niivue-multimodal-coreg
chmod +x start.sh
./start.sh
```
Then open **http://127.0.0.1:8765** in your browser.

---

### Windows (Windows 10 / 11 — Native & WSL2)

#### Option A: Native Windows 10 / 11 (No Administrator Privileges Required for Python Setup)

1. **Install Python 3.10+**:
   - Download the **Windows Installer (64-bit)** from [python.org/downloads/windows](https://www.python.org/downloads/windows/) (check **"Add python.exe to PATH"**; standard per-user installation requires no Administrator elevation), **or** run via `winget`:
     ```cmd
     winget install -e --id Python.Python.3.12 --scope user
     ```
2. **(Optional) Install `dcm2niix`**:
   - Download the latest Windows release (`dcm2niix_win.zip`) from [github.com/rordenlab/dcm2niix/releases](https://github.com/rordenlab/dcm2niix/releases) and place `dcm2niix.exe` in the project folder or on your user `PATH`.
3. **(Optional) Install HIPS-THOMAS & FastSurfer via Docker Desktop**:
   - Install and start [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) (WSL2 backend recommended), then run:
     ```cmd
     docker pull anagrammarian/sthomas:latest
     docker pull deepmi/fastsurfer:latest
     ```
4. **Launch NeuroCoreg Studio**:
   - Double-click **`start.bat`** in File Explorer, or run in Command Prompt / PowerShell:
     ```cmd
     git clone https://github.com/danxfriedman/niivue-multimodal-coreg.git
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

### Environment Variables & Custom Configuration
- **Custom Port** (default `8765`):
  ```bash
  PORT=9000 ./start.sh
  ```
- **Custom Python Binary**:
  ```bash
  NEUROCOREG_PYTHON=/path/to/python3 ./start.sh
  ```
- **Pin Custom HIPS-THOMAS Docker Image Tag** *(also configurable live inside Brain Atlas Studio)*:
  ```bash
  HIPS_THOMAS_DOCKER_IMAGE=myorg/hipsthomas:custom ./start.sh
  ```
- **Custom FSL / FastSurfer Install Paths** *(if installed outside standard directories)*:
  ```bash
  export FSLDIR="$HOME/fsl"
  export FASTSURFER_HOME="$HOME/FastSurfer"
  ./start.sh
  ```

---

## Project Structure

```text
niivue-multimodal-coreg/
├── index.html                        # Main workstation UI, top-bar studio toggles & modals
├── styles.css                        # Dark-mode neuroimaging UI, pop-out & atlas/drawing styling
├── app.js                            # Frontend NiiVue 2D/3D engine, Drawing Studio, Multi-Atlas Studio & async jobs
├── server.py                         # Cross-platform Python HTTP & neuroimaging API server
├── generate_sample_data.py           # Generates synthetic sample multimodal NIfTI & DICOM datasets
├── requirements.txt                  # Python dependencies (numpy, scipy, nibabel, pydicom)
├── start.sh                          # macOS / Linux zero-admin launcher script
├── start.bat                         # Windows zero-admin launcher script
├── scripts/
│   └── fetch_phase3_atlases.py       # Rebuilds bundled 1 mm MNI152 THOMAS & Tian subcortical atlases
└── sample_data/
    ├── dkt_lut.json                  # FreeSurfer / FastSurfer DKT & Destrieux lookup table
    ├── atlases/                      # Bundled MNI152 THOMAS (thomas_lut.json) & Tian S1–S4 NIfTI/LUT assets
    ├── nifti/                        # Sample T1w, CT, SEEG mask, FLAIR, DTI FA, DKT/Destrieux & MNI templates
    └── dicom_t2_flair/               # Sample 91-slice axial T2-FLAIR DICOM series
```
