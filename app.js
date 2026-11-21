import * as NiivueModule from "@niivue/niivue";

// Support both niivue/mono default export (NiiVueGPU) and named export (Niivue)
const NiivueClass = NiivueModule.Niivue || NiivueModule.default || NiivueModule.NiiVueGPU;
const NVImageClass = NiivueModule.NVImage;
const SLICE_TYPE = NiivueModule.SLICE_TYPE || {
  AXIAL: 0,
  CORONAL: 1,
  SAGITTAL: 2,
  MULTIPLANAR: 3,
  RENDER: 4,
};
const DRAG_MODE = NiivueModule.DRAG_MODE || {
  none: 0,
  contrast: 1,
  measurement: 2,
  pan: 3,
  slicer3D: 4,
  callbackOnly: 5,
};

const COLORMAPS = [
  "gray",
  "FreeSurferColorLUT",
  "hot",
  "viridis",
  "plasma",
  "inferno",
  "bone",
  "cool",
  "warm",
  "red",
  "green",
  "blue",
  "actc",
];

function loadSavedHipsCustomImage() {
  try {
    return typeof localStorage !== "undefined" ? (localStorage.getItem("neurocoreg_hips_docker_image") || "") : "";
  } catch (_) {
    return "";
  }
}

function persistHipsCustomImage(val) {
  try {
    if (typeof localStorage === "undefined") return;
    const trimmed = String(val || "").trim();
    if (trimmed) localStorage.setItem("neurocoreg_hips_docker_image", trimmed);
    else localStorage.removeItem("neurocoreg_hips_docker_image");
  } catch (_) {
    // ignore storage errors
  }
}

// Global Application State
const state = {
  nv: null,
  layerMeta: new Map(), // vol.id -> { label, modality, visible, savedOpacity, origAffine, params, isChildOfMov, isCheckedForChildApply, isCheckedForExport, isMask, isDktAtlas, isSyntheticVerification, isEdgeOfMov, isCheckerboard, rawBytes }
  refVolId: null,
  movVolId: null,
  soloVolId: null,
  layersPanelOpen: true,
  coregPanelOpen: false,
  hudPinned: false,
  blinkActive: false,
  blinkTimer: null,
  blinkPhaseRef: true,
  anaglyphActive: false,
  savedPreAnaglyph: new Map(),
  baselineMetrics: null,
  savedTransforms: [], // [{ id, name, worldMat, params }]
  lastCrosshairRAS: [0, 0, 0],
  dktLut: {}, // labelId string -> { id, name, title, rgb, hex }
  neuroToolsStatus: null,
  activeDktVolId: null,
  activeAtlasVolId: null,
  atlasRegistry: [],
  atlasLuts: new Map(), // atlasId -> normalized LUT object
  atlasPreferredModeById: new Map(), // atlasId -> user's explicitly chosen pipeline mode
  lastPopulatedAtlasModeId: null,
  atlasOverlayStyle: "filled", // "filled" | "filled_outline" | "outline"
  atlasStudioPopout: null,
  atlasStudioState: {
    search: "",
    hemi: "all", // "all" | "L" | "R"
    selectedAtlasVolId: null,
    customNiftiFile: null,
    customLutFile: null,
    customSpace: "subject",
    customName: "",
    warpCacheHit: null,
    hipsCacheHit: null,
    hipsContrast: "t1", // "t1" | "wmn"
    hipsForceRecompute: false,
    hipsCustomImage: loadSavedHipsCustomImage(),
  },
  activeAtlasJob: null, // { jobId, atlasId, atlasShort, regEntry, modeLabel, startedAtMs, timerId, elapsedTimerId, stage, logTail, cancelling }
  zoom2d: {
    zoom: 1.0,
    panMM: [0, 0, 0],
    panModeActive: false,
    isDraggingPan: false,
    panCalib: null,
  },
  popouts: new Map(),
  popoutTopZ: 1210,
  focusedPopoutId: null,
  drawing: {
    popout: null,
    paintEnabled: false,
    tool: "pen", // "pen" | "filled_pen" | "eraser" | "flood_fill" | "region_grow"
    brushRadius: 2, // 1..5 voxels
    brush3D: false,
    overwriteLabels: true,
    activeLayerId: null,
    layers: [],
    nextLayerCounter: 1,
    lastSliceAxis: 2,
    lastAxCorSag: 0,
    isStroking: false,
    exportFormat: "voi", // "voi" | "nii.gz" | "nii"
    interpAxis: "auto",
    regionGrow: {
      sourceVolId: null,
      scope: "3d",
      tolerancePct: 15,
      maxRadiusMm: 35,
    },
  },
};

// Utility: Show temporary toast notification
let toastTimeout = null;
function showToast(msg, durationMs = 3200) {
  const el = document.getElementById("statusToast");
  if (!el) return;
  el.textContent = msg;
  el.hidden = false;
  if (toastTimeout) clearTimeout(toastTimeout);
  if (durationMs > 0) {
    toastTimeout = setTimeout(() => {
      el.hidden = true;
    }, durationMs);
  }
}

function setCoregStatus(text, isBusy = false) {
  const pill = document.getElementById("coregStatusPill");
  if (!pill) return;
  pill.textContent = text;
  pill.classList.toggle("busy", isBusy);
}

// ============================================================================
// 4x4 Matrix Math & FSL FLIRT <-> World RAS Conversion Helpers
// ============================================================================
function cloneMat4(m) {
  return m.map((row) => [Number(row[0]), Number(row[1]), Number(row[2]), Number(row[3])]);
}

function identityMat4() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

function multiplyMat4(A, B) {
  const out = identityMat4();
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[r][c] =
        A[r][0] * B[0][c] +
        A[r][1] * B[1][c] +
        A[r][2] * B[2][c] +
        A[r][3] * B[3][c];
    }
  }
  return out;
}

function det3x3(M) {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}

function invertMat4(M) {
  const a = M.map((r) => [...r]);
  const inv = identityMat4();
  for (let i = 0; i < 4; i++) {
    let pivot = i;
    for (let j = i + 1; j < 4; j++) {
      if (Math.abs(a[j][i]) > Math.abs(a[pivot][i])) pivot = j;
    }
    if (Math.abs(a[pivot][i]) < 1e-12) return identityMat4();
    [a[i], a[pivot]] = [a[pivot], a[i]];
    [inv[i], inv[pivot]] = [inv[pivot], inv[i]];
    const div = a[i][i];
    for (let k = 0; k < 4; k++) {
      a[i][k] /= div;
      inv[i][k] /= div;
    }
    for (let j = 0; j < 4; j++) {
      if (j !== i) {
        const factor = a[j][i];
        for (let k = 0; k < 4; k++) {
          a[j][k] -= factor * a[i][k];
          inv[j][k] -= factor * inv[i][k];
        }
      }
    }
  }
  return inv;
}

function applyMat4Pt(M, pt) {
  const [x, y, z] = pt;
  return [
    M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3],
    M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3],
    M[2][0] * x + M[2][1] * y + M[2][2] * z + M[2][3],
  ];
}

// Safely retrieve [ndim, nx, ny, nz, ...] dimensions from an NVImage instance
function getVolDims(vol) {
  if (!vol) return [3, 91, 109, 91];
  if (vol.hdr && Array.isArray(vol.hdr.dims) && vol.hdr.dims.length >= 4) {
    return vol.hdr.dims;
  }
  if (Array.isArray(vol.dims) && vol.dims.length >= 4) {
    return vol.dims;
  }
  if (Array.isArray(vol.dimsRAS) && vol.dimsRAS.length >= 4) {
    return vol.dimsRAS;
  }
  return [3, 91, 109, 91];
}

// Construct FSL FLIRT scaled-voxel coordinate matrix S for a volume
function buildFslSamplingMat4(vol, affineOverride = null) {
  const aff = affineOverride || vol.hdr.affine;
  const dx = Math.hypot(aff[0][0], aff[1][0], aff[2][0]) || 1;
  const dy = Math.hypot(aff[0][1], aff[1][1], aff[2][1]) || 1;
  const dz = Math.hypot(aff[0][2], aff[1][2], aff[2][2]) || 1;
  const dims = getVolDims(vol);
  const nx = dims[1] || 91;

  const S = [
    [dx, 0, 0, 0],
    [0, dy, 0, 0],
    [0, 0, dz, 0],
    [0, 0, 0, 1],
  ];
  if (det3x3(aff) > 0) {
    S[0][0] = -dx;
    S[0][3] = (nx - 1) * dx;
  }
  return S;
}

// Convert FSL FLIRT .mat (scaled-voxel mov -> scaled-voxel ref) to 4x4 World RAS matrix W_ras
function flirtMatToWorldRAS(fslMat, refVol, movVol) {
  const refMeta = state.layerMeta.get(refVol.id);
  const movMeta = state.layerMeta.get(movVol.id);
  const A_ref = refMeta ? refMeta.origAffine : refVol.hdr.affine;
  const A_mov = movMeta ? movMeta.origAffine : movVol.hdr.affine;
  const S_ref = buildFslSamplingMat4(refVol, A_ref);
  const S_mov = buildFslSamplingMat4(movVol, A_mov);

  // W_ras = A_ref * inv(S_ref) * fslMat * S_mov * inv(A_mov)
  return multiplyMat4(
    A_ref,
    multiplyMat4(
      invertMat4(S_ref),
      multiplyMat4(fslMat, multiplyMat4(S_mov, invertMat4(A_mov)))
    )
  );
}

// Convert 4x4 World RAS matrix W_ras to FSL FLIRT .mat
function worldRASToFlirtMat(W_ras, refVol, movVol) {
  const refMeta = state.layerMeta.get(refVol.id);
  const movMeta = state.layerMeta.get(movVol.id);
  const A_ref = refMeta ? refMeta.origAffine : refVol.hdr.affine;
  const A_mov = movMeta ? movMeta.origAffine : movVol.hdr.affine;
  const S_ref = buildFslSamplingMat4(refVol, A_ref);
  const S_mov = buildFslSamplingMat4(movVol, A_mov);

  // M_flirt = S_ref * inv(A_ref) * W_ras * A_mov * inv(S_mov)
  return multiplyMat4(
    S_ref,
    multiplyMat4(
      invertMat4(A_ref),
      multiplyMat4(W_ras, multiplyMat4(A_mov, invertMat4(S_mov)))
    )
  );
}

// Compute volume geometric center in RAS world coordinates from its original affine
function getVolumeCenterRAS(vol, affineOverride = null) {
  const dims = getVolDims(vol);
  const nx = dims[1] || 91;
  const ny = dims[2] || 109;
  const nz = dims[3] || 91;
  const aff = affineOverride || vol.hdr.affine;
  return applyMat4Pt(aff, [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2]);
}

// Convert 9-DOF parameters {tx, ty, tz, rx, ry, rz, sx, sy, sz} around pivotRAS to a 4x4 world matrix
function paramsToWorldMat4(params, pivotRAS = [0, 0, 0]) {
  const { tx = 0, ty = 0, tz = 0, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1 } = params;
  const radX = (rx * Math.PI) / 180;
  const radY = (ry * Math.PI) / 180;
  const radZ = (rz * Math.PI) / 180;

  const cx = Math.cos(radX), sinX = Math.sin(radX);
  const cy = Math.cos(radY), sinY = Math.sin(radY);
  const cz = Math.cos(radZ), sinZ = Math.sin(radZ);

  const Rx = [
    [1, 0, 0, 0],
    [0, cx, -sinX, 0],
    [0, sinX, cx, 0],
    [0, 0, 0, 1],
  ];
  const Ry = [
    [cy, 0, sinY, 0],
    [0, 1, 0, 0],
    [-sinY, 0, cy, 0],
    [0, 0, 0, 1],
  ];
  const Rz = [
    [cz, -sinZ, 0, 0],
    [sinZ, cz, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  const S = [
    [sx, 0, 0, 0],
    [0, sy, 0, 0],
    [0, 0, sz, 0],
    [0, 0, 0, 1],
  ];

  const T_to_origin = [
    [1, 0, 0, -pivotRAS[0]],
    [0, 1, 0, -pivotRAS[1]],
    [0, 0, 1, -pivotRAS[2]],
    [0, 0, 0, 1],
  ];
  const T_back = [
    [1, 0, 0, pivotRAS[0] + tx],
    [0, 1, 0, pivotRAS[1] + ty],
    [0, 0, 1, pivotRAS[2] + tz],
    [0, 0, 0, 1],
  ];

  const R = multiplyMat4(Rz, multiplyMat4(Ry, multiplyMat4(Rx, S)));
  return multiplyMat4(T_back, multiplyMat4(R, T_to_origin));
}

// Decompose a 4x4 World RAS matrix W_ras around pivotRAS into approximate {tx, ty, tz, rx, ry, rz, sx, sy, sz}
function decomposeWorldMat4ToParams(W_ras, pivotRAS = [0, 0, 0]) {
  const cNew = applyMat4Pt(W_ras, pivotRAS);
  const tx = Number((cNew[0] - pivotRAS[0]).toFixed(2));
  const ty = Number((cNew[1] - pivotRAS[1]).toFixed(2));
  const tz = Number((cNew[2] - pivotRAS[2]).toFixed(2));

  const sx = Math.hypot(W_ras[0][0], W_ras[1][0], W_ras[2][0]) || 1;
  const sy = Math.hypot(W_ras[0][1], W_ras[1][1], W_ras[2][1]) || 1;
  const sz = Math.hypot(W_ras[0][2], W_ras[1][2], W_ras[2][2]) || 1;

  const r00 = W_ras[0][0] / sx, r01 = W_ras[0][1] / sy, r02 = W_ras[0][2] / sz;
  const r10 = W_ras[1][0] / sx, r11 = W_ras[1][1] / sy, r12 = W_ras[1][2] / sz;
  const r20 = W_ras[2][0] / sx, r21 = W_ras[2][1] / sy, r22 = W_ras[2][2] / sz;

  const sinY = Math.max(-1, Math.min(1, -r20));
  const ryRad = Math.asin(sinY);
  const cy = Math.cos(ryRad);

  let rxRad = 0;
  let rzRad = 0;
  if (Math.abs(cy) > 1e-5) {
    rxRad = Math.atan2(r21, r22);
    rzRad = Math.atan2(r10, r00);
  } else {
    rxRad = Math.atan2(-r12, r11);
    rzRad = 0;
  }

  return {
    tx: Number.isFinite(tx) ? tx : 0,
    ty: Number.isFinite(ty) ? ty : 0,
    tz: Number.isFinite(tz) ? tz : 0,
    rx: Number(((rxRad * 180) / Math.PI).toFixed(2)),
    ry: Number(((ryRad * 180) / Math.PI).toFixed(2)),
    rz: Number(((rzRad * 180) / Math.PI).toFixed(2)),
    sx: Number(sx.toFixed(3)),
    sy: Number(sy.toFixed(3)),
    sz: Number(sz.toFixed(3)),
  };
}

// Helper: Load an NVImage from raw Uint8Array bytes and automatically revoke the temporary Blob URL
async function loadNVImageFromBytes(bytes, opts = {}) {
  const blobUrl = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
  try {
    return await NVImageClass.loadFromUrl({
      ...opts,
      url: blobUrl,
    });
  } finally {
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  }
}

// Get the current 4x4 World RAS transformation matrix W_ras applied to the Moving volume
function getCurrentMovingWorldMatrix() {
  const movIdx = getVolumeIndexById(state.movVolId);
  if (movIdx < 0) return identityMat4();
  const movVol = state.nv.volumes[movIdx];
  const movMeta = state.layerMeta.get(movVol.id);
  if (!movMeta) return identityMat4();
  // W_ras = A_mov_curr * inv(A_mov_orig)
  return multiplyMat4(movVol.hdr.affine, invertMat4(movMeta.origAffine));
}

// Apply a 4x4 RAS affine matrix to a NiiVue volume and refresh WebGL/WebGPU textures
function setVolumeAffineMatrix(volIdx, newAffine, skipDraw = false) {
  const nv = state.nv;
  if (!nv || volIdx < 0 || volIdx >= nv.volumes.length) return;
  const vol = nv.volumes[volIdx];
  const affCopy = cloneMat4(newAffine);

  if (vol.hdr) {
    vol.hdr.affine = cloneMat4(affCopy);
    vol.hdr.sform_code = 1;
    if (vol.hdr.pixDims && vol.hdr.pixDims.length >= 4) {
      vol.hdr.pixDims[1] = Math.hypot(affCopy[0][0], affCopy[1][0], affCopy[2][0]) || vol.hdr.pixDims[1];
      vol.hdr.pixDims[2] = Math.hypot(affCopy[0][1], affCopy[1][1], affCopy[2][1]) || vol.hdr.pixDims[2];
      vol.hdr.pixDims[3] = Math.hypot(affCopy[0][2], affCopy[1][2], affCopy[2][2]) || vol.hdr.pixDims[3];
    }
    if (vol.hdr.srow_x) vol.hdr.srow_x = [...affCopy[0]];
    if (vol.hdr.srow_y) vol.hdr.srow_y = [...affCopy[1]];
    if (vol.hdr.srow_z) vol.hdr.srow_z = [...affCopy[2]];
  }

  if (typeof nv.setVolumeAffine === "function") {
    try {
      nv.setVolumeAffine(volIdx, cloneMat4(affCopy));
    } catch (_) {
      // Fallback below
    }
  }
  if (typeof vol.calculateRAS === "function") {
    vol.calculateRAS();
  }
  if (!skipDraw) {
    nv.updateGLVolume();
    nv.drawScene();
  }
}

// Propagate a 4x4 World RAS transformation matrix W_ras to all live-linked Child volumes (isChildOfMov)
function propagateWorldMatrixToLinkedChildren(W_ras) {
  const nv = state.nv;
  if (!nv) return 0;
  let count = 0;
  for (let i = 0; i < nv.volumes.length; i++) {
    const v = nv.volumes[i];
    if (v.id === state.refVolId || v.id === state.movVolId) continue;
    const m = state.layerMeta.get(v.id);
    if (!m) continue;
    if (m.isChildOfMov || m.isEdgeOfMov) {
      const childNewAffine = multiplyMat4(W_ras, m.origAffine);
      setVolumeAffineMatrix(i, childNewAffine, true);
      count++;
    }
  }
  return count;
}

// Apply a 4x4 World RAS matrix W_ras to specific target volumes ('mov_and_children' | 'children_only' | 'mov_only')
function applyWorldMatrixToTargets(W_ras, targetMode = "mov_and_children", optionalParams = null) {
  const nv = state.nv;
  if (!nv) return;

  if (targetMode === "mov_and_children" || targetMode === "mov_only") {
    const movIdx = getVolumeIndexById(state.movVolId);
    if (movIdx >= 0) {
      const movVol = nv.volumes[movIdx];
      const movMeta = state.layerMeta.get(movVol.id);
      if (movMeta) {
        const newMovAffine = multiplyMat4(W_ras, movMeta.origAffine);
        setVolumeAffineMatrix(movIdx, newMovAffine, true);
        if (optionalParams) {
          movMeta.params = { ...optionalParams };
        } else {
          // Decompose translation, rotation, and scale parameters from W_ras around volume center
          const c = getVolumeCenterRAS(movVol, movMeta.origAffine);
          movMeta.params = decomposeWorldMat4ToParams(W_ras, c);
        }
        renderAffineMatrixGrid(newMovAffine);
        updateTransformSlidersUI(movMeta.params);
      }
    }
  }

  if (targetMode === "mov_and_children" || targetMode === "children_only") {
    for (let i = 0; i < nv.volumes.length; i++) {
      const v = nv.volumes[i];
      if (v.id === state.refVolId || v.id === state.movVolId) continue;
      const m = state.layerMeta.get(v.id);
      if (!m) continue;
      if (m.isChildOfMov || m.isCheckedForChildApply || m.isEdgeOfMov) {
        const childNewAffine = multiplyMat4(W_ras, m.origAffine);
        setVolumeAffineMatrix(i, childNewAffine, true);
      }
    }
  }

  nv.updateGLVolume();
  nv.drawScene();
  computeAndDisplayAlignmentMetrics(false);
}

// Update the Moving volume's affine (and all linked Child volumes!) from its current slider parameters
function applyMovingTransformFromParams(recomputeMetricsAfter = false) {
  const movIdx = getVolumeIndexById(state.movVolId);
  if (movIdx < 0) return;
  const movVol = state.nv.volumes[movIdx];
  const meta = state.layerMeta.get(movVol.id);
  if (!meta) return;

  const pivotRAS = getVolumeCenterRAS(movVol, meta.origAffine);
  const W_ras = paramsToWorldMat4(meta.params, pivotRAS);
  const newAffine = multiplyMat4(W_ras, meta.origAffine);

  setVolumeAffineMatrix(movIdx, newAffine, true);
  propagateWorldMatrixToLinkedChildren(W_ras);

  state.nv.updateGLVolume();
  state.nv.drawScene();

  renderAffineMatrixGrid(newAffine);
  updateTransformSlidersUI(meta.params);

  if (recomputeMetricsAfter) {
    computeAndDisplayAlignmentMetrics(false);
  }
}

// Undo registration and clear any applied rigid/affine/12-DOF transform on Moving and linked/checked Child volumes
function undoAndClearCoregTransform() {
  if (!state.nv) return;
  const movIdx = getVolumeIndexById(state.movVolId);
  if (movIdx >= 0) {
    const movVol = state.nv.volumes[movIdx];
    const movMeta = state.layerMeta.get(movVol.id);
    if (movMeta) {
      movMeta.params = { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 };
      setVolumeAffineMatrix(movIdx, movMeta.origAffine, true);
      renderAffineMatrixGrid(movMeta.origAffine);
      updateTransformSlidersUI(movMeta.params);
    }
  }

  // Also restore original header affine on all linked or checked Child volumes (and 3D Edge Contour overlay)
  for (let i = 0; i < state.nv.volumes.length; i++) {
    const v = state.nv.volumes[i];
    if (v.id === state.refVolId || v.id === state.movVolId) continue;
    const m = state.layerMeta.get(v.id);
    if (m && (m.isChildOfMov || m.isCheckedForChildApply || m.isEdgeOfMov)) {
      setVolumeAffineMatrix(i, m.origAffine, true);
    }
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  computeAndDisplayAlignmentMetrics(false);
  showToast("Undid registration and cleared transform (restored original header orientation).");
}

function getVolumeIndexById(id) {
  if (!state.nv || !id) return -1;
  return state.nv.volumes.findIndex((v) => v.id === id);
}

// ============================================================================
// Responsive Workspace Scaling & Toggleable Sidebars (Layers L / Coreg C)
// ============================================================================
function resizeWorkspaceCanvas() {
  requestAnimationFrame(() => {
    if (state.nv) {
      if (typeof state.nv.resizeListener === "function") {
        state.nv.resizeListener();
      }
      state.nv.drawScene();
    }
    if (typeof syncStrokeOverlayCanvasSize === "function") {
      syncStrokeOverlayCanvasSize();
    }
  });
}

function toggleLayersPanel(forceState = null) {
  const nextOpen = forceState !== null ? Boolean(forceState) : !state.layersPanelOpen;
  state.layersPanelOpen = nextOpen;

  const grid = document.getElementById("workstationGrid");
  const panel = document.getElementById("layersPanel");
  const toggleBtn = document.getElementById("btnToggleLayersPanel");

  if (grid) grid.classList.toggle("layers-collapsed", !nextOpen);
  if (panel) panel.hidden = !nextOpen;
  if (toggleBtn) {
    toggleBtn.classList.toggle("active", nextOpen);
    toggleBtn.setAttribute("aria-expanded", String(nextOpen));
  }

  resizeWorkspaceCanvas();
}

function toggleCoregPanel(forceState = null) {
  const nextOpen = forceState !== null ? Boolean(forceState) : !state.coregPanelOpen;
  state.coregPanelOpen = nextOpen;

  const grid = document.getElementById("workstationGrid");
  const panel = document.getElementById("coregPanel");
  const toggleBtn = document.getElementById("btnToggleCoregPanel");
  const verifBar = document.getElementById("verificationBar");
  const metricsBar = document.getElementById("metricsBar");

  if (grid) grid.classList.toggle("coreg-collapsed", !nextOpen);
  if (panel) panel.hidden = !nextOpen;
  if (toggleBtn) {
    toggleBtn.classList.toggle("active", nextOpen);
    toggleBtn.setAttribute("aria-expanded", String(nextOpen));
  }

  // Auto-show/hide Verification & Metrics HUD unless pinned
  const showHud = nextOpen || state.hudPinned;
  if (verifBar) verifBar.hidden = !showHud;
  if (metricsBar) metricsBar.hidden = !showHud;

  resizeWorkspaceCanvas();
}

// Modality heuristic from filename
function inferModalityAndStyle(filename) {
  const lower = filename.toLowerCase();
  if (lower.includes("destrieux") || lower.includes("a2009s")) {
    return {
      modality: "Destrieux Atlas (148 ROIs)",
      colormap: "actc",
      opacity: 0.45,
      isMask: true,
      isAtlas: true,
      isDktAtlas: true,
      isFreeSurferAtlas: true,
      atlasType: "destrieux",
      atlasId: "destrieux",
      atlasShortName: "Destrieux",
      atlasBadge: "DESTRIEUX",
      atlasCategory: "cortical",
    };
  }
  if (lower.includes("thomas") || lower.includes("sthomas")) {
    return {
      modality: "THOMAS Thalamic Nuclei Atlas",
      colormap: "actc",
      opacity: 0.55,
      isMask: true,
      isAtlas: true,
      isDktAtlas: true,
      isFreeSurferAtlas: false,
      atlasType: "thomas",
      atlasId: "thomas",
      atlasShortName: "THOMAS",
      atlasBadge: "THOMAS",
      atlasCategory: "thalamic",
    };
  }
  const tianMatch = lower.match(/tian[_-]?subcortex[_-]?s([1-4])|tian[_-]?s([1-4])/);
  if (tianMatch) {
    const sc = tianMatch[1] || tianMatch[2];
    return {
      modality: `Tian Melbourne Subcortex S${sc}`,
      colormap: "actc",
      opacity: 0.5,
      isMask: true,
      isAtlas: true,
      isDktAtlas: true,
      isFreeSurferAtlas: false,
      atlasType: `tian_s${sc}`,
      atlasId: `tian_s${sc}`,
      atlasShortName: `Tian S${sc}`,
      atlasBadge: `TIAN S${sc}`,
      atlasCategory: "subcortical",
    };
  }
  if (lower.includes("harvardoxford") || lower.includes("jhu") || lower.includes("cerebellum") || lower.includes("_stn_") || lower.startsWith("stn")) {
    return {
      modality: "FSL MNI152 Atlas",
      colormap: "actc",
      opacity: 0.48,
      isMask: true,
      isAtlas: true,
      isDktAtlas: true,
      isFreeSurferAtlas: false,
      atlasType: "fsl",
      atlasId: "fsl",
      atlasShortName: "FSL Atlas",
      atlasBadge: "FSL",
      atlasCategory: "fsl",
    };
  }
  if (lower.includes("dkt") || lower.includes("aparc") || lower.includes("aseg") || lower.includes("atlas")) {
    return {
      modality: "DKT Atlas (104 ROIs)",
      colormap: "actc",
      opacity: 0.38,
      isMask: true,
      isAtlas: true,
      isDktAtlas: true,
      isFreeSurferAtlas: true,
      atlasType: "dkt",
      atlasId: "dkt",
      atlasShortName: "DKT",
      atlasBadge: "DKT",
      atlasCategory: "cortical",
    };
  }
  if (lower.includes("mask") || lower.includes("seeg") || lower.includes("elec") || lower.includes("roi") || lower.includes("seg")) {
    return { modality: "ROI / Contacts Mask", colormap: "green", opacity: 0.9, isMask: true, isDktAtlas: false };
  }
  if (lower.includes("ct")) {
    return { modality: "Post-Op CT", colormap: "hot", opacity: 0.65, isMask: false, isDktAtlas: false };
  }
  if (lower.includes("flair") || lower.includes("t2")) {
    return { modality: "T2-FLAIR MRI", colormap: "viridis", opacity: 0.7, isMask: false, isDktAtlas: false };
  }
  if (lower.includes("fa") || lower.includes("dti")) {
    return { modality: "DTI FA", colormap: "plasma", opacity: 0.7, isMask: false, isDktAtlas: false };
  }
  if (lower.includes("qsm")) {
    return { modality: "QSM", colormap: "cool", opacity: 0.7, isMask: false, isDktAtlas: false };
  }
  if (lower.includes("t1")) {
    return { modality: lower.includes("brain") ? "T1w Brain (BET)" : "T1w MRI", colormap: "gray", opacity: 1.0, isMask: false, isDktAtlas: false };
  }
  return { modality: "Neuro Volume", colormap: "gray", opacity: 0.85, isMask: false, isDktAtlas: false };
}

// Register metadata for a newly added NVImage volume
function registerVolumeMeta(vol, overrides = {}) {
  const inferred = inferModalityAndStyle(vol.name || "volume.nii.gz");
  const isFirst = state.nv.volumes.length === 1;
  const initialOpacity = overrides.opacity !== undefined ? overrides.opacity : isFirst ? 1.0 : inferred.opacity;
  const visible = overrides.visible !== undefined ? overrides.visible : initialOpacity > 0;
  const savedOpacity = initialOpacity > 0 ? initialOpacity : 0.75;

  const origAffine = vol.hdr && vol.hdr.affine ? cloneMat4(vol.hdr.affine) : identityMat4();
  const isChild = Boolean(overrides.isChildOfMov || overrides.role === "child");
  const isDrawingLayer = Boolean(overrides.isDrawingLayer);
  const isMask = overrides.isMask !== undefined ? Boolean(overrides.isMask) : isDrawingLayer || inferred.isMask;
  const isDktAtlas = !isDrawingLayer && (overrides.isDktAtlas !== undefined ? Boolean(overrides.isDktAtlas) : Boolean(overrides.isAtlas || inferred.isDktAtlas));
  const atlasType = overrides.atlasType || overrides.atlasId || inferred.atlasType || (isDktAtlas ? resolveAtlasTypeForVolume(vol) : null);
  const atlasId = overrides.atlasId || atlasType || null;
  const isFreeSurferAtlas =
    overrides.isFreeSurferAtlas !== undefined
      ? Boolean(overrides.isFreeSurferAtlas)
      : atlasId === "dkt" || atlasId === "destrieux";

  state.layerMeta.set(vol.id, {
    label: overrides.label || vol.name || "Volume",
    modality: overrides.modality || (isDrawingLayer ? "Drawing VOI" : inferred.modality),
    visible,
    savedOpacity,
    userAtlasOpacity: overrides.userAtlasOpacity ?? (initialOpacity > 0 ? initialOpacity : 0.48),
    origAffine,
    params: { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 },
    isChildOfMov: isChild,
    isCheckedForChildApply: !isDrawingLayer && (isChild || isMask),
    isCheckedForExport: !isDrawingLayer && (isChild || overrides.role === "moving" || isDktAtlas),
    isMask,
    isAtlas: isDktAtlas,
    isDktAtlas,
    isFreeSurferAtlas,
    isDrawingLayer,
    drawingLayerId: overrides.drawingLayerId || null,
    atlasType,
    atlasId,
    atlasName: overrides.atlasName || overrides.label || inferred.modality,
    atlasShortName: overrides.atlasShortName || inferred.atlasShortName || (atlasId ? atlasId.toUpperCase() : null),
    atlasCategory: overrides.atlasCategory || inferred.atlasCategory || "cortical",
    atlasBadge: overrides.atlasBadge || inferred.atlasBadge || (atlasId ? atlasId.toUpperCase() : "ATLAS"),
    atlasEngine: overrides.atlasEngine || null,
    atlasSpace: overrides.atlasSpace || null,
    lut: overrides.lut || null,
    regions: Array.isArray(overrides.regions) ? overrides.regions : null,
    visibleRegionIds: overrides.visibleRegionIds instanceof Set ? new Set(overrides.visibleRegionIds) : null,
    rawAtlasVoxels: overrides.rawAtlasVoxels || null,
    rawBytes: overrides.rawBytes || null,
    isSyntheticVerification: Boolean(overrides.isSyntheticVerification),
    isEdgeOfMov: Boolean(overrides.isEdgeOfMov),
    isCheckerboard: Boolean(overrides.isCheckerboard),
  });

  if (isDktAtlas) {
    state.activeDktVolId = vol.id;
    state.activeAtlasVolId = vol.id;
  }

  if (!state.refVolId && !overrides.isSyntheticVerification && !isDktAtlas && !isDrawingLayer) {
    state.refVolId = vol.id;
  } else if (
    !state.movVolId &&
    vol.id !== state.refVolId &&
    !isChild &&
    !overrides.isSyntheticVerification &&
    !isDktAtlas &&
    !isDrawingLayer
  ) {
    state.movVolId = vol.id;
  }
}

// Toggle visibility ON or OFF for a specific volume
function toggleVolumeVisibility(volId, forceState = null) {
  const idx = getVolumeIndexById(volId);
  if (idx < 0) return;
  const meta = state.layerMeta.get(volId);
  if (!meta) return;

  const nextVis = forceState !== null ? forceState : !meta.visible;

  // Atlas overlays share single-active-slice-overlay semantics while keeping all loaded atlases active for stacked hover
  if (meta.isAtlas || meta.isDktAtlas) {
    activateAtlasOverlay(volId, nextVis);
    return;
  }

  meta.visible = nextVis;
  const targetOpacity = nextVis ? (meta.savedOpacity > 0 ? meta.savedOpacity : 0.75) : 0.0;
  state.nv.setOpacity(idx, targetOpacity);

  if (meta.isDrawingLayer && meta.drawingLayerId) {
    const dLayer = state.drawing.layers.find((l) => l.id === meta.drawingLayerId);
    if (dLayer) {
      dLayer.visible = nextVis;
      renderDrawingStudioUI();
    }
  }

  state.nv.drawScene();
  renderLayerList();
}

// Render the Multimodal Layer Stack in the left sidebar
function renderLayerList() {
  const listEl = document.getElementById("layerList");
  if (!listEl || !state.nv) return;
  listEl.innerHTML = "";

  if (state.nv.volumes.length === 0) {
    listEl.innerHTML = `
      <div class="voxel-empty" style="padding:14px 10px;text-align:center;border:1px dashed var(--border-subtle);border-radius:8px;">
        Workspace is empty. Load NIfTI files, a DICOM folder, Sample Subject, or Open a Saved Project.
      </div>
    `;
    syncCoregVolumeSelectors();
    renderChildAndExportLists();
    syncDktOverlayControlsUI();
    return;
  }

  state.nv.volumes.forEach((vol) => {
    const meta = state.layerMeta.get(vol.id);
    if (!meta) return;

    const isDraw = Boolean(meta.isDrawingLayer);
    const drawLayer = isDraw ? state.drawing.layers.find((l) => l.id === meta.drawingLayerId) : null;
    const isActiveDraw = Boolean(drawLayer && state.drawing.activeLayerId === drawLayer.id);

    const isRef = vol.id === state.refVolId;
    const isMov = vol.id === state.movVolId;
    const isChild = Boolean(meta.isChildOfMov) && !isRef && !isMov;
    const isDkt = Boolean(meta.isAtlas || meta.isDktAtlas);
    const atlasBadgeInfo = isDkt ? getAtlasBadgeInfo(meta) : null;
    const isBet = Boolean(meta.isBetBrain);
    const isSolo = vol.id === state.soloVolId;

    const card = document.createElement("div");
    card.className = `layer-card ${!meta.visible ? "is-hidden" : ""} ${isDraw ? "is-drawing-layer" : ""} ${
      isActiveDraw ? "is-active-drawing" : ""
    } ${isRef ? "role-ref" : ""} ${isMov ? "role-mov" : ""} ${isChild ? "role-child" : ""}`;
    card.setAttribute("role", "listitem");

    const currentOpacityPct = Math.round((meta.visible ? vol.opacity : 0) * 100);
    const calMin = vol.cal_min !== undefined ? Number(vol.cal_min).toFixed(1) : "0";
    const calMax = vol.cal_max !== undefined ? Number(vol.cal_max).toFixed(1) : "100";
    const dims = getVolDims(vol);

    if (isDraw) {
      const swatchColor = drawLayer?.colorHex || "#f43f5e";
      const volCc = (drawLayer?.volumeCc ?? 0).toFixed(2);
      const voxCnt = drawLayer?.voxelCount ?? 0;
      const modeTag = drawLayer?.mode === "multilabel" ? "1–6" : "0/1";
      card.innerHTML = `
        <div class="layer-top-row">
          <label class="vis-switch-label" title="Toggle drawing layer visibility ON / OFF">
            <input type="checkbox" role="switch" class="vis-switch" data-action="toggle-vis" data-id="${vol.id}" ${meta.visible ? "checked" : ""} />
            <span class="draw-layer-card-swatch" style="background:${swatchColor};"></span>
            <span class="layer-title" title="${meta.label} (${vol.name})">${meta.label}</span>
          </label>
          <div class="layer-badges">
            <span class="role-badge draw" title="Editable Drawing / VOI Layer">DRAW</span>
            <button class="role-badge draw-edit" data-action="edit-drawing" data-id="${vol.id}" title="Select and edit this layer in Drawing Studio">Edit</button>
            <button class="icon-btn ${isSolo ? "solo-active" : ""}" data-action="solo" data-id="${vol.id}" title="Solo this layer">S</button>
            <button class="icon-btn" data-action="export-drawing" data-id="${vol.id}" title="Save Drawing Layer as .voi / NIfTI">⬇</button>
            <button class="icon-btn" data-action="remove" data-id="${vol.id}" title="Delete drawing layer">✕</button>
          </div>
        </div>

        <div class="layer-controls-grid">
          <div class="control-pair">
            <span>α</span>
            <input type="range" min="0" max="100" value="${currentOpacityPct}" data-action="opacity" data-id="${vol.id}" title="Adjust Drawing Layer Opacity" />
            <span class="opacity-pct">${currentOpacityPct}%</span>
          </div>
          <span class="draw-vol-pill" title="Painted volume (${modeTag} labels)">${volCc} cm³ • ${voxCnt} vox (${modeTag})</span>
        </div>

        <div class="layer-window-row">
          <span style="color:var(--text-muted);font-size:10px;">
            ${isActiveDraw ? "● Active in Drawing Studio" : "Click Edit to paint on this layer"}
          </span>
          <span style="margin-left:auto;color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">
            ${dims[1]}×${dims[2]}×${dims[3]}
          </span>
        </div>
      `;
    } else {
      const canConvert = !meta.isSyntheticVerification && !meta.isDktAtlas && !meta.isAtlas;
      card.innerHTML = `
        <div class="layer-top-row">
          <label class="vis-switch-label" title="Toggle volume visibility ON / OFF">
            <input type="checkbox" role="switch" class="vis-switch" data-action="toggle-vis" data-id="${vol.id}" ${meta.visible ? "checked" : ""} />
            <span class="layer-title" title="${meta.label} (${vol.name})">${meta.label}</span>
          </label>
          <div class="layer-badges">
            ${isBet ? `<span class="role-badge bet" title="Skull-Stripped Brain Extracted Volume (FSL BET)">BET</span>` : ""}
            ${isDkt && atlasBadgeInfo ? `<span class="role-badge dkt ${atlasBadgeInfo.cls}" title="Loaded Brain Atlas (${meta.atlasName || meta.label}) — Stacked Hover Active">${atlasBadgeInfo.text}</span>` : ""}
            <button class="role-badge ${isRef ? "ref" : "inactive"}" data-action="set-ref" data-id="${vol.id}" title="Set as Fixed Reference Volume">REF</button>
            <button class="role-badge ${isMov ? "mov" : "inactive"}" data-action="set-mov" data-id="${vol.id}" title="Set as Moving Parent Volume">MOV</button>
            <button class="role-badge ${isChild ? "child" : "inactive"}" data-action="toggle-child" data-id="${vol.id}" title="Link/Unlink as Child of Moving Volume (inherits Moving transforms automatically)">CHILD</button>
            <button class="icon-btn ${isSolo ? "solo-active" : ""}" data-action="solo" data-id="${vol.id}" title="Solo this volume">S</button>
            ${canConvert ? `<button class="icon-btn" data-action="convert-to-drawing" data-id="${vol.id}" title="Convert / duplicate this mask or volume into an editable Drawing Layer">✎</button>` : ""}
            <button class="icon-btn" data-action="quick-export" data-id="${vol.id}" title="Export this volume as coregistered NIfTI">⬇</button>
            <button class="icon-btn" data-action="remove" data-id="${vol.id}" title="Remove volume">✕</button>
          </div>
        </div>

        <div class="layer-controls-grid">
          <div class="control-pair">
            <span>α</span>
            <input type="range" min="0" max="100" value="${currentOpacityPct}" data-action="opacity" data-id="${vol.id}" title="Adjust Volume Opacity" />
            <span class="opacity-pct">${currentOpacityPct}%</span>
          </div>
          <select class="select-compact" data-action="colormap" data-id="${vol.id}" title="Volume Colormap">
            ${COLORMAPS.map((cm) => `<option value="${cm}" ${vol.colormap === cm ? "selected" : ""}>${cm}</option>`).join("")}
          </select>
        </div>

        <div class="layer-window-row">
          <span>Window:</span>
          <input type="number" class="window-input" step="any" value="${calMin}" data-action="cal-min" data-id="${vol.id}" title="Intensity Min (cal_min)" />
          <span>–</span>
          <input type="number" class="window-input" step="any" value="${calMax}" data-action="cal-max" data-id="${vol.id}" title="Intensity Max (cal_max)" />
          <button class="btn-xs" data-action="auto-window" data-id="${vol.id}" title="Reset intensity window to robust min/max">Auto</button>
          <span style="margin-left:auto;color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">
            ${dims[1]}×${dims[2]}×${dims[3]}
          </span>
        </div>
      `;
    }

    listEl.appendChild(card);
  });

  syncCoregVolumeSelectors();
  renderChildAndExportLists();
  syncDktOverlayControlsUI();
  if (typeof renderDrawingStudioUI === "function") {
    renderDrawingStudioUI();
  }
}

// Populate the Reference & Moving dropdowns in the right-hand Coregistration Studio
function syncCoregVolumeSelectors() {
  const refSelect = document.getElementById("selectRefVol");
  const movSelect = document.getElementById("selectMovVol");
  if (!refSelect || !movSelect || !state.nv) return;

  const realVolumes = state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isSyntheticVerification && !m.isDrawingLayer;
  });

  if (realVolumes.length > 0 && !realVolumes.some((v) => v.id === state.refVolId)) {
    state.refVolId = realVolumes[0].id;
  }
  if (realVolumes.length > 1 && (!state.movVolId || state.movVolId === state.refVolId || !realVolumes.some((v) => v.id === state.movVolId))) {
    const second = realVolumes.find((v) => v.id !== state.refVolId);
    if (second) state.movVolId = second.id;
  }

  refSelect.innerHTML = realVolumes
    .map((v) => {
      const m = state.layerMeta.get(v.id);
      return `<option value="${v.id}" ${v.id === state.refVolId ? "selected" : ""}>${m ? m.label : v.name}</option>`;
    })
    .join("");

  movSelect.innerHTML = realVolumes
    .map((v) => {
      const m = state.layerMeta.get(v.id);
      return `<option value="${v.id}" ${v.id === state.movVolId ? "selected" : ""}>${m ? m.label : v.name}</option>`;
    })
    .join("");

  const movIdx = getVolumeIndexById(state.movVolId);
  if (movIdx >= 0) {
    const movVol = state.nv.volumes[movIdx];
    const meta = state.layerMeta.get(movVol.id);
    if (meta) {
      updateTransformSlidersUI(meta.params);
      renderAffineMatrixGrid(movVol.hdr.affine);
    }
  } else {
    updateTransformSlidersUI({ tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 });
    renderAffineMatrixGrid(identityMat4());
  }

  syncNeuroT1Selector();
}

function syncNeuroT1Selector() {
  if (!state.nv) return;
  const selectIds = ["selectBetT1Vol", "selectDktT1Vol", "selectNeuroT1Vol"];
  const candidateVols = state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isSyntheticVerification && !m.isAtlas && !m.isDktAtlas && !m.isMask && !m.isDrawingLayer;
  });

  const t1Match = candidateVols.find(
    (v) =>
      (v.name && v.name.toLowerCase().includes("t1") && !v.name.toLowerCase().includes("brain")) ||
      state.layerMeta.get(v.id)?.modality?.toLowerCase().includes("t1")
  );
  const defaultId = t1Match ? t1Match.id : state.refVolId || candidateVols[0]?.id || "";

  selectIds.forEach((id) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    if (candidateVols.length === 0) {
      sel.innerHTML = `<option value="">No structural volume loaded</option>`;
      return;
    }
    const prevVal = sel.value;
    const selectedId = prevVal && candidateVols.some((v) => v.id === prevVal) ? prevVal : defaultId;
    sel.innerHTML = candidateVols
      .map((v) => {
        const m = state.layerMeta.get(v.id);
        return `<option value="${v.id}" ${v.id === selectedId ? "selected" : ""}>${m ? m.label : v.name}</option>`;
      })
      .join("");
  });
  if (typeof syncAtlasStudioUI === "function" && state.atlasStudioPopout?.isOpen?.()) {
    syncAtlasStudioUI();
  }
}

// Render Section 4 (Child Images list) and Section 5 (NIfTI Export Checklist)
function renderChildAndExportLists() {
  const childListEl = document.getElementById("childVolumesList");
  const exportListEl = document.getElementById("exportVolumesChecklist");
  if (!state.nv) return;

  const candidateChildren = state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isSyntheticVerification && !m.isDrawingLayer && v.id !== state.refVolId && v.id !== state.movVolId;
  });

  if (childListEl) {
    if (candidateChildren.length === 0) {
      childListEl.innerHTML = `<div class="voxel-empty">Load additional volumes (masks, electrodes, fMRI/DTI) to link or transform as Child images.</div>`;
    } else {
      childListEl.innerHTML = candidateChildren
        .map((v) => {
          const m = state.layerMeta.get(v.id);
          return `
            <div class="child-row ${m.isChildOfMov ? "is-linked" : ""}">
              <label class="child-check-label" title="Select ${m.label} for on-demand transform application">
                <input type="checkbox" data-child-check="${v.id}" ${m.isCheckedForChildApply || m.isChildOfMov ? "checked" : ""} />
                <span>${m.label}</span>
              </label>
              <button class="role-badge ${m.isChildOfMov ? "child" : "inactive"}" data-child-link="${v.id}" title="Toggle live parent-child linking with Moving volume">
                ${m.isChildOfMov ? "🔗 LIVE CHILD" : "LINK"}
              </button>
            </div>
          `;
        })
        .join("");
    }
  }

  if (exportListEl) {
    const exportCandidates = state.nv.volumes.filter((v) => {
      const m = state.layerMeta.get(v.id);
      return m && !m.isSyntheticVerification && !m.isDrawingLayer && v.id !== state.refVolId;
    });

    if (exportCandidates.length === 0) {
      exportListEl.innerHTML = `<div class="voxel-empty">No Moving or Child volumes loaded to export.</div>`;
    } else {
      exportListEl.innerHTML = exportCandidates
        .map((v) => {
          const m = state.layerMeta.get(v.id);
          const isMov = v.id === state.movVolId;
          const badge = isMov ? `<span class="role-badge mov">MOV</span>` : m.isChildOfMov ? `<span class="role-badge child">CHILD</span>` : "";
          const checked = isMov || m.isChildOfMov || m.isCheckedForExport;
          return `
            <div class="export-row">
              <label class="child-check-label">
                <input type="checkbox" data-export-check="${v.id}" ${checked ? "checked" : ""} />
                <span>${m.label}</span>
              </label>
              <div style="display:flex;align-items:center;gap:4px;">
                ${badge}
                <button class="btn-xs" data-export-single="${v.id}" title="Export only ${v.name} as NIfTI">Export</button>
              </div>
            </div>
          `;
        })
        .join("");
    }
  }
}

function renderSavedTransformsDropdown() {
  const sel = document.getElementById("selectSavedTransform");
  if (!sel) return;
  const options = [
    `<option value="__current_mov__">Current Moving Volume Transform</option>`,
    ...state.savedTransforms.map(
      (t, i) => `<option value="${t.id}" ${i === state.savedTransforms.length - 1 ? "selected" : ""}>${t.name}</option>`
    ),
  ];
  sel.innerHTML = options.join("");
}

function updateTransformSlidersUI(params) {
  const keys = ["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"];
  keys.forEach((k) => {
    const slider = document.getElementById(`slider_${k}`);
    const valEl = document.getElementById(`val_${k}`);
    const v = params[k] !== undefined ? params[k] : k.startsWith("s") ? 1 : 0;
    if (slider) slider.value = String(v);
    if (valEl) valEl.textContent = k.startsWith("s") ? Number(v).toFixed(2) : Number(v).toFixed(1);
  });
}

function renderAffineMatrixGrid(mat) {
  const grid = document.getElementById("affineMatrixGrid");
  if (!grid || !mat) return;
  grid.innerHTML = "";
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const cell = document.createElement("div");
      cell.className = `matrix-cell ${c === 3 && r < 3 ? "trans-col" : ""}`;
      cell.textContent = Number(mat[r][c]).toFixed(2);
      grid.appendChild(cell);
    }
  }
}

// ============================================================================
// Voxel Sampling, In-Browser Resampling & Quantitative Alignment Metrics
// ============================================================================
function sampleVolumeAtRAS(vol, invAffine, rasPt, useNearest = false) {
  const vox = applyMat4Pt(invAffine, rasPt);
  const dims = getVolDims(vol);
  const nx = dims[1] || 1, ny = dims[2] || 1, nz = dims[3] || 1;
  const ix = vox[0], iy = vox[1], iz = vox[2];
  const maxX = Math.max(0, nx - 1);
  const maxY = Math.max(0, ny - 1);
  const maxZ = Math.max(0, nz - 1);
  if (
    !Number.isFinite(ix) || !Number.isFinite(iy) || !Number.isFinite(iz) ||
    ix < 0 || ix > maxX || iy < 0 || iy > maxY || iz < 0 || iz > maxZ
  ) {
    return 0;
  }
  const sliceStride = nx * ny;
  const img = vol.img;
  if (!img) return 0;

  let raw = 0;
  if (useNearest || nx < 2 || ny < 2 || nz < 2) {
    const rx = Math.min(maxX, Math.max(0, Math.round(ix)));
    const ry = Math.min(maxY, Math.max(0, Math.round(iy)));
    const rz = Math.min(maxZ, Math.max(0, Math.round(iz)));
    raw = img[rz * sliceStride + ry * nx + rx];
  } else {
    const x0 = Math.min(nx - 2, Math.max(0, Math.floor(ix)));
    const y0 = Math.min(ny - 2, Math.max(0, Math.floor(iy)));
    const z0 = Math.min(nz - 2, Math.max(0, Math.floor(iz)));
    const fx = ix - x0, fy = iy - y0, fz = iz - z0;
    const idx000 = z0 * sliceStride + y0 * nx + x0;
    const c00 = img[idx000] * (1 - fx) + img[idx000 + 1] * fx;
    const c10 = img[idx000 + nx] * (1 - fx) + img[idx000 + nx + 1] * fx;
    const c01 = img[idx000 + sliceStride] * (1 - fx) + img[idx000 + sliceStride + 1] * fx;
    const c11 = img[idx000 + sliceStride + nx] * (1 - fx) + img[idx000 + sliceStride + nx + 1] * fx;
    const c0 = c00 * (1 - fy) + c10 * fy;
    const c1 = c01 * (1 - fy) + c11 * fy;
    raw = c0 * (1 - fz) + c1 * fz;
  }

  const slope = vol.hdr && vol.hdr.scl_slope && vol.hdr.scl_slope !== 0 ? vol.hdr.scl_slope : 1;
  const inter = vol.hdr && vol.hdr.scl_inter ? vol.hdr.scl_inter : 0;
  const val = raw * slope + inter;
  return Number.isFinite(val) ? val : 0;
}

function computeMetricsInBrowser(refVol, movVol, customMovAffine = null, stride = 3) {
  if (!refVol || !movVol || !refVol.img || !movVol.img) {
    return { nmi: 1.0, ncc: 0.0, edgeOverlap: 0.0 };
  }
  const refAff = refVol.hdr.affine;
  const movAff = customMovAffine || movVol.hdr.affine;
  const invMovAff = invertMat4(movAff);
  const voxToVox = multiplyMat4(invMovAff, refAff);

  const refDims = getVolDims(refVol);
  const movDims = getVolDims(movVol);
  const nx = refDims[1] || 1, ny = refDims[2] || 1, nz = refDims[3] || 1;
  const mnx = movDims[1] || 1, mny = movDims[2] || 1, mnz = movDims[3] || 1;
  const rSamples = [];
  const mSamples = [];
  const rGradSamples = [];
  const mGradSamples = [];

  let rMin = Infinity, rMax = -Infinity;
  let mMin = Infinity, mMax = -Infinity;

  const refImg = refVol.img;
  const sliceStride = nx * ny;
  const kStart = nz > 2 * stride ? stride : 0;
  const kEnd = nz > 2 * stride ? nz - stride : nz;
  const kStep = nz > 2 * stride ? stride : 1;

  for (let k = kStart; k < kEnd; k += kStep) {
    for (let j = stride; j < ny - stride; j += stride) {
      for (let i = stride; i < nx - stride; i += stride) {
        const idx = k * sliceStride + j * nx + i;
        const rVal = refImg[idx];
        if (!Number.isFinite(rVal)) continue;
        const mVox = applyMat4Pt(voxToVox, [i, j, k]);
        const mx = mVox[0], my = mVox[1], mz = mVox[2];
        if (mx < 1 || mx >= mnx - 2 || my < 1 || my >= mny - 2 || mz < 0 || mz >= Math.max(1, mnz - 1)) {
          continue;
        }
        const mVal = sampleVolumeAtRAS(movVol, invMovAff, applyMat4Pt(refAff, [i, j, k]), false);
        if (!Number.isFinite(mVal) || (rVal <= 0 && mVal <= 0)) continue;

        rSamples.push(rVal);
        mSamples.push(mVal);
        if (rVal < rMin) rMin = rVal;
        if (rVal > rMax) rMax = rVal;
        if (mVal < mMin) mMin = mVal;
        if (mVal > mMax) mMax = mVal;

        const rgx = refImg[idx + 1] - refImg[idx - 1];
        const rgy = refImg[idx + nx] - refImg[idx - nx];
        const rgz = k > 0 && k < nz - 1 ? refImg[idx + sliceStride] - refImg[idx - sliceStride] : 0;
        const rGrad = Math.hypot(rgx, rgy, rgz);

        const mIx = Math.round(mx), mIy = Math.round(my), mIz = Math.min(mnz - 1, Math.max(0, Math.round(mz)));
        const mSlice = mnx * mny;
        const mIdx = mIz * mSlice + mIy * mnx + mIx;
        const mgx = movVol.img[mIdx + 1] - movVol.img[mIdx - 1];
        const mgy = movVol.img[mIdx + mnx] - movVol.img[mIdx - mnx];
        const mgz = mIz > 0 && mIz < mnz - 1 ? movVol.img[mIdx + mSlice] - movVol.img[mIdx - mSlice] : 0;
        const mGrad = Math.hypot(mgx, mgy, mgz);

        rGradSamples.push(Number.isFinite(rGrad) ? rGrad : 0);
        mGradSamples.push(Number.isFinite(mGrad) ? mGrad : 0);
      }
    }
  }

  const N = rSamples.length;
  if (N < 50) return { nmi: 1.0, ncc: 0.0, edgeOverlap: 0.0 };

  const BINS = 24;
  const hist2d = new Float64Array(BINS * BINS);
  const px = new Float64Array(BINS);
  const py = new Float64Array(BINS);
  const rSpan = rMax - rMin || 1;
  const mSpan = mMax - mMin || 1;

  let sumR = 0, sumM = 0;
  for (let n = 0; n < N; n++) {
    const r = rSamples[n];
    const m = mSamples[n];
    sumR += r;
    sumM += m;
    let bx = Math.floor(((r - rMin) / rSpan) * (BINS - 1));
    let by = Math.floor(((m - mMin) / mSpan) * (BINS - 1));
    if (bx < 0) bx = 0; else if (bx >= BINS) bx = BINS - 1;
    if (by < 0) by = 0; else if (by >= BINS) by = BINS - 1;
    hist2d[bx * BINS + by] += 1;
    px[bx] += 1;
    py[by] += 1;
  }

  let hx = 0, hy = 0, hxy = 0;
  for (let b = 0; b < BINS; b++) {
    if (px[b] > 0) {
      const p = px[b] / N;
      hx -= p * Math.log(p);
    }
    if (py[b] > 0) {
      const p = py[b] / N;
      hy -= p * Math.log(p);
    }
  }
  for (let b = 0; b < BINS * BINS; b++) {
    if (hist2d[b] > 0) {
      const p = hist2d[b] / N;
      hxy -= p * Math.log(p);
    }
  }
  const nmi = hxy > 1e-9 ? (hx + hy) / hxy : 1.0;

  const meanR = sumR / N;
  const meanM = sumM / N;
  let num = 0, denR = 0, denM = 0;
  let gDot = 0, gDenR = 0, gDenM = 0;
  for (let n = 0; n < N; n++) {
    const dr = rSamples[n] - meanR;
    const dm = mSamples[n] - meanM;
    num += dr * dm;
    denR += dr * dr;
    denM += dm * dm;

    const gr = rGradSamples[n];
    const gm = mGradSamples[n];
    gDot += gr * gm;
    gDenR += gr * gr;
    gDenM += gm * gm;
  }
  const ncc = denR > 0 && denM > 0 ? Math.abs(num / Math.sqrt(denR * denM)) : 0;
  const edgeOverlap = gDenR > 0 && gDenM > 0 ? gDot / Math.sqrt(gDenR * gDenM) : 0;

  return {
    nmi: Number(nmi.toFixed(4)),
    ncc: Number(ncc.toFixed(4)),
    edgeOverlap: Number(edgeOverlap.toFixed(4)),
  };
}

function computeAndDisplayAlignmentMetrics(setAsBaselineIfEmpty = false) {
  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) return null;

  const refVol = state.nv.volumes[refIdx];
  const movVol = state.nv.volumes[movIdx];
  const metrics = computeMetricsInBrowser(refVol, movVol, null, 3);

  if (setAsBaselineIfEmpty && !state.baselineMetrics) {
    state.baselineMetrics = { ...metrics };
  }

  updateMetricsFooterUI(metrics);
  return metrics;
}

function updateMetricsFooterUI(metrics) {
  const nmiEl = document.getElementById("metricNMI");
  const nccEl = document.getElementById("metricNCC");
  const edgeEl = document.getElementById("metricEdge");
  if (!nmiEl || !metrics) return;

  nmiEl.textContent = metrics.nmi.toFixed(4);
  nccEl.textContent = metrics.ncc.toFixed(4);
  edgeEl.textContent = metrics.edgeOverlap.toFixed(4);

  const renderDelta = (elId, curr, base) => {
    const dEl = document.getElementById(elId);
    if (!dEl) return;
    if (base === undefined || base === null) {
      dEl.textContent = "";
      dEl.className = "metric-delta";
      return;
    }
    const diff = curr - base;
    if (Math.abs(diff) < 0.0002) {
      dEl.textContent = "0.0000";
      dEl.className = "metric-delta";
    } else if (diff > 0) {
      dEl.textContent = `+${diff.toFixed(4)}`;
      dEl.className = "metric-delta improved";
    } else {
      dEl.textContent = `${diff.toFixed(4)}`;
      dEl.className = "metric-delta degraded";
    }
  };

  if (state.baselineMetrics) {
    renderDelta("metricNMIDelta", metrics.nmi, state.baselineMetrics.nmi);
    renderDelta("metricNCCDelta", metrics.ncc, state.baselineMetrics.ncc);
    renderDelta("metricEdgeDelta", metrics.edgeOverlap, state.baselineMetrics.edgeOverlap);
  }
}

// Compute 3D Center of Mass in RAS World Coordinates for a volume
function computeVolumeCenterOfMassRAS(vol, affineOverride = null) {
  const aff = affineOverride || vol.hdr.affine;
  const dims = getVolDims(vol);
  const nx = dims[1], ny = dims[2], nz = dims[3];
  const img = vol.img;
  if (!img) return getVolumeCenterRAS(vol, aff);

  let sumW = 0, sumI = 0, sumJ = 0, sumK = 0;
  const sliceStride = nx * ny;
  for (let k = 0; k < nz; k += 2) {
    for (let j = 0; j < ny; j += 2) {
      for (let i = 0; i < nx; i += 2) {
        const val = img[k * sliceStride + j * nx + i];
        if (val > 0) {
          sumW += val;
          sumI += i * val;
          sumJ += j * val;
          sumK += k * val;
        }
      }
    }
  }
  if (sumW <= 0) return getVolumeCenterRAS(vol, aff);
  return applyMat4Pt(aff, [sumI / sumW, sumJ / sumW, sumK / sumW]);
}

function alignCentersOfMass() {
  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) {
    showToast("Select both a Reference and Moving volume first.");
    return;
  }
  const refVol = state.nv.volumes[refIdx];
  const movVol = state.nv.volumes[movIdx];
  const meta = state.layerMeta.get(movVol.id);
  if (!meta) return;

  const refCoM = computeVolumeCenterOfMassRAS(refVol);
  const movCoM = computeVolumeCenterOfMassRAS(movVol, meta.origAffine);

  meta.params.tx = Number((refCoM[0] - movCoM[0]).toFixed(2));
  meta.params.ty = Number((refCoM[1] - movCoM[1]).toFixed(2));
  meta.params.tz = Number((refCoM[2] - movCoM[2]).toFixed(2));

  applyMovingTransformFromParams(true);
  showToast(`Aligned 3D Centers of Mass (+ linked children): Δ = [${meta.params.tx}, ${meta.params.ty}, ${meta.params.tz}] mm`);
}

// ============================================================================
// Automated Multimodal Coregistration (In-Browser & Backend + Child Propagation)
// ============================================================================
async function runInBrowserAutoCoregistration(dof = 6) {
  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) {
    showToast("Please load at least 2 volumes (Reference & Moving).");
    return;
  }

  const refVol = state.nv.volumes[refIdx];
  const movVol = state.nv.volumes[movIdx];
  const meta = state.layerMeta.get(movVol.id);
  if (!meta) return;

  state.baselineMetrics = computeMetricsInBrowser(refVol, movVol, null, 3);
  updateMetricsFooterUI(state.baselineMetrics);

  const progBox = document.getElementById("coregProgressBox");
  const progBar = document.getElementById("coregProgressBar");
  const progPct = document.getElementById("coregProgressPct");
  const progTxt = document.getElementById("coregProgressText");
  progBox.hidden = false;
  setCoregStatus("Optimizing...", true);

  const pivotRAS = getVolumeCenterRAS(movVol, meta.origAffine);

  const evaluateParams = (p, stride = 4) => {
    const W = paramsToWorldMat4(p, pivotRAS);
    const candidateAff = multiplyMat4(W, meta.origAffine);
    const m = computeMetricsInBrowser(refVol, movVol, candidateAff, stride);
    return m.nmi + 0.35 * m.edgeOverlap;
  };

  const comRef = computeVolumeCenterOfMassRAS(refVol);
  const comMov = computeVolumeCenterOfMassRAS(movVol, meta.origAffine);
  const candidates = [
    { ...meta.params },
    { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 },
    {
      tx: Number((comRef[0] - comMov[0]).toFixed(1)),
      ty: Number((comRef[1] - comMov[1]).toFixed(1)),
      tz: Number((comRef[2] - comMov[2]).toFixed(1)),
      rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1,
    },
  ];

  let bestParams = candidates[0];
  let bestScore = -Infinity;
  for (const c of candidates) {
    const s = evaluateParams(c, 4);
    if (s > bestScore) {
      bestScore = s;
      bestParams = { ...c };
    }
  }

  const activeKeys = dof >= 9
    ? ["tx", "ty", "tz", "rz", "rx", "ry", "sx", "sy", "sz"]
    : ["tx", "ty", "tz", "rz", "rx", "ry"];

  const stages = [
    { label: "Coarse 3D Search", transStep: 8.0, rotStep: 8.0, scaleStep: 0.04, stride: 4 },
    { label: "Medium 3D Refinement", transStep: 4.0, rotStep: 4.0, scaleStep: 0.02, stride: 4 },
    { label: "Fine Anatomical Alignment", transStep: 2.0, rotStep: 2.0, scaleStep: 0.01, stride: 3 },
    { label: "Sub-voxel Convergence", transStep: 0.5, rotStep: 0.5, scaleStep: 0.005, stride: 3 },
  ];

  const totalSteps = stages.length * activeKeys.length;
  let completedSteps = 0;

  for (const stage of stages) {
    progTxt.textContent = `${stage.label}...`;
    for (const key of activeKeys) {
      const stepSize = key.startsWith("t")
        ? stage.transStep
        : key.startsWith("r")
        ? stage.rotStep
        : stage.scaleStep;

      let improvedInDir = true;
      let iterGuard = 0;
      while (improvedInDir && iterGuard < 4) {
        improvedInDir = false;
        iterGuard++;
        for (const dir of [-1, 1]) {
          const trial = { ...bestParams, [key]: Number((bestParams[key] + dir * stepSize).toFixed(3)) };
          const score = evaluateParams(trial, stage.stride);
          if (score > bestScore + 1e-4) {
            bestScore = score;
            bestParams = trial;
            improvedInDir = true;
          }
        }
      }

      meta.params = { ...bestParams };
      applyMovingTransformFromParams(false);

      completedSteps++;
      const pct = Math.min(100, Math.round((completedSteps / totalSteps) * 100));
      progBar.style.width = `${pct}%`;
      progPct.textContent = `${pct}%`;
      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  meta.params = { ...bestParams };
  applyMovingTransformFromParams(true);
  saveTransformSnapshot(`Auto NMI (${dof}-DOF)`);
  setCoregStatus("Aligned", false);
  setTimeout(() => {
    progBox.hidden = true;
  }, 1200);
  showToast("Automated 3D Coregistration complete (Moving + linked Child volumes aligned)!");
}

async function runBackendCoregistration(methodVal, dof) {
  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) return;

  const refVol = state.nv.volumes[refIdx];
  const movVol = state.nv.volumes[movIdx];
  const refMeta = state.layerMeta.get(refVol.id);
  const movMeta = state.layerMeta.get(movVol.id);

  if (!refMeta?.rawBytes || !movMeta?.rawBytes) {
    await runInBrowserAutoCoregistration(dof);
    return;
  }

  state.baselineMetrics = computeMetricsInBrowser(refVol, movVol, null, 3);
  updateMetricsFooterUI(state.baselineMetrics);

  const progBox = document.getElementById("coregProgressBox");
  const progBar = document.getElementById("coregProgressBar");
  const progPct = document.getElementById("coregProgressPct");
  const progTxt = document.getElementById("coregProgressText");
  progBox.hidden = false;
  progBar.style.width = "45%";
  progPct.textContent = "Running...";
  progTxt.textContent = methodVal.startsWith("fsl_flirt")
    ? "Running FSL FLIRT Coregistration..."
    : "Running Multi-Res 3D NMI Powell Optimizer...";
  setCoregStatus("Running...", true);

  try {
    const uint8ToB64 = (bytes) => {
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
      }
      return btoa(binary);
    };

    const mode = methodVal.startsWith("fsl_flirt") ? "fsl_flirt" : "nmi_powell";
    const cost = methodVal === "fsl_flirt_corratio" ? "corratio" : "mutualinfo";

    const resp = await fetch("/api/coregister", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        refName: refVol.name,
        movName: movVol.name,
        refBase64: uint8ToB64(refMeta.rawBytes),
        movBase64: uint8ToB64(movMeta.rawBytes),
        mode,
        cost,
        dof,
      }),
    });
    const res = await resp.json();
    if (!resp.ok) throw new Error(res.error || "Coregistration failed");

    progBar.style.width = "100%";
    progPct.textContent = "100%";

    if (res.worldMatrix) {
      let parsedParams = null;
      if (res.params && res.params.length >= 6) {
        parsedParams = {
          tx: res.params[0],
          ty: res.params[1],
          tz: res.params[2],
          rx: res.params[3],
          ry: res.params[4],
          rz: res.params[5],
          sx: res.params[6] || 1,
          sy: res.params[7] || 1,
          sz: res.params[8] || 1,
        };
      }
      applyWorldMatrixToTargets(res.worldMatrix, "mov_and_children", parsedParams);
      saveTransformSnapshot(res.engine || "Coregistered Transform", res.worldMatrix);
    }

    setCoregStatus("Aligned", false);
    showToast(`Coregistration complete via ${res.engine} (Moving + linked Child volumes updated)!`);
  } catch (err) {
    console.error("Backend coreg error, falling back to browser optimizer:", err);
    await runInBrowserAutoCoregistration(dof);
  } finally {
    setTimeout(() => {
      progBox.hidden = true;
    }, 1000);
  }
}

// ============================================================================
// Child Transform Management, .mat Import/Export & NIfTI File Export
// ============================================================================
function saveTransformSnapshot(customLabel = null, explicitWorldMat = null) {
  const movIdx = getVolumeIndexById(state.movVolId);
  if (movIdx < 0) return;
  const movMeta = state.layerMeta.get(state.movVolId);
  const W_ras = explicitWorldMat ? cloneMat4(explicitWorldMat) : getCurrentMovingWorldMatrix();
  const id = `tfm_${Date.now()}_${state.savedTransforms.length + 1}`;
  const name =
    customLabel ||
    `Transform #${state.savedTransforms.length + 1} (Tx=${movMeta?.params.tx ?? 0}, Rz=${movMeta?.params.rz ?? 0}°)`;

  state.savedTransforms.push({
    id,
    name,
    worldMat: W_ras,
    params: movMeta ? { ...movMeta.params } : null,
  });
  renderSavedTransformsDropdown();
  return id;
}

// Parse a 4x4 matrix from text (.mat or whitespace/comma-separated 16 numbers)
function parseMatrix4x4FromText(text) {
  const nums = text
    .trim()
    .split(/[\s,;]+/)
    .map((s) => parseFloat(s))
    .filter((n) => Number.isFinite(n));
  if (nums.length < 16) {
    throw new Error(`Expected 16 numbers for a 4×4 matrix, found ${nums.length}.`);
  }
  return [
    [nums[0], nums[1], nums[2], nums[3]],
    [nums[4], nums[5], nums[6], nums[7]],
    [nums[8], nums[9], nums[10], nums[11]],
    [nums[12], nums[13], nums[14], nums[15]],
  ];
}

async function handleImportMatFile(file) {
  try {
    const text = await file.text();
    const rawMat = parseMatrix4x4FromText(text);
    const convention = document.getElementById("selectImportedMatConvention")?.value || "world_ras";
    const targetMode = document.getElementById("selectApplyTarget")?.value || "mov_and_children";

    let W_ras = rawMat;
    if (convention === "fsl_flirt") {
      const refIdx = getVolumeIndexById(state.refVolId);
      const movIdx = getVolumeIndexById(state.movVolId);
      if (refIdx < 0 || movIdx < 0) {
        throw new Error("Reference and Moving volumes must be selected to convert an FSL FLIRT .mat matrix.");
      }
      W_ras = flirtMatToWorldRAS(rawMat, state.nv.volumes[refIdx], state.nv.volumes[movIdx]);
    }

    saveTransformSnapshot(`Imported: ${file.name} (${convention === "fsl_flirt" ? "FLIRT" : "RAS"})`, W_ras);
    applyWorldMatrixToTargets(W_ras, targetMode, null);
    showToast(`Imported ${file.name} and applied to ${targetMode.replace(/_/g, " ")}!`);
  } catch (err) {
    console.error("Failed to import .mat file:", err);
    showToast(`Import .mat Error: ${err.message}`, 4500);
  }
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function downloadBlobBytes(filename, uint8Bytes) {
  const blob = new Blob([uint8Bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function cloneNVImage(vol, newName = "cloned.nii.gz") {
  if (typeof vol.clone === "function") {
    try {
      const c = vol.clone();
      c.name = newName;
      return c;
    } catch (_) {
      // Fallback below
    }
  }
  const meta = state.layerMeta.get(vol.id);
  if (meta && meta.rawBytes) {
    const cloned = await loadNVImageFromBytes(meta.rawBytes, {
      name: newName,
      colormap: vol.colormap || "gray",
      opacity: vol.opacity ?? 1.0,
      cal_min: vol.cal_min,
      cal_max: vol.cal_max,
    });
    if (vol.hdr && cloned.hdr) {
      cloned.hdr.affine = cloneMat4(vol.hdr.affine);
    }
    return cloned;
  }
  throw new Error("Unable to clone volume");
}

// Export a single volume as NIfTI (.nii.gz or .nii) in either Header-Only or Resampled-to-Reference mode
async function exportSingleVolumeAsNifti(volId) {
  const volIdx = getVolumeIndexById(volId);
  if (volIdx < 0) return;
  const vol = state.nv.volumes[volIdx];
  const meta = state.layerMeta.get(volId);
  const refIdx = getVolumeIndexById(state.refVolId);
  const refVol = refIdx >= 0 ? state.nv.volumes[refIdx] : null;
  const refMeta = refVol ? state.layerMeta.get(refVol.id) : null;

  const exportMode = document.getElementById("selectExportMode")?.value || "header_only";
  const interpSetting = document.getElementById("selectExportInterp")?.value || "auto";
  const formatSetting = document.getElementById("selectExportFormat")?.value || "nii_gz";
  const compress = formatSetting === "nii_gz";

  const interp =
    interpSetting === "auto"
      ? meta?.isMask
        ? "nearest"
        : "trilinear"
      : interpSetting;

  const baseClean = (vol.name || "volume").replace(/\.nii\.gz$/i, "").replace(/\.nii$/i, "");
  const suffix = exportMode === "resample_to_ref" ? "_coreg_inRefSpace" : "_coreg_sform";
  const ext = compress ? ".nii.gz" : ".nii";
  const outFilename = `${baseClean}${suffix}${ext}`;

  const uint8ToB64 = (bytes) => {
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  };

  // 1. Try high-precision Nibabel/SciPy backend (/api/export-nifti) if source rawBytes are available
  if (meta?.rawBytes && (exportMode === "header_only" || refMeta?.rawBytes)) {
    try {
      const resp = await fetch("/api/export-nifti", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sourceBase64: uint8ToB64(meta.rawBytes),
          refBase64: refMeta?.rawBytes ? uint8ToB64(refMeta.rawBytes) : null,
          newAffine: vol.hdr.affine,
          exportMode,
          interp,
          compress,
        }),
      });
      const data = await resp.json();
      if (resp.ok && data.niftiBase64) {
        const outBytes = base64ToUint8Array(data.niftiBase64);
        downloadBlobBytes(outFilename, outBytes);
        return outFilename;
      }
    } catch (e) {
      console.info("Falling back to in-browser NIfTI export:", e);
    }
  }

  // 2. Pure In-Browser NIfTI Export (supports both Header-Only and Resampled-to-Reference Grid!)
  let targetVolToSave = vol;
  if (exportMode === "resample_to_ref" && refVol) {
    const resampledVol = await cloneNVImage(refVol, outFilename);
    const rDims = getVolDims(resampledVol);
    const nx = rDims[1], ny = rDims[2], nz = rDims[3];
    const slice = nx * ny;
    const refAff = refVol.hdr.affine;
    const invSrcAff = invertMat4(vol.hdr.affine);
    const useNearest = interp === "nearest";

    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const ras = applyMat4Pt(refAff, [i, j, k]);
          resampledVol.img[k * slice + j * nx + i] = sampleVolumeAtRAS(vol, invSrcAff, ras, useNearest);
        }
      }
    }
    targetVolToSave = resampledVol;
  }

  const browserFilename = outFilename.replace(/\.gz$/i, "");
  if (typeof targetVolToSave.saveToDisk === "function") {
    targetVolToSave.saveToDisk(browserFilename);
  } else {
    state.nv.saveImage({ filename: browserFilename, isSaveDrawing: false, volumeByIndex: volIdx });
  }
  return browserFilename;
}

async function exportSelectedVolumesAsNifti() {
  const checkedBoxes = Array.from(document.querySelectorAll("input[data-export-check]:checked"));
  if (checkedBoxes.length === 0) {
    showToast("Check at least one Moving or Child volume to export.");
    return;
  }

  setCoregStatus("Exporting...", true);
  const exportedNames = [];
  for (const cb of checkedBoxes) {
    const volId = cb.getAttribute("data-export-check");
    if (volId) {
      const fname = await exportSingleVolumeAsNifti(volId);
      if (fname) exportedNames.push(fname);
      await new Promise((r) => setTimeout(r, 180));
    }
  }
  setCoregStatus("Ready", false);
  showToast(`Exported ${exportedNames.length} NIfTI file(s): ${exportedNames.join(", ")}`, 4500);
}

// ============================================================================
// Verification Overlays: 3D Sobel Edges, 3D Checkerboard, Red/Green Anaglyph, Blink
// ============================================================================
async function toggle3DEdgeContourOverlay() {
  const btn = document.getElementById("btnEdgeOverlay");
  const existingIdx = state.nv.volumes.findIndex((v) => {
    const m = state.layerMeta.get(v.id);
    return m && m.isEdgeOfMov;
  });

  if (existingIdx >= 0) {
    const existingVol = state.nv.volumes[existingIdx];
    state.nv.removeVolumeByIndex(existingIdx);
    state.layerMeta.delete(existingVol.id);
    btn.classList.remove("active");
    renderLayerList();
    showToast("Removed 3D Edge Contour verification layer.");
    return;
  }

  const movIdx = getVolumeIndexById(state.movVolId);
  if (movIdx < 0) {
    showToast("Select a Moving volume first.");
    return;
  }
  const movVol = state.nv.volumes[movIdx];
  const movMeta = state.layerMeta.get(movVol.id);
  showToast("Computing 3D Sobel anatomical edge contours...", 2000);

  const edgeVol = await cloneNVImage(movVol, `Edges_${movVol.name}`);
  const eDims = getVolDims(edgeVol);
  const nx = eDims[1], ny = eDims[2], nz = eDims[3];
  const src = movVol.img;
  const dst = edgeVol.img;
  const slice = nx * ny;
  dst.fill(0);

  let maxGrad = 0;
  for (let k = 1; k < nz - 1; k++) {
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const idx = k * slice + j * nx + i;
        const gx = src[idx + 1] - src[idx - 1];
        const gy = src[idx + nx] - src[idx - nx];
        const gz = src[idx + slice] - src[idx - slice];
        const mag = Math.hypot(gx, gy, gz);
        dst[idx] = mag;
        if (mag > maxGrad) maxGrad = mag;
      }
    }
  }

  edgeVol.colormap = "green";
  edgeVol.opacity = 0.85;
  edgeVol.cal_min = maxGrad * 0.18;
  edgeVol.cal_max = maxGrad * 0.75;

  state.nv.addVolume(edgeVol);
  registerVolumeMeta(edgeVol, {
    label: `3D Edges (${movMeta?.label || movVol.name})`,
    modality: "Edge Contours",
    opacity: 0.85,
    isSyntheticVerification: true,
    isEdgeOfMov: true,
  });
  // Ensure origAffine of edgeVol matches movMeta.origAffine so W_ras propagation stays exact
  const edgeMeta = state.layerMeta.get(edgeVol.id);
  if (edgeMeta && movMeta) {
    edgeMeta.origAffine = cloneMat4(movMeta.origAffine);
  }

  btn.classList.add("active");
  renderLayerList();
  showToast("Added 3D Edge Contour verification overlay (neon green).");
}

async function toggle3DCheckerboardOverlay() {
  const btn = document.getElementById("btnCheckerboard");
  const existingIdx = state.nv.volumes.findIndex((v) => {
    const m = state.layerMeta.get(v.id);
    return m && m.isCheckerboard;
  });

  if (existingIdx >= 0) {
    const existingVol = state.nv.volumes[existingIdx];
    state.nv.removeVolumeByIndex(existingIdx);
    state.layerMeta.delete(existingVol.id);
    btn.classList.remove("active");
    renderLayerList();
    showToast("Removed 3D Checkerboard fusion volume.");
    return;
  }

  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) {
    showToast("Need both Reference and Moving volumes for 3D Checkerboard.");
    return;
  }

  const refVol = state.nv.volumes[refIdx];
  const movVol = state.nv.volumes[movIdx];
  const tileVox = parseInt(document.getElementById("checkerSizeSelect").value || "9", 10);

  const checkerVol = await cloneNVImage(refVol, "3D_Checkerboard_Ref_Mov.nii.gz");
  const cDims = getVolDims(checkerVol);
  const nx = cDims[1], ny = cDims[2], nz = cDims[3];
  const slice = nx * ny;

  const refAff = refVol.hdr.affine;
  const invMovAff = invertMat4(movVol.hdr.affine);

  const rMin = refVol.cal_min || 0;
  const rMax = refVol.cal_max || 1000;
  const mMin = movVol.cal_min || 0;
  const mMax = movVol.cal_max || 1000;
  const rSpan = rMax - rMin || 1;
  const mSpan = mMax - mMin || 1;

  for (let k = 0; k < nz; k++) {
    const kz = Math.floor(k / tileVox);
    for (let j = 0; j < ny; j++) {
      const jy = Math.floor(j / tileVox);
      for (let i = 0; i < nx; i++) {
        const ix = Math.floor(i / tileVox);
        const idx = k * slice + j * nx + i;
        if ((ix + jy + kz) % 2 === 1) {
          const ras = applyMat4Pt(refAff, [i, j, k]);
          const mVal = sampleVolumeAtRAS(movVol, invMovAff, ras, false);
          const norm = Math.max(0, Math.min(1, (mVal - mMin) / mSpan));
          checkerVol.img[idx] = rMin + norm * rSpan;
        }
      }
    }
  }

  checkerVol.colormap = "gray";
  checkerVol.opacity = 1.0;
  checkerVol.cal_min = rMin;
  checkerVol.cal_max = rMax;

  state.nv.addVolume(checkerVol);
  registerVolumeMeta(checkerVol, {
    label: `3D Checkerboard (Ref ⊞ Mov)`,
    modality: "Checkerboard",
    opacity: 1.0,
    isSyntheticVerification: true,
    isCheckerboard: true,
  });

  btn.classList.add("active");
  renderLayerList();
  showToast("Generated 3D RAS Checkerboard fusion volume.");
}

function toggleAnaglyphFusion() {
  const btn = document.getElementById("btnAnaglyphMode");
  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) return;

  const refVol = state.nv.volumes[refIdx];
  const movVol = state.nv.volumes[movIdx];

  if (!state.anaglyphActive) {
    state.savedPreAnaglyph.set(refVol.id, { colormap: refVol.colormap, opacity: refVol.opacity });
    state.savedPreAnaglyph.set(movVol.id, { colormap: movVol.colormap, opacity: movVol.opacity });
    refVol.colormap = "red";
    movVol.colormap = "green";
    state.nv.setOpacity(refIdx, 1.0);
    state.nv.setOpacity(movIdx, 0.55);
    state.nv.updateGLVolume();
    state.nv.drawScene();
    state.anaglyphActive = true;
    btn.classList.add("active");
    renderLayerList();
    showToast("Red/Green Complementary Fusion active (overlapping structures appear yellow/white).");
  } else {
    const rSaved = state.savedPreAnaglyph.get(refVol.id);
    const mSaved = state.savedPreAnaglyph.get(movVol.id);
    if (rSaved) {
      refVol.colormap = rSaved.colormap;
      state.nv.setOpacity(refIdx, rSaved.opacity);
    }
    if (mSaved) {
      movVol.colormap = mSaved.colormap;
      state.nv.setOpacity(movIdx, mSaved.opacity);
    }
    state.nv.updateGLVolume();
    state.nv.drawScene();
    state.anaglyphActive = false;
    btn.classList.remove("active");
    renderLayerList();
    showToast("Restored standard volume colormaps.");
  }
}

function stepBlinkComparator() {
  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  if (refIdx < 0 || movIdx < 0) return;

  state.blinkPhaseRef = !state.blinkPhaseRef;
  const refMeta = state.layerMeta.get(state.refVolId);
  const movMeta = state.layerMeta.get(state.movVolId);

  if (state.blinkPhaseRef) {
    if (refMeta) refMeta.visible = true;
    if (movMeta) movMeta.visible = false;
    state.nv.setOpacity(refIdx, 1.0);
    state.nv.setOpacity(movIdx, 0.0);
  } else {
    if (refMeta) refMeta.visible = true;
    if (movMeta) movMeta.visible = true;
    state.nv.setOpacity(refIdx, 0.2);
    state.nv.setOpacity(movIdx, 1.0);
  }
  state.nv.drawScene();
  renderLayerList();
}

function toggleBlinkMode() {
  const btn = document.getElementById("btnBlinkToggle");
  const speedMs = parseInt(document.getElementById("blinkSpeedSelect").value || "0", 10);

  if (speedMs === 0) {
    stepBlinkComparator();
    btn.classList.add("active");
    setTimeout(() => btn.classList.remove("active"), 250);
    return;
  }

  if (state.blinkActive) {
    clearInterval(state.blinkTimer);
    state.blinkTimer = null;
    state.blinkActive = false;
    btn.classList.remove("active");
    const refIdx = getVolumeIndexById(state.refVolId);
    const movIdx = getVolumeIndexById(state.movVolId);
    if (refIdx >= 0) toggleVolumeVisibility(state.refVolId, true);
    if (movIdx >= 0) toggleVolumeVisibility(state.movVolId, true);
    showToast("Stopped auto-blink comparator.");
  } else {
    state.blinkActive = true;
    btn.classList.add("active");
    stepBlinkComparator();
    state.blinkTimer = setInterval(stepBlinkComparator, speedMs);
    showToast("Auto-blink Ref ↔ Mov active (press Space or click button to stop).");
  }
}

// ============================================================================
// Data Loading: DICOM Folders (WASM + Server) & NIfTI Files
// ============================================================================
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result);
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
      }
      resolve(btoa(binary));
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

function base64ToUint8Array(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    out[i] = bin.charCodeAt(i);
  }
  return out;
}

async function loadDicomFolderFiles(fileList, useSampleServerDicom = false) {
  try {
    setCoregStatus("Converting DICOM...", true);
    showToast("Converting DICOM series to 3D NIfTI via dcm2niix...", 0);

    if (!useSampleServerDicom && fileList && fileList.length > 0) {
      try {
        const dcmExt = await import("@niivue/nv-ext-dcm2niix");
        if (dcmExt && typeof dcmExt.runDcm2niix === "function") {
          const niftiUrls = await dcmExt.runDcm2niix(fileList);
          if (niftiUrls && niftiUrls.length > 0) {
            for (const urlOrFile of niftiUrls) {
              let wasmRawBytes = null;
              let tempUrl = null;
              try {
                if (typeof urlOrFile === "string") {
                  const r = await fetch(urlOrFile);
                  wasmRawBytes = new Uint8Array(await r.arrayBuffer());
                } else if (urlOrFile && typeof urlOrFile.arrayBuffer === "function") {
                  wasmRawBytes = new Uint8Array(await urlOrFile.arrayBuffer());
                }
              } catch (_) {
                // Fallback to direct URL load if fetch/arrayBuffer not available
              }

              const volName = `DICOM_WASM_${Date.now()}.nii.gz`;
              let vol;
              if (wasmRawBytes) {
                vol = await loadNVImageFromBytes(wasmRawBytes, {
                  name: volName,
                  colormap: "viridis",
                  opacity: 0.8,
                });
              } else {
                tempUrl = typeof urlOrFile === "string" ? urlOrFile : URL.createObjectURL(urlOrFile);
                vol = await NVImageClass.loadFromUrl({
                  url: tempUrl,
                  name: volName,
                  colormap: "viridis",
                  opacity: 0.8,
                });
                if (typeof urlOrFile !== "string" && tempUrl) {
                  setTimeout(() => URL.revokeObjectURL(tempUrl), 1000);
                }
              }

              state.nv.addVolume(vol);
              registerVolumeMeta(vol, {
                label: `DICOM Series (WASM dcm2niix)`,
                modality: "DICOM MR/CT",
                opacity: 0.8,
                rawBytes: wasmRawBytes,
              });
            }
            renderLayerList();
            computeAndDisplayAlignmentMetrics(true);
            setCoregStatus("Ready", false);
            showToast("Loaded DICOM folder via in-browser WebAssembly dcm2niix!");
            return;
          }
        }
      } catch (wasmErr) {
        console.info("Falling back to local dcm2niix server endpoint:", wasmErr);
      }
    }

    let payload = {};
    if (useSampleServerDicom) {
      payload = { useSampleDicom: true };
    } else {
      const dcmFiles = Array.from(fileList).filter(
        (f) => !f.name.startsWith(".") && (f.name.toLowerCase().endsWith(".dcm") || !f.name.includes("."))
      );
      if (dcmFiles.length === 0) {
        showToast("No DICOM (.dcm) files found in the selected folder.");
        setCoregStatus("Ready", false);
        return;
      }
      const encodedFiles = await Promise.all(
        dcmFiles.map(async (f) => ({
          name: f.name,
          dataBase64: await fileToBase64(f),
        }))
      );
      payload = { files: encodedFiles };
    }

    const resp = await fetch("/api/convert-dicom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || !data.volumes || data.volumes.length === 0) {
      throw new Error(data.error || "Failed to convert DICOM series");
    }

    for (const vInfo of data.volumes) {
      const niiBytes = base64ToUint8Array(vInfo.niftiBase64);
      const seriesDesc = vInfo.metadata?.SeriesDescription || vInfo.name;
      const modalityTag = vInfo.metadata?.Modality || "DICOM";

      const vol = await loadNVImageFromBytes(niiBytes, {
        name: vInfo.name,
        colormap: "cool",
        opacity: 0.75,
      });
      state.nv.addVolume(vol);
      registerVolumeMeta(vol, {
        label: `DICOM: ${seriesDesc}`,
        modality: `${modalityTag} (${vInfo.shape.join("×")})`,
        opacity: 0.75,
        rawBytes: niiBytes,
      });
    }

    renderLayerList();
    computeAndDisplayAlignmentMetrics(true);
    setCoregStatus("Ready", false);
    showToast(`Converted & loaded ${data.volumes.length} DICOM volume(s) (${data.volumes[0].name})!`);
  } catch (err) {
    console.error("DICOM load error:", err);
    setCoregStatus("Error", false);
    showToast(`DICOM Load Error: ${err.message}`, 4500);
  }
}

async function loadLocalNiftiFiles(fileList) {
  const files = Array.from(fileList);
  if (files.length === 0) return;

  // If user dropped or selected a saved NeuroCoreg project folder containing project.json
  if (files.some((f) => f.name === "project.json" || f.name.endsWith(".neurocoreg.json"))) {
    await loadProjectFromFileList(files);
    return;
  }

  if (files.some((f) => f.name.toLowerCase().endsWith(".dcm"))) {
    await loadDicomFolderFiles(files, false);
    return;
  }

  // Load anatomical volumes first so any accompanying .voi files snap onto the Reference grid
  const orderedFiles = [
    ...files.filter((f) => !f.name.toLowerCase().endsWith(".voi")),
    ...files.filter((f) => f.name.toLowerCase().endsWith(".voi")),
  ];

  let importedVoiCount = 0;
  for (const file of orderedFiles) {
    try {
      if (file.name.toLowerCase().endsWith(".voi")) {
        if (typeof importFileAsDrawingLayer === "function") {
          const imported = await importFileAsDrawingLayer(file, { openStudio: true, quiet: true });
          if (imported) importedVoiCount++;
        }
        continue;
      }

      showToast(`Loading ${file.name}...`, 2000);
      const ab = await file.arrayBuffer();
      let rawBytes = new Uint8Array(ab);
      let fileName = file.name;
      const inferred = inferModalityAndStyle(fileName);

      // Convert FreeSurfer .mgz / .mgh via backend nibabel endpoint if needed
      if (fileName.toLowerCase().endsWith(".mgz") || fileName.toLowerCase().endsWith(".mgh")) {
        const resp = await fetch("/api/convert-volume", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sourceName: fileName,
            sourceBase64: uint8ArrayToBase64(rawBytes),
            isDktAtlas: Boolean(inferred.isDktAtlas),
          }),
        });
        const convData = await resp.json();
        if (!resp.ok || !convData.niftiBase64) {
          throw new Error(convData.error || "Failed to convert FreeSurfer volume");
        }
        rawBytes = base64ToUint8Array(convData.niftiBase64);
        fileName = convData.fileName || `${fileName.replace(/\.(mgz|mgh)$/i, "")}.nii.gz`;
        if (convData.dktLut) state.dktLut = convData.dktLut;
      }

      if (inferred.isDktAtlas || inferred.isAtlas) {
        await addAtlasVolumeToWorkspace(rawBytes, {
          fileName,
          displayLabel: `${inferred.atlasShortName || "Atlas"} (${file.name.replace(/\.(nii\.gz|nii|mgz|mgh)$/i, "")})`,
          atlasId: inferred.atlasId || "dkt",
          atlasName: inferred.modality,
          atlasShortName: inferred.atlasShortName || "DKT",
          atlasCategory: inferred.atlasCategory || "cortical",
          atlasBadge: inferred.atlasBadge || "DKT",
          isFreeSurferAtlas: Boolean(inferred.isFreeSurferAtlas),
        });
        continue;
      }

      const vol = await loadNVImageFromBytes(rawBytes, {
        name: fileName,
        colormap: inferred.colormap,
        opacity: inferred.opacity,
      });
      state.nv.addVolume(vol);
      registerVolumeMeta(vol, {
        label: file.name.replace(/\.(nii\.gz|nii|mgz|mgh)$/i, ""),
        modality: inferred.modality,
        opacity: inferred.opacity,
        isMask: inferred.isMask,
        rawBytes,
      });
    } catch (err) {
      console.error("Failed to load file:", file.name, err);
      showToast(`Failed to load ${file.name}: ${err.message}`, 4000);
    }
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  renderLayerList();
  computeAndDisplayAlignmentMetrics(true);
  if (importedVoiCount > 0 && orderedFiles.length === importedVoiCount) {
    showToast(`Imported ${importedVoiCount} .voi file(s) as editable Drawing Layer(s).`);
  } else {
    showToast(`Loaded ${orderedFiles.length} file(s).`);
  }
}

async function loadSampleMultimodalSubject() {
  try {
    setCoregStatus("Loading Subject...", true);
    showToast("Loading single-subject multimodal dataset (T1w + Misaligned CT + Child SEEG Mask + T2-FLAIR + DTI FA)...", 0);

    const resp = await fetch("/api/sample-manifest");
    const manifest = await resp.json();

    createNewPatientWorkspace(false, "sub-01");

    for (const item of manifest.niftiVolumes) {
      const fileResp = await fetch(item.url);
      const ab = await fileResp.arrayBuffer();
      const rawBytes = new Uint8Array(ab);

      const vol = await loadNVImageFromBytes(rawBytes, {
        name: item.name,
        colormap: item.colormap,
        opacity: item.visible === false ? 0.0 : item.opacity,
        cal_min: item.cal_min,
        cal_max: item.cal_max,
      });

      state.nv.addVolume(vol);
      registerVolumeMeta(vol, {
        label: item.label,
        modality: item.modality,
        opacity: item.opacity,
        visible: item.visible !== false,
        role: item.role,
        isChildOfMov: item.role === "child",
        isMask: Boolean(item.isMask),
        rawBytes,
      });

      if (item.role === "reference") state.refVolId = vol.id;
      if (item.role === "moving") state.movVolId = vol.id;
    }

    state.nv.updateGLVolume();
    state.nv.drawScene();
    renderLayerList();

    state.baselineMetrics = null;
    computeAndDisplayAlignmentMetrics(true);
    reset2DZoom();
    setCoregStatus("Ready", false);
    showToast("Loaded Subject sub-01: T1w (Ref) + Misaligned CT (Mov) + Linked Child SEEG Contacts Mask!");
  } catch (err) {
    console.error("Failed to load sample subject:", err);
    setCoregStatus("Error", false);
    showToast(`Error loading sample dataset: ${err.message}`, 4000);
  }
}

// ============================================================================
// New Patient Workspace Reset & Project Save / Open (Folder of User's Choice)
// ============================================================================
function createNewPatientWorkspace(confirmIfLoaded = true, presetPatientId = null) {
  if (!state.nv) return false;
  const patientInput = document.getElementById("patientIdInput");
  const currentId = patientInput?.value?.trim() || "sub-01";

  if (confirmIfLoaded && state.nv.volumes.length > 0) {
    const match = currentId.match(/^(.*?)(\d+)$/);
    const suggestedNextId = match
      ? `${match[1]}${String(parseInt(match[2], 10) + 1).padStart(match[2].length, "0")}`
      : "sub-02";

    const enteredId = window.prompt(
      "Start a New Patient workspace?\nAll currently loaded volumes, overlays, and unsaved transformations will be cleared.\n\nEnter New Patient / Subject ID:",
      suggestedNextId
    );
    if (enteredId === null) {
      return false; // User cancelled
    }
    if (patientInput && enteredId.trim()) {
      patientInput.value = enteredId.trim();
    }
  } else if (presetPatientId && patientInput) {
    patientInput.value = presetPatientId;
  }

  // Stop any active blink timer or anaglyph state
  if (state.blinkActive && state.blinkTimer) {
    clearInterval(state.blinkTimer);
    state.blinkTimer = null;
    state.blinkActive = false;
  }
  state.anaglyphActive = false;
  state.savedPreAnaglyph.clear();

  ["btnBlinkToggle", "btnEdgeOverlay", "btnCheckerboard", "btnAnaglyphMode"].forEach((btnId) => {
    document.getElementById(btnId)?.classList.remove("active");
  });

  // Remove all volumes from NiiVue
  while (state.nv.volumes.length > 0) {
    state.nv.removeVolumeByIndex(0);
  }
  state.layerMeta.clear();
  state.refVolId = null;
  state.movVolId = null;
  state.soloVolId = null;
  state.activeDktVolId = null;
  state.activeAtlasVolId = null;
  state.atlasStudioState.selectedAtlasVolId = null;
  state.atlasStudioState.warpCacheHit = null;
  state.baselineMetrics = null;
  state.savedTransforms = [];
  state.zoom2d.panCalib = null;
  if (typeof resetDrawingWorkspace === "function") {
    resetDrawingWorkspace();
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  reset2DZoom();

  renderSavedTransformsDropdown();
  renderLayerList();
  resetDktInspectorUI();
  syncAtlasStudioUI();

  // Reset voxel inspector & quantitative metrics footer
  const voxelTable = document.getElementById("voxelReadoutTable");
  if (voxelTable) {
    voxelTable.innerHTML = `<div class="voxel-empty">Load a volume to inspect voxel values</div>`;
  }
  ["metricNMI", "metricNCC", "metricEdge"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "—";
  });
  ["metricNMIDelta", "metricNCCDelta", "metricEdgeDelta"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = "";
      el.className = "metric-delta";
    }
  });

  setCoregStatus("Ready", false);
  if (confirmIfLoaded) {
    const activePatient = patientInput?.value?.trim() || "new-patient";
    showToast(`Cleared workspace for New Patient (${activePatient}). Ready to load NIfTI or DICOM data!`);
  }
  return true;
}

function uint8ArrayToBase64(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function formatMat4AsText(mat) {
  return mat.map((row) => row.map((v) => Number(v).toFixed(6)).join("  ")).join("\n") + "\n";
}

// Build complete workspace project manifest + transform files + volume metadata
function buildProjectBundleData() {
  const patientId = document.getElementById("patientIdInput")?.value?.trim() || "sub-01";
  const safePatient = patientId.replace(/[^a-zA-Z0-9._-]/g, "_") || "sub-01";

  const realVolumes = state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isSyntheticVerification && !m.isDrawingLayer;
  });

  const refIdx = getVolumeIndexById(state.refVolId);
  const movIdx = getVolumeIndexById(state.movVolId);
  const refVol = refIdx >= 0 ? state.nv.volumes[refIdx] : null;
  const movVol = movIdx >= 0 ? state.nv.volumes[movIdx] : null;

  const currentMetrics = computeAndDisplayAlignmentMetrics(false);
  const W_ras = getCurrentMovingWorldMatrix();
  const flirtMat = refVol && movVol ? worldRASToFlirtMat(W_ras, refVol, movVol) : identityMat4();

  const usedNames = new Set();
  const volumeEntries = realVolumes.map((vol, idx) => {
    const m = state.layerMeta.get(vol.id);
    let base = (vol.name || `volume_${idx + 1}`)
      .replace(/\.nii\.gz$/i, "")
      .replace(/\.nii$/i, "")
      .replace(/[^a-zA-Z0-9._-]/g, "_");
    let fileName = `${base}.nii.gz`;
    let counter = 2;
    while (usedNames.has(fileName)) {
      fileName = `${base}_${counter}.nii.gz`;
      counter++;
    }
    usedNames.add(fileName);

    const role =
      vol.id === state.refVolId
        ? "reference"
        : vol.id === state.movVolId
        ? "moving"
        : m?.isChildOfMov
        ? "child"
        : "overlay";

    return {
      vol,
      meta: m,
      entry: {
        fileName,
        name: vol.name,
        label: m?.label || vol.name,
        modality: m?.modality || "Neuro Volume",
        role,
        isChildOfMov: Boolean(m?.isChildOfMov),
        isCheckedForChildApply: Boolean(m?.isCheckedForChildApply),
        isCheckedForExport: Boolean(m?.isCheckedForExport),
        isMask: Boolean(m?.isMask),
        isAtlas: Boolean(m?.isAtlas || m?.isDktAtlas),
        isDktAtlas: Boolean(m?.isDktAtlas || m?.isAtlas),
        atlasType: m?.atlasType || null,
        atlasId: m?.atlasId || m?.atlasType || null,
        atlasName: m?.atlasName || m?.label || null,
        atlasShortName: m?.atlasShortName || null,
        atlasCategory: m?.atlasCategory || null,
        atlasBadge: m?.atlasBadge || null,
        isFreeSurferAtlas: m?.isFreeSurferAtlas !== undefined ? Boolean(m.isFreeSurferAtlas) : null,
        userAtlasOpacity: m?.userAtlasOpacity ?? null,
        atlasEngine: m?.atlasEngine || null,
        atlasSpace: m?.atlasSpace || null,
        visibleRegionIds: m?.visibleRegionIds instanceof Set ? Array.from(m.visibleRegionIds) : null,
        customLut: m?.lut && (!m.isFreeSurferAtlas || m.atlasCategory === "custom") ? m.lut : null,
        visible: m ? Boolean(m.visible) : true,
        opacity: vol.opacity,
        savedOpacity: m?.savedOpacity ?? 0.75,
        colormap: vol.colormap,
        cal_min: vol.cal_min,
        cal_max: vol.cal_max,
        origAffine: m ? cloneMat4(m.origAffine) : cloneMat4(vol.hdr.affine),
        currentAffine: cloneMat4(vol.hdr.affine),
        params: m?.params ? { ...m.params } : { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 },
      },
    };
  });

  const usedDrawingNames = new Set();
  const drawingEntries = (state.drawing?.layers || []).map((layer, idx) => {
    const rawBase = (layer.name || `drawing_${idx + 1}`).replace(/[^a-zA-Z0-9._-]/g, "_");
    let fileName = `${rawBase}.voi`;
    let counter = 2;
    while (usedDrawingNames.has(fileName)) {
      fileName = `${rawBase}_${counter}.voi`;
      counter++;
    }
    usedDrawingNames.add(fileName);
    return {
      layer,
      entry: {
        fileName,
        id: layer.id,
        name: layer.name,
        mode: layer.mode,
        colorHex: layer.colorHex,
        opacity: layer.opacity,
        visible: Boolean(layer.visible),
        activeLabel: layer.activeLabel || 1,
        labelPalette: Array.isArray(layer.labelPalette) ? layer.labelPalette.map((p) => ({ ...p })) : [],
        dims: [...layer.dims],
        affine: cloneMat4(layer.affine),
      },
    };
  });

  const manifest = {
    schema: "neurocoreg-project-v1",
    savedAt: new Date().toISOString(),
    patientId,
    viewState: {
      coregPanelOpen: state.coregPanelOpen,
      hudPinned: state.hudPinned,
      crosshairWidth: state.nv?.opts?.crosshairWidth ?? 1,
      pan2Dxyzmm: getCurrentPan2D(),
    },
    baselineMetrics: state.baselineMetrics,
    currentMetrics,
    activeMovingWorldMatrix: W_ras,
    activeMovingFlirtMatrix: flirtMat,
    atlasOverlayStyle: state.atlasOverlayStyle || "filled",
    savedTransforms: state.savedTransforms,
    volumes: volumeEntries.map((ve) => ve.entry),
    drawingLayers: drawingEntries.map((de) => de.entry),
  };

  const transformFiles = [
    {
      fileName: "active_moving_world_ras.mat",
      content: formatMat4AsText(W_ras),
    },
    {
      fileName: "active_moving_fsl_flirt.mat",
      content: formatMat4AsText(flirtMat),
    },
    ...state.savedTransforms.map((t, i) => ({
      fileName: `saved_${i + 1}_${(t.name || "transform").replace(/[^a-zA-Z0-9._-]/g, "_")}_world_ras.mat`,
      content: formatMat4AsText(t.worldMat),
    })),
  ];

  return { patientId, safePatient, manifest, volumeEntries, drawingEntries, transformFiles };
}

// Save Project to a Folder of the User's Choice (returns true if saved, false if cancelled/failed)
async function saveProjectToUserFolder() {
  if (!state.nv) return false;
  const { patientId, safePatient, manifest, volumeEntries, drawingEntries, transformFiles } = buildProjectBundleData();
  if (volumeEntries.length === 0 && drawingEntries.length === 0) {
    showToast("No volumes loaded in the workspace to save.");
    return false;
  }

  try {
    setCoregStatus("Saving Project...", true);

    // 1. Browser File System Access API (window.showDirectoryPicker) — opens native OS folder picker in Chrome/Edge/Brave
    if (typeof window.showDirectoryPicker === "function") {
      let chosenDirHandle = null;
      try {
        chosenDirHandle = await window.showDirectoryPicker({ mode: "readwrite" });
      } catch (pickerErr) {
        if (pickerErr && pickerErr.name === "AbortError") {
          setCoregStatus("Ready", false);
          return false;
        }
        console.info("showDirectoryPicker unavailable in this context, falling back to server save:", pickerErr);
      }

      if (chosenDirHandle) {
        try {
          showToast(`Saving workspace bundle (${volumeEntries.length} NIfTI volumes, ${drawingEntries.length} VOI layers)...`, 0);

          // Create <patientId>_project subfolder unless the user already selected a folder named for this patient
          const dirName = chosenDirHandle.name || "";
          const projDirHandle =
            dirName === `${safePatient}_project` || dirName === safePatient
              ? chosenDirHandle
              : await chosenDirHandle.getDirectoryHandle(`${safePatient}_project`, { create: true });

          const niftiDirHandle = await projDirHandle.getDirectoryHandle("nifti", { create: true });
          const tfmDirHandle = await projDirHandle.getDirectoryHandle("transforms", { create: true });

          // Write project.json
          const projFileHandle = await projDirHandle.getFileHandle("project.json", { create: true });
          const projWritable = await projFileHandle.createWritable();
          await projWritable.write(JSON.stringify(manifest, null, 2));
          await projWritable.close();

          // Write transform .mat files
          for (const tfm of transformFiles) {
            const tfmHandle = await tfmDirHandle.getFileHandle(tfm.fileName, { create: true });
            const tfmWritable = await tfmHandle.createWritable();
            await tfmWritable.write(tfm.content);
            await tfmWritable.close();
          }

          // Write each NIfTI volume (.nii.gz) with updated coregistered sform/qform affine header
          for (const { vol, meta, entry } of volumeEntries) {
            let outBytes = meta?.rawBytes || null;
            if (meta?.rawBytes) {
              try {
                const resp = await fetch("/api/export-nifti", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    sourceBase64: uint8ArrayToBase64(meta.rawBytes),
                    newAffine: vol.hdr.affine,
                    exportMode: "header_only",
                    compress: true,
                  }),
                });
                const data = await resp.json();
                if (resp.ok && data.niftiBase64) {
                  outBytes = base64ToUint8Array(data.niftiBase64);
                }
              } catch (_) {
                // Use meta.rawBytes fallback
              }
            }
            if (outBytes) {
              const niiHandle = await niftiDirHandle.getFileHandle(entry.fileName, { create: true });
              const niiWritable = await niiHandle.createWritable();
              await niiWritable.write(outBytes);
              await niiWritable.close();
            }
          }

          // Write each drawing layer as MRIcroGL .voi under drawings/
          if (drawingEntries.length > 0) {
            const drawingsDirHandle = await projDirHandle.getDirectoryHandle("drawings", { create: true });
            for (const { layer, entry } of drawingEntries) {
              const resp = await fetch("/api/drawing-save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  voxelsBase64: uint8ArrayToBase64(layer.voxels),
                  dims: layer.dims,
                  affine: layer.affine,
                  format: "voi",
                  fileName: entry.fileName,
                  mode: "download",
                }),
              });
              const data = await resp.json();
              if (resp.ok && data.fileBase64) {
                const voiBytes = base64ToUint8Array(data.fileBase64);
                const voiHandle = await drawingsDirHandle.getFileHandle(entry.fileName, { create: true });
                const voiWritable = await voiHandle.createWritable();
                await voiWritable.write(voiBytes);
                await voiWritable.close();
              }
            }
          }

          setCoregStatus("Ready", false);
          showToast(
            `Saved project "${projDirHandle.name}" (${volumeEntries.length} NIfTIs, ${drawingEntries.length} .voi layers, ${transformFiles.length} .mat files)!`,
            4500
          );
          return true;
        } catch (fsWriteErr) {
          console.warn(
            "Browser File System Access API write failed (e.g. .crswap restriction on mounted network share); falling back to /api/save-project:",
            fsWriteErr
          );
        }
      }
    }

    // 2. Fallback for Safari / Firefox: Native macOS Finder Folder Picker or Path via server.py (/api/save-project)
    const volumesPayload = volumeEntries.map(({ meta, entry }) => ({
      fileName: entry.fileName,
      currentAffine: entry.currentAffine,
      rawBase64: meta?.rawBytes ? uint8ArrayToBase64(meta.rawBytes) : null,
    }));
    const drawingsPayload = drawingEntries.map(({ layer, entry }) => ({
      fileName: entry.fileName,
      dims: layer.dims,
      affine: layer.affine,
      voxelsBase64: uint8ArrayToBase64(layer.voxels),
    }));

    let resp = await fetch("/api/save-project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patientId,
        manifest,
        volumes: volumesPayload,
        drawingFiles: drawingsPayload,
        transformFiles,
      }),
    });
    let res = await resp.json();

    if (res.cancelled) {
      setCoregStatus("Ready", false);
      return false;
    }

    if (res.needsPath) {
      const userDir = window.prompt(
        "Enter destination folder path on disk to save this NeuroCoreg project:",
        res.defaultDir
      );
      if (!userDir) {
        setCoregStatus("Ready", false);
        return false;
      }
      resp = await fetch("/api/save-project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targetDir: userDir.trim(),
          patientId,
          manifest,
          volumes: volumesPayload,
          drawingFiles: drawingsPayload,
          transformFiles,
        }),
      });
      res = await resp.json();
    }

    if (!resp.ok || !res.saved) {
      throw new Error(res.error || "Failed to save project");
    }

    setCoregStatus("Ready", false);
    showToast(`Saved project to ${res.projectDir} (${res.niftiFiles.length} NIfTIs + project.json)!`, 5000);
    return true;
  } catch (err) {
    console.error("Save project error:", err);
    setCoregStatus("Ready", false);
    showToast(`Save Project Error: ${err.message}`, 4500);
    return false;
  }
}

// Close App Modal & Backend Server Shutdown
function openCloseAppModal() {
  const modal = document.getElementById("closeAppModal");
  if (modal) {
    modal.hidden = false;
    modal.removeAttribute("hidden");
    modal.classList.remove("hidden");
  }
}

function closeCloseAppModal() {
  const modal = document.getElementById("closeAppModal");
  if (modal) {
    modal.hidden = true;
    modal.setAttribute("hidden", "");
    modal.classList.add("hidden");
  }
}

async function shutdownServerAndCloseTab() {
  try {
    setCoregStatus("Shutting Down...", true);
    showToast("Shutting down NeuroCoreg Studio server...", 0);
    await fetch("/api/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  } catch (_) {
    // Ignore network disconnect if server closed immediately
  }

  try {
    window.open("", "_self");
    window.close();
  } catch (_) {
    // Browser security may block closing tabs not opened via script
  }

  setTimeout(() => {
    document.body.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100vh;width:100vw;background:#070a10;color:#f1f5f9;font-family:'Inter',-apple-system,sans-serif;padding:24px;box-sizing:border-box;">
        <div style="max-width:440px;width:100%;background:rgba(15,23,42,0.95);border:1px solid rgba(148,163,184,0.25);border-radius:14px;padding:28px 24px;text-align:center;box-shadow:0 24px 60px rgba(0,0,0,0.75);">
          <div style="width:46px;height:46px;margin:0 auto 14px;border-radius:12px;background:rgba(16,185,129,0.16);border:1px solid rgba(16,185,129,0.4);color:#34d399;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;">✓</div>
          <h2 style="margin:0 0 8px;font-size:18px;font-weight:700;color:#f8fafc;">NeuroCoreg Studio Closed</h2>
          <p style="margin:0;font-size:13px;line-height:1.55;color:#94a3b8;">The backend server has been shut down cleanly.<br/>You may now close this browser tab.</p>
        </div>
      </div>
    `;
  }, 150);
}

// Restore a saved project workspace given its parsed project.json manifest and a map of fileName -> Uint8Array
async function restoreProjectWorkspace(manifest, niftiBytesByName, drawingBytesByName = new Map()) {
  setCoregStatus("Loading Project...", true);
  createNewPatientWorkspace(false, manifest.patientId || "sub-01");

  for (const vEntry of manifest.volumes || []) {
    const rawBytes =
      niftiBytesByName.get(vEntry.fileName) ||
      niftiBytesByName.get(vEntry.name) ||
      null;
    if (!rawBytes) continue;

    const vol = await loadNVImageFromBytes(rawBytes, {
      name: vEntry.name || vEntry.fileName,
      colormap: vEntry.colormap || "gray",
      opacity: vEntry.visible === false ? 0.0 : vEntry.opacity ?? 0.85,
      cal_min: vEntry.cal_min,
      cal_max: vEntry.cal_max,
    });

    state.nv.addVolume(vol);
    const vIsAtlas = Boolean(vEntry.isAtlas || vEntry.isDktAtlas);
    const vAtlasId = vEntry.atlasId || vEntry.atlasType || null;
    const vIsFS =
      vEntry.isFreeSurferAtlas !== null && vEntry.isFreeSurferAtlas !== undefined
        ? Boolean(vEntry.isFreeSurferAtlas)
        : vAtlasId === "dkt" || vAtlasId === "destrieux";
    registerVolumeMeta(vol, {
      label: vEntry.label || vEntry.name,
      modality: vEntry.modality || "Neuro Volume",
      opacity: vEntry.savedOpacity ?? vEntry.opacity ?? 0.85,
      userAtlasOpacity: vEntry.userAtlasOpacity ?? vEntry.opacity ?? vEntry.savedOpacity ?? 0.48,
      visible: vEntry.visible !== false,
      role: vEntry.role,
      isChildOfMov: Boolean(vEntry.isChildOfMov),
      isMask: Boolean(vEntry.isMask),
      isDktAtlas: vIsAtlas,
      atlasType: vEntry.atlasType || vAtlasId,
      atlasId: vAtlasId,
      atlasName: vEntry.atlasName || vEntry.label,
      atlasShortName: vEntry.atlasShortName || null,
      atlasCategory: vEntry.atlasCategory || null,
      atlasBadge: vEntry.atlasBadge || null,
      isFreeSurferAtlas: vIsFS,
      lut: vEntry.customLut || null,
      visibleRegionIds: Array.isArray(vEntry.visibleRegionIds) ? new Set(vEntry.visibleRegionIds.map(Number)) : null,
      atlasEngine: vEntry.atlasEngine || null,
      atlasSpace: vEntry.atlasSpace || null,
      rawBytes,
    });

    const meta = state.layerMeta.get(vol.id);
    if (meta) {
      if (vEntry.origAffine) meta.origAffine = cloneMat4(vEntry.origAffine);
      if (vEntry.params) meta.params = { ...vEntry.params };
      if (vEntry.isCheckedForChildApply !== undefined) meta.isCheckedForChildApply = Boolean(vEntry.isCheckedForChildApply);
      if (vEntry.isCheckedForExport !== undefined) meta.isCheckedForExport = Boolean(vEntry.isCheckedForExport);
      if (vIsAtlas && vol.img) {
        meta.rawAtlasVoxels = new Int16Array(vol.img.length);
        for (let i = 0; i < vol.img.length; i++) {
          meta.rawAtlasVoxels[i] = Math.round(vol.img[i]);
        }
        if (!meta.lut && vAtlasId && !vIsFS) {
          meta.lut = await ensureAtlasLutLoaded(vAtlasId);
        }
      }
    }

    const volIdx = getVolumeIndexById(vol.id);
    if (volIdx >= 0 && vEntry.currentAffine) {
      setVolumeAffineMatrix(volIdx, vEntry.currentAffine, true);
    }

    if (vIsAtlas || vEntry.colormap === "FreeSurferColorLUT") {
      await applyAtlasColormapToVolume(vol);
      if (vIsAtlas && vEntry.visible !== false) {
        state.activeDktVolId = vol.id;
        state.activeAtlasVolId = vol.id;
      }
    }

    if (vEntry.role === "reference") state.refVolId = vol.id;
    if (vEntry.role === "moving") state.movVolId = vol.id;
  }

  state.atlasOverlayStyle = manifest.atlasOverlayStyle || "filled";
  const restoredPrimaryAtlas = getDktVolume();
  if (restoredPrimaryAtlas) {
    const pm = state.layerMeta.get(restoredPrimaryAtlas.id);
    if (pm?.isFreeSurferAtlas && findActiveBrainMaskVolume()) {
      await applyBrainMaskToDktVolume(restoredPrimaryAtlas, findActiveBrainMaskVolume());
    }
    activateAtlasOverlay(restoredPrimaryAtlas.id, pm?.visible !== false);
  }

  state.savedTransforms = Array.isArray(manifest.savedTransforms) ? manifest.savedTransforms : [];
  state.baselineMetrics = manifest.baselineMetrics || null;

  if (manifest.viewState) {
    if (manifest.viewState.hudPinned !== undefined) {
      state.hudPinned = Boolean(manifest.viewState.hudPinned);
      document.getElementById("btnPinHud")?.classList.toggle("active", state.hudPinned);
    }
    if (manifest.viewState.coregPanelOpen !== undefined) {
      toggleCoregPanel(Boolean(manifest.viewState.coregPanelOpen));
    }
    if (Array.isArray(manifest.viewState.pan2Dxyzmm) && manifest.viewState.pan2Dxyzmm.length === 4) {
      applyPan2Dxyzmm(manifest.viewState.pan2Dxyzmm);
    } else {
      reset2DZoom();
    }
  } else {
    reset2DZoom();
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  if (typeof rehydrateRestoredDrawingLayers === "function" && Array.isArray(manifest.drawingLayers) && manifest.drawingLayers.length > 0) {
    await rehydrateRestoredDrawingLayers(manifest.drawingLayers, drawingBytesByName);
  }
  renderSavedTransformsDropdown();
  renderLayerList();
  computeAndDisplayAlignmentMetrics(true);
  setCoregStatus("Ready", false);
  showToast(
    `Opened project for Patient "${manifest.patientId || "sub-01"}" (${state.nv.volumes.length} layers restored)!`,
    4200
  );
}

// Open a saved project folder from local disk or a mounted network share
async function openSavedProject() {
  try {
    if (typeof window.showDirectoryPicker === "function") {
      let dirHandle = null;
      try {
        dirHandle = await window.showDirectoryPicker({ mode: "read" });
      } catch (pickerErr) {
        if (pickerErr && pickerErr.name === "AbortError") return;
        console.info("showDirectoryPicker unavailable in this context, falling back to server open-project:", pickerErr);
      }

      if (dirHandle) {
        try {
          let projJsonHandle;
          try {
            projJsonHandle = await dirHandle.getFileHandle("project.json");
          } catch (_) {
            showToast("No project.json found in the selected folder. Please select a saved NeuroCoreg project folder.", 4500);
            return;
          }

          const projFile = await projJsonHandle.getFile();
          const manifest = JSON.parse(await projFile.text());
          const niftiBytesByName = new Map();

          let niftiDirHandle = dirHandle;
          try {
            niftiDirHandle = await dirHandle.getDirectoryHandle("nifti");
          } catch (_) {
            // Fallback to root dirHandle
          }

          for (const vEntry of manifest.volumes || []) {
            const fname = vEntry.fileName || vEntry.name;
            try {
              const fh = await niftiDirHandle.getFileHandle(fname);
              const f = await fh.getFile();
              niftiBytesByName.set(fname, new Uint8Array(await f.arrayBuffer()));
            } catch (_) {
              console.warn("Could not read volume from project folder:", fname);
            }
          }

          const drawingBytesByName = new Map();
          if (Array.isArray(manifest.drawingLayers) && manifest.drawingLayers.length > 0) {
            let drawingsDirHandle = dirHandle;
            try {
              drawingsDirHandle = await dirHandle.getDirectoryHandle("drawings");
            } catch (_) {
              // Fallback to root dirHandle
            }
            for (const dEntry of manifest.drawingLayers) {
              const dname = dEntry.fileName;
              if (!dname) continue;
              try {
                const fh = await drawingsDirHandle.getFileHandle(dname);
                const f = await fh.getFile();
                drawingBytesByName.set(dname, new Uint8Array(await f.arrayBuffer()));
              } catch (_) {
                console.warn("Could not read drawing layer from project folder:", dname);
              }
            }
          }

          await restoreProjectWorkspace(manifest, niftiBytesByName, drawingBytesByName);
          return;
        } catch (fsReadErr) {
          console.warn("showDirectoryPicker read failed on selected folder, falling back to /api/open-project:", fsReadErr);
        }
      }
    }

    // 2. Native OS Folder Picker or Path Prompt via server.py (/api/open-project) — supports mounted network folders (/Volumes, /mnt, Z:\, \\server\share)
    setCoregStatus("Opening Project...", true);
    let resp = await fetch("/api/open-project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    let res = await resp.json();

    if (res.cancelled) {
      setCoregStatus("Ready", false);
      return;
    }

    if (res.needsPath) {
      const userDir = window.prompt(
        "Enter path to saved NeuroCoreg project folder (local or mounted network folder):",
        res.defaultDir
      );
      if (!userDir) {
        setCoregStatus("Ready", false);
        return;
      }
      resp = await fetch("/api/open-project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetDir: userDir.trim() }),
      });
      res = await resp.json();
    }

    if (resp.ok && res.manifest && Array.isArray(res.volumes)) {
      const niftiBytesByName = new Map();
      for (const v of res.volumes) {
        if (v.fileName && v.niftiBase64) {
          niftiBytesByName.set(v.fileName, base64ToUint8Array(v.niftiBase64));
        }
      }
      const drawingBytesByName = new Map();
      for (const d of res.drawings || []) {
        if (d.fileName && d.voiBase64) {
          drawingBytesByName.set(d.fileName, base64ToUint8Array(d.voiBase64));
        }
      }
      await restoreProjectWorkspace(res.manifest, niftiBytesByName, drawingBytesByName);
      return;
    }

    // 3. Final fallback: browser folder input (webkitdirectory)
    setCoregStatus("Ready", false);
    document.getElementById("openProjectFolderInput")?.click();
  } catch (err) {
    console.error("Open project error:", err);
    setCoregStatus("Ready", false);
    showToast(`Open Project Error: ${err.message}`, 4500);
  }
}

// Load a saved project from a browser FileList (webkitdirectory or drag-and-drop)
async function loadProjectFromFileList(files) {
  try {
    const projFile = files.find((f) => f.name === "project.json" || f.name.endsWith(".neurocoreg.json"));
    if (!projFile) {
      showToast("No project.json found in the selected folder.");
      return;
    }
    const manifest = JSON.parse(await projFile.text());
    const niftiBytesByName = new Map();
    const drawingBytesByName = new Map();

    for (const f of files) {
      if (f.name.toLowerCase().endsWith(".voi")) {
        drawingBytesByName.set(f.name, new Uint8Array(await f.arrayBuffer()));
      } else if (f.name.endsWith(".nii.gz") || f.name.endsWith(".nii")) {
        niftiBytesByName.set(f.name, new Uint8Array(await f.arrayBuffer()));
      }
    }

    await restoreProjectWorkspace(manifest, niftiBytesByName, drawingBytesByName);
  } catch (err) {
    console.error("Failed to load project from folder:", err);
    setCoregStatus("Ready", false);
    showToast(`Open Project Error: ${err.message}`, 4500);
  }
}

// ============================================================================
// FSL BET Skull Stripping, FastSurfer Availability Check & Interactive DKT Atlas Hover
// ============================================================================
async function checkNeuroToolsStatus(showToastMsg = false) {
  const betBadge = document.getElementById("betStatusBadge");
  const fsBadge = document.getElementById("fastsurferStatusBadge");
  const fsNote = document.getElementById("fastsurferDetailNote");

  if (fsBadge) {
    fsBadge.textContent = "Checking...";
    fsBadge.className = "tool-mini-badge checking";
  }

  try {
    const customImg = String(state.atlasStudioState?.hipsCustomImage || "").trim();
    const qSuffix = customImg ? `?hipsImage=${encodeURIComponent(customImg)}` : "";
    const resp = await fetch(`/api/neuro-tools-status${qSuffix}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Status check failed");

    state.neuroToolsStatus = data;
    if (data.dktLut && Object.keys(data.dktLut).length > 0) {
      state.dktLut = data.dktLut;
      state.atlasLuts.set("dkt", data.dktLut);
      state.atlasLuts.set("destrieux", data.dktLut);
    }

    if (Array.isArray(data.atlases) && data.atlases.length > 0) {
      state.atlasRegistry = data.atlases;
      populateAtlasTypeSelect(state.atlasRegistry);
    } else {
      try {
        const regResp = await fetch(`/api/atlases${qSuffix}`);
        if (regResp.ok) {
          const regData = await regResp.json();
          if (Array.isArray(regData.atlases)) {
            state.atlasRegistry = regData.atlases;
            populateAtlasTypeSelect(state.atlasRegistry);
          }
          if (regData.hipsThomas) state.neuroToolsStatus.hipsThomas = regData.hipsThomas;
        }
      } catch (_) {
        // fallback gracefully
      }
    }

    if (betBadge) {
      const betOk = Boolean(data.fslBet?.available);
      betBadge.textContent = betOk ? "FSL BET Ready" : "BET Fallback (SciPy)";
      betBadge.className = `tool-mini-badge ${betOk ? "ready" : "fallback"}`;
    }

    if (fsBadge) {
      const fs = data.fastsurfer || {};
      const fnirtOk = Boolean(data.fnirt?.available);
      const hipsOk = Boolean(data.hipsThomas?.available);
      const atlasCnt = data.atlasCount || state.atlasRegistry.length || 7;
      if (fs.available) {
        const extras = [fnirtOk ? "FNIRT" : "", hipsOk ? "HIPS" : ""].filter(Boolean);
        fsBadge.textContent = extras.length ? `FS + ${extras.join("/")} · ${atlasCnt} Atlases` : "FastSurfer Ready";
        fsBadge.title = `${fs.statusLabel || fs.fastsurferHome || "FastSurfer Ready"}${fnirtOk ? " + FSL FNIRT Nonlinear" : ""}${hipsOk ? ` + ${data.hipsThomas.statusLabel || "HIPS-THOMAS"}` : ""}`;
        fsBadge.className = "tool-mini-badge ready";
      } else if (fs.fallbackAvailable || fnirtOk || hipsOk) {
        const engTag = hipsOk && fnirtOk ? "HIPS + FNIRT" : hipsOk ? "HIPS-THOMAS" : fnirtOk ? "FNIRT" : "";
        fsBadge.textContent = engTag ? `${engTag} · ${atlasCnt} Atlases` : `${atlasCnt} Atlases Ready`;
        fsBadge.title = data.hipsThomas?.statusLabel || `${atlasCnt} Atlases Ready`;
        fsBadge.className = "tool-mini-badge fallback";
      } else {
        fsBadge.textContent = "FastSurfer Not Found";
        fsBadge.className = "tool-mini-badge checking";
      }
    }

    syncAtlasTypeSelectionUI(false, false);
    syncAtlasStudioUI();

    const serverMissingAtlasJob = Boolean(data && !data.capabilities?.atlasJob);
    if (serverMissingAtlasJob) {
      console.warn(
        "NeuroCoreg backend is running an older server.py without /api/atlas-job — restart server.py for async HIPS-THOMAS progress & Cancel."
      );
      if (showToastMsg || data.hipsThomas?.available) {
        showToast(
          "Restart server.py to enable live HIPS-THOMAS background progress, Cancel, & default Docker priority.",
          5500
        );
      }
    } else if (showToastMsg) {
      const fs = data.fastsurfer || {};
      const fnirtNote = data.fnirt?.available ? " + FSL FNIRT" : "";
      const hipsNote = data.hipsThomas?.available ? " + HIPS-THOMAS Docker" : "";
      const msg = fs.available
        ? `Brain Atlases ready (${state.atlasRegistry.length || 7} atlases): FastSurfer${fnirtNote}${hipsNote} + FSL BET!`
        : `Brain Atlases ready (${state.atlasRegistry.length || 7} atlases): Template Registration${fnirtNote}${hipsNote} + FSL BET!`;
      showToast(msg, 4500);
    }
  } catch (err) {
    console.warn("Neuro tools status check fallback:", err);
    // Fallback: try loading dkt_lut.json directly from static sample_data
    try {
      const lutResp = await fetch("/sample_data/dkt_lut.json");
      if (lutResp.ok) {
        state.dktLut = await lutResp.json();
      }
    } catch (_) {
      // ignore
    }
  }
}

// Toggle a collapsible neuro module section (BET or FastSurfer DKT) independently between minimized and expanded
function toggleNeuroModuleSection(sectionId, headerId, bodyId, btnId, forceExpand = null) {
  const sectionEl = document.getElementById(sectionId);
  const headerEl = document.getElementById(headerId);
  const bodyEl = document.getElementById(bodyId);
  const btnEl = document.getElementById(btnId);
  if (!sectionEl) return;

  const currentlyMinimized = sectionEl.classList.contains("is-minimized");
  const shouldExpand = forceExpand !== null ? Boolean(forceExpand) : currentlyMinimized;

  sectionEl.classList.toggle("is-minimized", !shouldExpand);
  if (headerEl) {
    headerEl.setAttribute("aria-expanded", String(shouldExpand));
  }
  if (bodyEl) {
    bodyEl.hidden = !shouldExpand;
  }
  if (btnEl) {
    btnEl.textContent = shouldExpand ? "−" : "+";
  }
}

// Run FSL BET skull stripping on the selected T1w structural volume and automatically add the brain MRI to workspace
async function runFslBetSkullStrip() {
  if (!state.nv) return;
  const t1Select = document.getElementById("selectBetT1Vol") || document.getElementById("selectNeuroT1Vol");
  const targetVolId = t1Select?.value || state.refVolId;
  const volIdx = getVolumeIndexById(targetVolId);
  if (volIdx < 0) {
    showToast("Select a T1w structural volume first.");
    return;
  }

  const srcVol = state.nv.volumes[volIdx];
  const srcMeta = state.layerMeta.get(srcVol.id);
  if (!srcMeta?.rawBytes) {
    showToast("Selected volume does not have raw NIfTI bytes available for FSL BET.");
    return;
  }

  const fracInput = document.getElementById("betFracInput");
  const frac = Math.max(0.05, Math.min(0.95, parseFloat(fracInput?.value || "0.5") || 0.5));
  const btn = document.getElementById("btnRunBet");
  const origBtnText = btn ? btn.textContent : "Run FSL BET Skull Strip";

  try {
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Running FSL BET...";
    }
    setCoregStatus("Running BET...", true);
    showToast(`Running FSL BET skull stripping (-f ${frac.toFixed(2)}) on ${srcMeta.label}...`, 0);

    const resp = await fetch("/api/skull-strip-bet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sourceName: srcVol.name,
        sourceBase64: uint8ArrayToBase64(srcMeta.rawBytes),
        frac,
        currentAffine: srcVol.hdr.affine,
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.niftiBase64) {
      throw new Error(data.error || "FSL BET skull stripping failed");
    }

    const brainBytes = base64ToUint8Array(data.niftiBase64);
    const brainVol = await loadNVImageFromBytes(brainBytes, {
      name: data.fileName,
      colormap: "gray",
      opacity: 1.0,
    });

    // If a previous BET brain with the same filename exists, remove it so re-running with a new -f replaces cleanly
    const existingIdx = state.nv.volumes.findIndex((v) => v.name === data.fileName);
    if (existingIdx >= 0) {
      const oldId = state.nv.volumes[existingIdx].id;
      state.nv.removeVolumeByIndex(existingIdx);
      state.layerMeta.delete(oldId);
    }

    state.nv.addVolume(brainVol);
    const cleanBaseLabel = (srcMeta.label || srcVol.name).replace(/\s*\(Reference\)/i, "");
    registerVolumeMeta(brainVol, {
      label: `${cleanBaseLabel} — Brain (BET -f=${frac.toFixed(2)})`,
      modality: "T1w Brain (FSL BET)",
      opacity: 1.0,
      visible: true,
      isBetBrain: true,
      rawBytes: brainBytes,
    });

    // Place the newly created brain-extracted image immediately under the Reference image in the Image Stack
    const refIdx = getVolumeIndexById(state.refVolId);
    const targetIdx = refIdx >= 0 ? Math.min(refIdx + 1, state.nv.volumes.length - 1) : Math.min(1, state.nv.volumes.length - 1);
    const currentIdx = state.nv.volumes.indexOf(brainVol);
    if (currentIdx >= 0 && currentIdx !== targetIdx) {
      if (typeof state.nv.setVolume === "function") {
        try {
          state.nv.setVolume(brainVol, targetIdx);
        } catch (_) {
          // Fallback to direct array reorder below
        }
      }
      const afterIdx = state.nv.volumes.indexOf(brainVol);
      if (afterIdx >= 0 && afterIdx !== targetIdx) {
        state.nv.volumes.splice(afterIdx, 1);
        state.nv.volumes.splice(targetIdx, 0, brainVol);
      }
    }

    // If a DKT Atlas volume is loaded in the workspace, automatically update its brain mask to this new BET skull-strip mask
    const activeDkt = getDktVolume();
    if (activeDkt) {
      await applyBrainMaskToDktVolume(activeDkt, brainVol);
    }

    state.nv.updateGLVolume();
    state.nv.drawScene();
    renderLayerList();
    setCoregStatus("Ready", false);
    showToast(
      `Added skull-stripped T1w "${data.fileName}" under Reference image (${data.brainVolumeCm3} cm³, ${data.brainVoxels.toLocaleString()} brain voxels)${activeDkt ? " & updated DKT brain mask" : ""}!`,
      4800
    );
  } catch (err) {
    console.error("FSL BET error:", err);
    setCoregStatus("Ready", false);
    showToast(`FSL BET Error: ${err.message}`, 4500);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = origBtnText;
    }
  }
}

function getLoadedAtlasVolumes() {
  if (!state.nv || !Array.isArray(state.nv.volumes)) return [];
  return state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return Boolean(m && (m.isAtlas || m.isDktAtlas));
  });
}

function getAtlasBadgeInfo(meta) {
  const aid = String(meta?.atlasId || meta?.atlasType || "").toLowerCase();
  const badgeRaw = meta?.atlasBadge;
  if (aid === "thomas" || badgeRaw === "THOMAS") return { text: "THOMAS", cls: "atlas-thomas" };
  if (aid.startsWith("tian")) {
    const sc = aid.replace("tian_", "").toUpperCase();
    return { text: `TIAN ${sc}`, cls: "atlas-tian" };
  }
  if (aid.startsWith("fsl") || badgeRaw === "FSL") {
    return { text: meta?.atlasShortName ? `FSL:${meta.atlasShortName}` : "FSL", cls: "atlas-fsl" };
  }
  if (meta?.atlasCategory === "custom" || badgeRaw === "CUSTOM" || aid.startsWith("custom_")) {
    return { text: "CUSTOM", cls: "atlas-custom" };
  }
  if (aid === "destrieux" || badgeRaw === "DESTRIEUX") {
    return { text: "DESTRIEUX", cls: "atlas-destrieux" };
  }
  return { text: badgeRaw || "DKT", cls: "" };
}

function generateDeterministicRgb(labelId, hemi = "C") {
  const idx = Math.max(1, Math.abs(Math.round(labelId)));
  const hue = (idx * 0.618033988749895 + (hemi === "R" ? 0.04 : 0.0)) % 1.0;
  const sat = 0.74;
  const val = hemi === "R" ? 0.94 : 0.88;
  const h6 = hue * 6;
  const sector = Math.floor(h6) % 6;
  const f = h6 - Math.floor(h6);
  const p = val * (1 - sat);
  const q = val * (1 - f * sat);
  const t = val * (1 - (1 - f) * sat);
  const table = [
    [val, t, p],
    [q, val, p],
    [p, val, t],
    [p, q, val],
    [t, p, val],
    [val, p, q],
  ];
  const [r, g, b] = table[sector];
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function getAtlasRegistryEntry(atlasId) {
  const aid = String(atlasId || "dkt").toLowerCase();
  return (state.atlasRegistry || []).find((a) => a.id === aid) || null;
}

function populateAtlasTypeSelect(registry = []) {
  const selectEl = document.getElementById("selectAtlasType");
  if (!selectEl || !Array.isArray(registry) || registry.length === 0) return;
  const prevVal = selectEl.value || state.selectedAtlasType || "dkt";
  const groupLabels = [
    { key: "cortical", label: "Cortical & Whole-Brain" },
    { key: "thalamic", label: "Thalamic Nuclei" },
    { key: "subcortical", label: "Subcortical Parcellations (Tian)" },
    { key: "fsl", label: "Local FSL Atlases (Auto-Discovered)" },
  ];
  selectEl.innerHTML = "";
  for (const grp of groupLabels) {
    const items = registry.filter((a) => a.category === grp.key);
    if (!items.length) continue;
    const og = document.createElement("optgroup");
    og.label = grp.label;
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = item.available ? item.name : `${item.name} [Missing Assets]`;
      opt.disabled = !item.available;
      og.appendChild(opt);
    }
    selectEl.appendChild(og);
  }
  const hasPrev = Array.from(selectEl.options).some((o) => o.value === prevVal && !o.disabled);
  selectEl.value = hasPrev ? prevVal : "dkt";
}

function getAtlasPipelineModeOptions(atlasId) {
  const aid = String(atlasId || "dkt").toLowerCase();
  const isFS = aid === "dkt" || aid === "destrieux";
  const fnirtOk = Boolean(state.neuroToolsStatus?.fnirt?.available ?? true);
  const hipsInfo = state.neuroToolsStatus?.hipsThomas || {};
  const hipsOk = Boolean(hipsInfo.available);

  if (isFS) {
    if (aid === "destrieux") {
      return [
        { v: "fastsurfer_vinn", l: "FastSurfer Surface + --fsaparc (aparc.a2009s, Slower)" },
        { v: "template_fnirt", l: fnirtOk ? "MNI152 Nonlinear FNIRT (Cached)" : "MNI152 Nonlinear (Fallback FLIRT)" },
        { v: "template_flirt", l: "Quick Template Destrieux (FSL FLIRT ~2s)" },
      ];
    }
    return [
      { v: "fastsurfer_vinn", l: "FastSurferVINN DKT-Only (Fast, Recommended)" },
      { v: "fastsurfer_full", l: "Full FastSurfer (--seg_only + Cereb/HypVINN)" },
      { v: "template_fnirt", l: fnirtOk ? "MNI152 Nonlinear FNIRT (Cached)" : "MNI152 Nonlinear (Fallback FLIRT)" },
      { v: "template_flirt", l: "Quick Template DKT (FSL FLIRT ~2s)" },
    ];
  }

  if (aid === "thomas") {
    if (hipsOk) {
      const containerTag = hipsInfo.mode === "native" ? "CLI" : "Docker";
      return [
        { v: "hips_thomas_docker", l: `HIPS-THOMAS ${containerTag} (Max Accuracy — Default)` },
        {
          v: "fnirt_nonlinear",
          l: fnirtOk ? "Backup: Nonlinear FNIRT (MNI152 -> Subject, Cached)" : "Backup: Template Registration (FLIRT/SciPy)",
        },
        { v: "flirt_12dof", l: "Backup: Fast Linear FLIRT (12-DOF ~2s)" },
      ];
    }
    const hipsDisabledLabel =
      hipsInfo.dockerInstalled && !hipsInfo.dockerRunning
        ? "HIPS-THOMAS Docker (Start Docker & click Check)"
        : "HIPS-THOMAS Docker (Not Installed — pull anagrammarian/sthomas)";
    return [
      {
        v: "fnirt_nonlinear",
        l: fnirtOk ? "Nonlinear FNIRT (MNI152 -> Subject, Backup Default)" : "Template Registration (FLIRT/SciPy Backup)",
      },
      { v: "flirt_12dof", l: "Fast Linear FLIRT (12-DOF ~2s, Cached)" },
      { v: "hips_thomas_docker", l: hipsDisabledLabel, disabled: true },
    ];
  }

  return [
    {
      v: "fnirt_nonlinear",
      l: fnirtOk ? "Nonlinear FNIRT (MNI152 -> Subject, Recommended)" : "Template Registration (FLIRT/SciPy Fallback)",
    },
    { v: "flirt_12dof", l: "Fast Linear FLIRT (12-DOF ~2s, Cached)" },
  ];
}

function resolvePreferredAtlasPipelineMode(atlasId, opts) {
  const aid = String(atlasId || "dkt").toLowerCase();
  const isEnabled = (val) => Boolean(val && opts.some((o) => o.v === val && !o.disabled));
  const savedPref = state.atlasPreferredModeById?.get(aid);
  if (isEnabled(savedPref)) return savedPref;
  const firstEnabled = opts.find((o) => !o.disabled);
  return firstEnabled ? firstEnabled.v : (opts[0]?.v || "fnirt_nonlinear");
}

function populateAtlasPipelineModes(atlasId) {
  const modeSelect = document.getElementById("selectDktMode");
  if (!modeSelect) return;
  const aid = String(atlasId || "dkt").toLowerCase();
  const opts = getAtlasPipelineModeOptions(aid);

  modeSelect.innerHTML = "";
  for (const o of opts) {
    const el = document.createElement("option");
    el.value = o.v;
    el.textContent = o.l;
    if (o.disabled) el.disabled = true;
    modeSelect.appendChild(el);
  }
  modeSelect.value = resolvePreferredAtlasPipelineMode(aid, opts);
  state.lastPopulatedAtlasModeId = aid;
}

// Helper: Resolve canonical atlas ID/type for any loaded atlas volume
function resolveAtlasTypeForVolume(vol = null, explicitType = null) {
  if (explicitType) {
    const t = String(explicitType).toLowerCase();
    if (t.includes("destrieux") || t.includes("a2009s")) return "destrieux";
    if (t.includes("thomas")) return "thomas";
    const tm = t.match(/tian[_-]?s?([1-4])/);
    if (tm) return `tian_s${tm[1]}`;
    if (t === "dkt" || t.startsWith("fsl_") || t.startsWith("custom_")) return t;
    return t;
  }
  if (vol) {
    const meta = state.layerMeta.get(vol.id);
    if (meta?.atlasId) return meta.atlasId;
    if (meta?.atlasType) return meta.atlasType;
    const inf = inferModalityAndStyle(vol.name || meta?.label || "");
    if (inf.atlasId) return inf.atlasId;
    const srcArr = meta?.rawAtlasVoxels || meta?.unmaskedDktData || vol.img;
    if (srcArr && srcArr.length) {
      for (let i = 0; i < srcArr.length; i += 8) {
        if (srcArr[i] >= 11100) return "destrieux";
      }
    }
  }
  return document.getElementById("selectAtlasType")?.value || state.selectedAtlasType || "dkt";
}

function getAtlasDefaultMaxLabel(vol = null, explicitType = null) {
  if (vol) {
    const meta = state.layerMeta.get(vol.id);
    if (meta?.lut) {
      let mx = 1;
      for (const k of Object.keys(meta.lut)) {
        const n = parseInt(k, 10);
        if (Number.isFinite(n) && n > mx) mx = n;
      }
      if (mx > 1) return mx;
    }
  }
  const aid = resolveAtlasTypeForVolume(vol, explicitType);
  if (aid === "destrieux") return 12175;
  if (aid === "dkt") return 2035;
  if (aid === "thomas") return 118;
  return 255;
}

function getAtlasShortTitle(vol = null, explicitType = null) {
  if (vol) {
    const meta = state.layerMeta.get(vol.id);
    if (meta?.atlasShortName) return meta.atlasShortName;
  }
  const aid = resolveAtlasTypeForVolume(vol, explicitType);
  const reg = getAtlasRegistryEntry(aid);
  if (reg?.shortName) return reg.shortName;
  if (aid === "destrieux") return "Destrieux Atlas";
  if (aid === "thomas") return "THOMAS";
  if (aid.startsWith("tian_s")) return `Tian ${aid.replace("tian_", "").toUpperCase()}`;
  return "DKT Atlas";
}

// Synchronize the Brain Atlases Panel UI (Pipeline options, Slow Notice, Buttons, and Active Atlas Volume)
function syncAtlasTypeSelectionUI(showSlowNoticeToast = false, switchActiveVolume = false) {
  const atlasSelect = document.getElementById("selectAtlasType");
  const modeSelect = document.getElementById("selectDktMode");
  const warnBanner = document.getElementById("destrieuxSlowWarning");
  const runBtn = document.getElementById("btnRunDktAtlas");
  const loadLbl = document.getElementById("lblLoadDktBtn");
  const detailNote = document.getElementById("fastsurferDetailNote");

  const atlasId = atlasSelect?.value || state.selectedAtlasType || "dkt";
  const isDestrieux = atlasId === "destrieux";
  const isDkt = atlasId === "dkt";
  state.selectedAtlasType = atlasId;

  populateAtlasPipelineModes(atlasId);

  const modeVal = modeSelect?.value || "fastsurfer_vinn";
  const isSlowFsAparc = isDestrieux && !modeVal.startsWith("template_");

  if (warnBanner) {
    warnBanner.hidden = !isSlowFsAparc;
  }

  const licStatusEl = document.getElementById("fsLicenseStatusText");
  const licBtnLbl = document.getElementById("lblSelectFsLicenseBtn");
  const licInfo = state.neuroToolsStatus?.fsLicense || state.neuroToolsStatus?.fastsurfer?.fsLicense || null;
  if (licStatusEl) {
    if (licInfo?.available) {
      licStatusEl.textContent = `✓ FS License: ${licInfo.label || "license.txt"}`;
      licStatusEl.title = licInfo.path || "Valid FreeSurfer license detected";
      licStatusEl.className = "fs-license-status is-valid";
      if (licBtnLbl) licBtnLbl.textContent = "Change...";
    } else {
      licStatusEl.textContent = "⚠ FS License: Not Found";
      licStatusEl.title = "No valid FreeSurfer license file found";
      licStatusEl.className = "fs-license-status is-missing";
      if (licBtnLbl) licBtnLbl.textContent = "Select License...";
    }
  }

  const regEntry = getAtlasRegistryEntry(atlasId);
  const shortTitle = regEntry?.shortName || getAtlasShortTitle(null, atlasId);
  if (detailNote) {
    if (atlasId === "thomas") {
      const hipsInfo = state.neuroToolsStatus?.hipsThomas || {};
      if (hipsInfo.available) {
        const containerLbl =
          hipsInfo.dockerImage || (hipsInfo.cliPath ? hipsInfo.cliPath.split("/").pop() : "HIPS-THOMAS");
        detailNote.textContent = `THOMAS: Using HIPS-THOMAS (${containerLbl}) by default for maximal native subject accuracy (cached per T1w; FNIRT/FLIRT available as fast backups).`;
      } else {
        const hipsNote = hipsInfo.statusLabel || "HIPS-THOMAS Docker not found";
        detailNote.textContent = `THOMAS: ${hipsNote}. Using MNI152 FNIRT/FLIRT backup (pull anagrammarian/sthomas and click Check to enable HIPS-THOMAS).`;
      }
    } else if (regEntry?.description) {
      detailNote.textContent = `${regEntry.shortName}: ${regEntry.description}`;
    } else if (isDestrieux) {
      detailNote.textContent = "Destrieux (aparc.a2009s+aseg, 148 regions): runs FastSurfer --fsaparc or MNI152 FNIRT/FLIRT.";
    } else {
      detailNote.textContent = "DKT (aparc.DKTatlas+aseg, 104 regions): runs FastSurferVINN or MNI152 FNIRT/FLIRT.";
    }
  }

  if (runBtn && !state.activeAtlasJob && !runBtn.disabled) {
    runBtn.textContent = `Generate ${shortTitle}`;
  }
  if (loadLbl) {
    loadLbl.textContent = isDkt || isDestrieux ? `Load ${shortTitle}` : "Load";
  }

  syncHipsThomasOptionsUI();
  renderActiveAtlasJobUI();

  if (switchActiveVolume && state.nv && Array.isArray(state.nv.volumes)) {
    const matchingAtlasVol = state.nv.volumes.find((v) => {
      const m = state.layerMeta.get(v.id);
      return (m?.isAtlas || m?.isDktAtlas) && resolveAtlasTypeForVolume(v) === atlasId;
    });
    if (matchingAtlasVol) {
      activateAtlasOverlay(matchingAtlasVol.id, true);
    } else {
      syncDktOverlayControlsUI();
    }
  } else {
    syncDktOverlayControlsUI();
  }

  if (showSlowNoticeToast && isSlowFsAparc) {
    showToast(
      "Destrieux (aparc.a2009s) selected: FastSurfer will run with --fsaparc and without --seg_only (full surface reconstruction — processing is slower). Or pick MNI152 FNIRT/FLIRT.",
      5800
    );
  }
}

async function ensureAtlasLutLoaded(atlasId) {
  const aid = String(atlasId || "dkt").toLowerCase();
  if (aid === "dkt" || aid === "destrieux") {
    return await ensureDktLutLoaded();
  }
  if (state.atlasLuts.has(aid)) {
    const cached = state.atlasLuts.get(aid);
    if (cached && Object.keys(cached).length > 0) return cached;
  }
  try {
    const resp = await fetch(`/api/atlases?lut=${encodeURIComponent(aid)}`);
    if (resp.ok) {
      const data = await resp.json();
      if (data.lut && Object.keys(data.lut).length > 0) {
        state.atlasLuts.set(aid, data.lut);
        return data.lut;
      }
    }
  } catch (_) {
    // fallback to static bundled files if server unavailable
  }
  const staticMap = {
    thomas: "/sample_data/atlases/thomas_lut.json",
    tian_s1: "/sample_data/atlases/tian_s1_lut.json",
    tian_s2: "/sample_data/atlases/tian_s2_lut.json",
    tian_s3: "/sample_data/atlases/tian_s3_lut.json",
    tian_s4: "/sample_data/atlases/tian_s4_lut.json",
  };
  if (staticMap[aid]) {
    try {
      const r = await fetch(staticMap[aid]);
      if (r.ok) {
        const j = await r.json();
        const lutObj = {};
        for (const item of j.labels || []) {
          if (item && item.id > 0) {
            const rgb = [item.r ?? 128, item.g ?? 128, item.b ?? 128];
            lutObj[String(item.id)] = {
              id: item.id,
              name: item.name,
              shortName: item.shortName || item.name,
              title: item.name,
              hemisphere: item.hemisphere || "C",
              rgb,
              hex: `#${rgb.map((c) => Number(c).toString(16).padStart(2, "0")).join("")}`,
            };
          }
        }
        state.atlasLuts.set(aid, lutObj);
        return lutObj;
      }
    } catch (_) {
      // ignore
    }
  }
  return {};
}

// Validate & save a user-selected FreeSurfer license file via /api/set-fs-license
async function uploadAndSetFsLicenseFile(file) {
  if (!file) return false;
  try {
    const contentText = await file.text();
    const resp = await fetch("/api/set-fs-license", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fileName: file.name,
        contentText,
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.valid) {
      throw new Error(data.error || "Invalid FreeSurfer license file format");
    }

    if (!state.neuroToolsStatus) state.neuroToolsStatus = {};
    state.neuroToolsStatus.fsLicense = data.fsLicense;
    if (state.neuroToolsStatus.fastsurfer) {
      state.neuroToolsStatus.fastsurfer.fsLicense = data.fsLicense;
    }
    syncAtlasTypeSelectionUI(false, false);
    showToast(`Validated & saved FreeSurfer license "${file.name}"!`, 4200);

    if (state._pendingDestrieuxRunAfterLicense) {
      state._pendingDestrieuxRunAfterLicense = false;
      await runDktSegmentation();
    }
    return true;
  } catch (err) {
    state._pendingDestrieuxRunAfterLicense = false;
    console.error("FreeSurfer license validation error:", err);
    showToast(`Invalid FreeSurfer License: ${err.message}`, 5500);
    return false;
  }
}

function isTargetVolumeSkullStripped(vol) {
  if (!vol) return false;
  const meta = state.layerMeta.get(vol.id);
  if (meta?.isBetBrain || meta?.modality === "BRAIN") return true;
  const label = `${vol.name || ""} ${meta?.label || ""}`;
  return /(_brain|\bbet\b|skull[_-]?strip)/i.test(label);
}

function syncHipsThomasOptionsUI(explicitT1VolId = null) {
  const atlasId = String(document.getElementById("selectAtlasType")?.value || state.selectedAtlasType || "dkt").toLowerCase();
  const modeVal = String(document.getElementById("selectDktMode")?.value || "");
  const showHipsOpts = atlasId === "thomas" && modeVal === "hips_thomas_docker";

  const t1Id =
    explicitT1VolId ||
    document.getElementById("selectDktT1Vol")?.value ||
    state.refVolId;
  const t1Idx = getVolumeIndexById(t1Id);
  const t1Vol = t1Idx >= 0 ? state.nv?.volumes?.[t1Idx] : null;
  const isDm = isTargetVolumeSkullStripped(t1Vol);

  const sideBox = document.getElementById("hipsThomasOptionsRow");
  if (sideBox) sideBox.hidden = !showHipsOpts;
  const sideContrast = document.getElementById("selectHipsContrast");
  if (sideContrast) sideContrast.value = state.atlasStudioState?.hipsContrast || "t1";
  const sideForce = document.getElementById("chkHipsForceRecompute");
  if (sideForce) sideForce.checked = Boolean(state.atlasStudioState?.hipsForceRecompute);
  const sideDmBadge = document.getElementById("hipsAutoDmBadge");
  if (sideDmBadge) sideDmBadge.hidden = !(showHipsOpts && isDm);
}

function formatElapsedMMSS(totalSec) {
  const s = Math.max(0, Math.floor(Number(totalSec) || 0));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

function renderActiveAtlasJobUI() {
  const job = state.activeAtlasJob;
  const runBtn = document.getElementById("btnRunDktAtlas");
  const sideStrip = document.getElementById("hipsJobProgressStrip");
  const sideStage = document.getElementById("hipsJobStageText");
  const sideElapsed = document.getElementById("hipsJobElapsed");
  const sideLog = document.getElementById("hipsJobLogTail");
  const sideCancel = document.getElementById("btnCancelHipsJob");

  const popBody = state.atlasStudioPopout?.bodyEl;
  const studioGenBtn = popBody?.querySelector("#atlasStudioBtnGenerate");
  const studioStrip = popBody?.querySelector("#atlasStudioJobProgress");
  const studioStage = popBody?.querySelector("#atlasStudioJobStage");
  const studioElapsed = popBody?.querySelector("#atlasStudioJobElapsed");
  const studioLog = popBody?.querySelector("#atlasStudioJobLog");
  const studioCancel = popBody?.querySelector("#atlasStudioBtnCancelJob");

  if (!job) {
    if (sideStrip) sideStrip.hidden = true;
    if (studioStrip) studioStrip.hidden = true;
    if (runBtn) {
      runBtn.disabled = false;
      const regEntry = getAtlasRegistryEntry(state.selectedAtlasType);
      const shortTitle = regEntry?.shortName || getAtlasShortTitle(null, state.selectedAtlasType);
      runBtn.textContent = `Generate ${shortTitle}`;
    }
    if (studioGenBtn) {
      studioGenBtn.disabled = false;
      studioGenBtn.textContent = "Generate Atlas in Subject Space";
    }
    return;
  }

  const elapsedSec = job.elapsedSec !== undefined
    ? job.elapsedSec
    : (Date.now() - (job.startedAtMs || Date.now())) / 1000;
  const elapsedTxt = formatElapsedMMSS(elapsedSec);
  const stageTxt = job.stage || "Running HIPS-THOMAS...";
  const lastLog = Array.isArray(job.logTail) && job.logTail.length > 0
    ? job.logTail[job.logTail.length - 1]
    : "Executing hipsthomas.sh inside Docker...";

  if (sideStrip) sideStrip.hidden = false;
  if (sideStage) sideStage.textContent = stageTxt;
  if (sideElapsed) sideElapsed.textContent = elapsedTxt;
  if (sideLog) {
    sideLog.textContent = lastLog;
    sideLog.title = (job.logTail || []).join("\n");
  }
  if (sideCancel) {
    sideCancel.disabled = Boolean(job.cancelling);
    sideCancel.textContent = job.cancelling ? "Stopping..." : "Cancel";
  }

  if (studioStrip) studioStrip.hidden = false;
  if (studioStage) studioStage.textContent = stageTxt;
  if (studioElapsed) studioElapsed.textContent = elapsedTxt;
  if (studioLog) {
    studioLog.textContent = lastLog;
    studioLog.title = (job.logTail || []).join("\n");
  }
  if (studioCancel) {
    studioCancel.disabled = Boolean(job.cancelling);
    studioCancel.textContent = job.cancelling ? "Stopping..." : "Cancel";
  }

  if (runBtn) {
    runBtn.disabled = true;
    runBtn.textContent = job.cancelling ? "Cancelling HIPS-THOMAS..." : "Running HIPS-THOMAS...";
  }
  if (studioGenBtn) {
    studioGenBtn.disabled = true;
    studioGenBtn.textContent = job.cancelling ? "Cancelling HIPS-THOMAS..." : "Running HIPS-THOMAS...";
  }
}

function clearActiveAtlasJobTimers(job) {
  if (!job) return;
  if (job.timerId) clearInterval(job.timerId);
  if (job.elapsedTimerId) clearInterval(job.elapsedTimerId);
  job.timerId = null;
  job.elapsedTimerId = null;
}

async function applyAtlasGenerationResult(data, { atlasId, atlasShort, regEntry, modeLabel }) {
  const resolvedId = data.atlasId || data.atlasType || atlasId;
  const isFS =
    data.isFreeSurferAtlas !== undefined
      ? Boolean(data.isFreeSurferAtlas)
      : resolvedId === "dkt" || resolvedId === "destrieux";
  const lutObj = data.lut || data.dktLut || null;
  if (lutObj && Object.keys(lutObj).length > 0) {
    if (isFS) state.dktLut = lutObj;
    else state.atlasLuts.set(resolvedId, lutObj);
  }

  state.atlasStudioState.warpCacheHit = data.warpCacheHit !== undefined ? Boolean(data.warpCacheHit) : null;
  state.atlasStudioState.hipsCacheHit =
    data.hipsCacheHit === true ? true : data.regModeUsed === "hips_thomas_docker" ? false : null;

  const bytes = base64ToUint8Array(data.niftiBase64);
  const unmaskedBytes = data.unmaskedNiftiBase64 ? base64ToUint8Array(data.unmaskedNiftiBase64) : null;
  const vol = await addAtlasVolumeToWorkspace(bytes, {
    fileName: data.fileName,
    displayLabel: `${data.shortName || atlasShort} (${data.regionCount || 0} Regions)`,
    atlasId: resolvedId,
    atlasName: data.atlasName || atlasShort,
    atlasShortName: data.shortName || atlasShort,
    atlasCategory: data.category || regEntry?.category || "cortical",
    atlasBadge: data.badge || regEntry?.badge || atlasShort.toUpperCase(),
    isFreeSurferAtlas: isFS,
    lut: lutObj,
    regions: data.regions || null,
    unmaskedBytes,
    atlasEngine: data.engineUsed || data.engine || modeLabel,
    atlasSpace: data.space || "subject",
  });

  setCoregStatus("Ready", false);
  if (data.fallbackUsed) {
    const reasonTxt = data.fallbackReason ? ` (${data.fallbackReason})` : "";
    showToast(
      `${data.shortName || atlasShort}: HIPS-THOMAS fallback -> ${data.engineUsed || data.engine}${reasonTxt}`,
      6500
    );
  } else {
    const cacheTag = data.hipsCacheHit
      ? " [HIPS-THOMAS Cache Hit ~1s]"
      : data.warpCacheHit
      ? " [Warp Cache Hit ~1s]"
      : "";
    showToast(
      `${data.shortName || atlasShort} ready (${data.regionCount || 0} ROIs via ${data.engineUsed || data.engine}${cacheTag})!`,
      5200
    );
  }
  syncAtlasStudioUI();
  return vol;
}

async function cancelActiveAtlasJob() {
  const job = state.activeAtlasJob;
  if (!job || job.cancelling) return false;
  job.cancelling = true;
  job.stage = "Cancelling HIPS-THOMAS container...";
  renderActiveAtlasJobUI();
  setCoregStatus("Cancelling HIPS-THOMAS...", true);
  try {
    await fetch(`/api/atlas-job/${encodeURIComponent(job.jobId)}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    return true;
  } catch (err) {
    console.warn("Cancel atlas job request error:", err);
    return false;
  }
}

async function pollActiveAtlasJobOnce() {
  const job = state.activeAtlasJob;
  if (!job || job.pollInFlight) return null;
  job.pollInFlight = true;
  try {
    const resp = await fetch(`/api/atlas-job/${encodeURIComponent(job.jobId)}`);
    const jobData = await resp.json();
    if (state.activeAtlasJob !== job) return null;

    if (!resp.ok) {
      throw new Error(jobData.error || "Atlas job status failed");
    }

    job.stage = jobData.stage || job.stage;
    if (Array.isArray(jobData.logTail)) job.logTail = jobData.logTail;
    if (typeof jobData.elapsedSec === "number") job.elapsedSec = jobData.elapsedSec;
    renderActiveAtlasJobUI();

    if (jobData.status === "done" && jobData.result) {
      clearActiveAtlasJobTimers(job);
      state.activeAtlasJob = null;
      renderActiveAtlasJobUI();
      return await applyAtlasGenerationResult(jobData.result, {
        atlasId: job.atlasId,
        atlasShort: job.atlasShort,
        regEntry: job.regEntry,
        modeLabel: job.modeLabel,
      });
    }
    if (jobData.status === "cancelled") {
      clearActiveAtlasJobTimers(job);
      state.activeAtlasJob = null;
      renderActiveAtlasJobUI();
      setCoregStatus("Ready", false);
      showToast("HIPS-THOMAS job cancelled.", 3500);
      syncAtlasTypeSelectionUI(false, false);
      syncAtlasStudioUI();
      return null;
    }
    if (jobData.status === "error") {
      clearActiveAtlasJobTimers(job);
      state.activeAtlasJob = null;
      renderActiveAtlasJobUI();
      setCoregStatus("Ready", false);
      showToast(`${job.atlasShort} Error: ${jobData.error || "Background job failed"}`, 5000);
      syncAtlasTypeSelectionUI(false, false);
      syncAtlasStudioUI();
      return null;
    }
    return jobData;
  } catch (err) {
    if (state.activeAtlasJob === job) {
      clearActiveAtlasJobTimers(job);
      state.activeAtlasJob = null;
      renderActiveAtlasJobUI();
      setCoregStatus("Ready", false);
      showToast(`${job.atlasShort} Poll Error: ${err.message}`, 4500);
    }
    return null;
  } finally {
    job.pollInFlight = false;
  }
}

async function runAtlasGeneration(explicitAtlasId = null, explicitMode = null, explicitT1VolId = null) {
  if (!state.nv) return null;
  if (state.activeAtlasJob) {
    showToast("A HIPS-THOMAS job is already running. Wait or click Cancel first.", 3500);
    return null;
  }
  const t1Select = document.getElementById("selectDktT1Vol") || document.getElementById("selectNeuroT1Vol");
  const targetVolId = explicitT1VolId || t1Select?.value || state.refVolId;
  const volIdx = getVolumeIndexById(targetVolId);
  if (volIdx < 0) {
    showToast("Select a T1w structural volume first.");
    return null;
  }

  const srcVol = state.nv.volumes[volIdx];
  const srcMeta = state.layerMeta.get(srcVol.id);
  const atlasId = (explicitAtlasId || document.getElementById("selectAtlasType")?.value || "dkt").toLowerCase();
  const isDestrieux = atlasId === "destrieux";
  const regEntry = getAtlasRegistryEntry(atlasId);
  const atlasShort = regEntry?.shortName || getAtlasShortTitle(null, atlasId);

  if (!srcMeta?.rawBytes) {
    showToast(`Selected volume does not have raw NIfTI bytes available for ${atlasShort}.`);
    return null;
  }

  let mode = explicitMode || document.getElementById("selectDktMode")?.value || regEntry?.defaultMode || "fnirt_nonlinear";

  // Check FreeSurfer license if Destrieux + FastSurfer --fsaparc selected
  if (isDestrieux && !mode.startsWith("template_")) {
    if (!state.neuroToolsStatus) await checkNeuroToolsStatus(false);
    const fsAvail = Boolean(state.neuroToolsStatus?.fastsurfer?.available);
    const licAvail = Boolean(
      state.neuroToolsStatus?.fsLicense?.available || state.neuroToolsStatus?.fastsurfer?.fsLicense?.available
    );
    if (fsAvail && !licAvail) {
      const wantSelect = window.confirm(
        "FastSurfer --fsaparc requires a valid FreeSurfer license file (license.txt). Click OK to choose your license file, or Cancel to run Template Destrieux instead."
      );
      if (wantSelect) {
        state._pendingDestrieuxRunAfterLicense = true;
        document.getElementById("fsLicenseFileInput")?.click();
        return null;
      }
      mode = "template_fnirt";
    }
  }

  const btn = document.getElementById("btnRunDktAtlas");
  const fracInput = document.getElementById("betFracInput");
  const betFrac = Math.max(0.05, Math.min(0.95, parseFloat(fracInput?.value || "0.5") || 0.5));
  const activeBrainVol = findActiveBrainMaskVolume();
  const activeBrainMeta = activeBrainVol ? state.layerMeta.get(activeBrainVol.id) : null;
  const useAsyncHipsJob = atlasId === "thomas" && mode === "hips_thomas_docker";
  const contrastVal = state.atlasStudioState?.hipsContrast === "wmn" ? "wmn" : "t1";
  const forceRecomputeVal = Boolean(state.atlasStudioState?.hipsForceRecompute);
  const customHipsImgVal = String(state.atlasStudioState?.hipsCustomImage || "").trim();

  const modeLabel = mode === "hips_thomas_docker"
    ? `HIPS-THOMAS Container (${contrastVal === "wmn" ? "WMn Native" : "T1w -t1"} — background job, cached per T1w)`
    : mode === "fnirt_nonlinear" || mode === "template_fnirt"
    ? "FNIRT Nonlinear (MNI152 -> Subject)"
    : mode === "flirt_12dof" || mode === "template_flirt"
    ? "FLIRT 12-DOF Affine"
    : mode;

  try {
    if (btn) {
      btn.disabled = true;
      btn.textContent = useAsyncHipsJob ? "Starting HIPS-THOMAS..." : `Generating ${atlasShort}...`;
    }
    setCoregStatus(`Generating ${atlasShort}...`, true);
    showToast(`Generating ${atlasShort} for ${srcMeta.label} via ${modeLabel}...`, useAsyncHipsJob ? 4000 : 0);

    const reqPayload = {
      atlasId,
      atlasType: atlasId,
      sourceName: srcVol.name,
      sourceBase64: uint8ArrayToBase64(srcMeta.rawBytes),
      currentAffine: srcVol.hdr.affine,
      mode,
      dktMode: mode,
      betFrac,
    };
    if (activeBrainMeta?.rawBytes) {
      reqPayload.brainMaskBase64 = uint8ArrayToBase64(activeBrainMeta.rawBytes);
      reqPayload.brainMaskAffine = activeBrainVol.hdr?.affine || null;
    }
    if (atlasId === "thomas") {
      reqPayload.contrast = contrastVal;
      reqPayload.forceRecompute = forceRecomputeVal;
      reqPayload.isSkullStripped = isTargetVolumeSkullStripped(srcVol);
      if (customHipsImgVal) reqPayload.hipsDockerImage = customHipsImgVal;
    }

    if (useAsyncHipsJob) {
      const startResp = await fetch("/api/atlas-job", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqPayload),
      });
      const startData = await startResp.json().catch(() => ({}));
      if (startResp.status === 404 || /unknown api endpoint/i.test(startData?.error || "")) {
        showToast(
          "Older server.py detected — restart server.py for live progress & Cancel. Running synchronous /api/atlas-generate...",
          5500
        );
      } else {
        if (!startResp.ok || !startData.jobId) {
          throw new Error(startData.error || "Could not start HIPS-THOMAS background job");
        }
        const job = {
          jobId: startData.jobId,
          atlasId,
          atlasShort,
          regEntry,
          modeLabel,
          startedAtMs: Date.now(),
          stage: startData.stage || "Starting HIPS-THOMAS...",
          logTail: [],
          cancelling: false,
          pollInFlight: false,
          timerId: null,
          elapsedTimerId: null,
        };
        state.activeAtlasJob = job;
        renderActiveAtlasJobUI();
        job.timerId = setInterval(() => {
          pollActiveAtlasJobOnce();
        }, 1500);
        job.elapsedTimerId = setInterval(() => {
          renderActiveAtlasJobUI();
        }, 1000);
        return { asyncJob: true, jobId: job.jobId };
      }
    }

    const resp = await fetch("/api/atlas-generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqPayload),
    });
    const data = await resp.json();
    if (!resp.ok || !data.niftiBase64) {
      throw new Error(data.error || `${atlasShort} generation failed`);
    }

    return await applyAtlasGenerationResult(data, { atlasId, atlasShort, regEntry, modeLabel });
  } catch (err) {
    console.error(`${atlasShort} generation error:`, err);
    setCoregStatus("Ready", false);
    showToast(`${atlasShort} Error: ${err.message}`, 4500);
    return null;
  } finally {
    if (!state.activeAtlasJob) {
      if (btn) {
        btn.disabled = false;
        syncAtlasTypeSelectionUI(false, false);
      }
      syncAtlasStudioUI();
    }
  }
}

// Backward-compatible entry point
async function runDktSegmentation() {
  return await runAtlasGeneration();
}

async function importCustomAtlasFromFiles({ niftiFile, lutFile = null, atlasName = "", space = "subject", refVolId = null }) {
  if (!niftiFile) {
    showToast("Select a custom atlas NIfTI (.nii, .nii.gz, .mgz) file first.");
    return null;
  }
  try {
    setCoregStatus("Importing Custom Atlas...", true);
    showToast(`Importing atlas "${niftiFile.name}"...`, 0);
    const niiBytes = new Uint8Array(await niftiFile.arrayBuffer());
    const lutText = lutFile ? await lutFile.text() : "";
    const payload = {
      importKind: "custom",
      fileName: niftiFile.name,
      niftiBase64: uint8ArrayToBase64(niiBytes),
      lutText,
      lutFileName: lutFile ? lutFile.name : "",
      atlasName: atlasName || niftiFile.name.replace(/\.(nii\.gz|nii|mgz|mgh)$/i, ""),
      space: space || "subject",
    };

    if (space === "mni152_fnirt" || space === "mni152_flirt") {
      const t1Idx = getVolumeIndexById(refVolId || state.refVolId);
      const refVol = t1Idx >= 0 ? state.nv.volumes[t1Idx] : null;
      const refMeta = refVol ? state.layerMeta.get(refVol.id) : null;
      if (!refMeta?.rawBytes) {
        throw new Error("Reference T1w raw bytes required to warp MNI152 custom atlas to subject space.");
      }
      payload.refName = refVol.name;
      payload.refBase64 = uint8ArrayToBase64(refMeta.rawBytes);
      payload.refAffine = refVol.hdr?.affine || null;
    }

    const resp = await fetch("/api/atlas-import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || !data.niftiBase64) {
      throw new Error(data.error || "Custom atlas import failed");
    }

    const outBytes = base64ToUint8Array(data.niftiBase64);
    const vol = await addAtlasVolumeToWorkspace(outBytes, {
      fileName: data.fileName,
      displayLabel: `${data.atlasName} (${data.regionCount} ROIs)`,
      atlasId: data.atlasId,
      atlasName: data.atlasName,
      atlasShortName: data.shortName,
      atlasCategory: "custom",
      atlasBadge: "CUSTOM",
      isFreeSurferAtlas: false,
      lut: data.lut || {},
      regions: data.regions || null,
      atlasEngine: data.engineUsed,
      atlasSpace: data.space,
    });
    setCoregStatus("Ready", false);
    showToast(`Imported Custom Atlas "${data.atlasName}" (${data.regionCount} ROIs)!`, 4500);
    return vol;
  } catch (err) {
    console.error("Custom atlas import error:", err);
    setCoregStatus("Ready", false);
    showToast(`Custom Atlas Import Error: ${err.message}`, 4500);
    return null;
  }
}

async function importHipsThomasFiles(fileList) {
  const arr = Array.from(fileList || []).filter((f) => /\.(nii|nii\.gz)$/i.test(f.name || ""));
  if (!arr.length) {
    showToast("Select HIPS-THOMAS NIfTI files (sthomas_LR_labels.nii.gz, thomasfull_L/R, or nucleus folder).");
    return null;
  }
  try {
    setCoregStatus("Merging HIPS-THOMAS...", true);
    showToast(`Importing & merging ${arr.length} HIPS-THOMAS file(s)...`, 0);
    const filesPayload = [];
    for (const f of arr) {
      const ab = await f.arrayBuffer();
      filesPayload.push({
        name: f.name,
        relativePath: f.webkitRelativePath || f.name,
        dataBase64: uint8ArrayToBase64(new Uint8Array(ab)),
      });
    }
    const resp = await fetch("/api/atlas-import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        importKind: arr.length > 2 ? "hips_thomas_folder" : "hips_thomas_pair",
        files: filesPayload,
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.niftiBase64) {
      throw new Error(data.error || "HIPS-THOMAS import failed");
    }
    if (data.lut) state.atlasLuts.set("thomas", data.lut);
    const outBytes = base64ToUint8Array(data.niftiBase64);
    const vol = await addAtlasVolumeToWorkspace(outBytes, {
      fileName: data.fileName,
      displayLabel: `${data.atlasName} (${data.regionCount} Nuclei)`,
      atlasId: "thomas",
      atlasName: data.atlasName,
      atlasShortName: "THOMAS",
      atlasCategory: "thalamic",
      atlasBadge: "THOMAS",
      isFreeSurferAtlas: false,
      lut: data.lut,
      regions: data.regions || null,
      atlasEngine: data.engineUsed,
      atlasSpace: "subject",
    });
    setCoregStatus("Ready", false);
    showToast(`Imported HIPS-THOMAS Atlas (${data.regionCount} thalamic nuclei)!`, 4500);
    return vol;
  } catch (err) {
    console.error("HIPS-THOMAS import error:", err);
    setCoregStatus("Ready", false);
    showToast(`HIPS-THOMAS Import Error: ${err.message}`, 4500);
    return null;
  }
}

// Build a NiiVue discrete label LUT ({ R, G, B, A, I, labels }) from FreeSurferColorLUT (state.dktLut)
async function ensureDktLutLoaded() {
  if (state.dktLut && Object.keys(state.dktLut).length > 0) {
    return state.dktLut;
  }
  try {
    const resp = await fetch("/sample_data/dkt_lut.json");
    if (resp.ok) {
      state.dktLut = await resp.json();
    }
  } catch (_) {
    // ignore
  }
  return state.dktLut || {};
}

// FreeSurfer LUT IDs that represent background, exterior, extracerebral CSF, skull, scalp, air, or unknown regions outside the brain
const NON_BRAIN_FREESURFER_LABELS = new Set([
  0, 1, 6, 24, 29, 30, 40, 45, 61, 62, 80, 81, 82, 85, 98,
  118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
  140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
  165, 166, 167, 168, 224,
  256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
  701, 901, 902, 906, 907, 908, 909, 911, 915, 916, 920, 930, 931,
  1000, 1001, 1100, 2000, 2001, 2100, 11100, 12100,
]);

const NON_BRAIN_FREESURFER_NAME_RE =
  /unknown|exterior|extracerebral|skull|scalp|cranium|bone|air|fat|adipose|muscle|skin|epidermis|dura|sinus|eyeball|globe|cornea|vitreous|lens|aqueous|iris|ear|tongue|nasal|teeth|eustachian|background|unchanged|vessel|artery|mucosa|periosteum|endosteum|diploe|table|\bcsf\b|csf-sa|csf-fsl|cortical-csf|spinal-csf/i;

function isBrainFreeSurferLutEntry(id, entry) {
  const numId = Number(id);
  if (!Number.isFinite(numId) || numId <= 0) return false;
  if (NON_BRAIN_FREESURFER_LABELS.has(numId)) return false;
  if (!entry || !Array.isArray(entry.rgb)) return false;
  const r = entry.rgb[0] ?? 0;
  const g = entry.rgb[1] ?? 0;
  const b = entry.rgb[2] ?? 0;
  if (r === 0 && g === 0 && b === 0) return false;
  if (entry.name && NON_BRAIN_FREESURFER_NAME_RE.test(entry.name)) return false;
  return true;
}

function buildFreeSurferNiiVueLabelColormap(dktLutObj, maxVolLabel = 2100, calMin = 0.5, calMax = Infinity) {
  const lutSource = dktLutObj && Object.keys(dktLutObj).length > 0 ? dktLutObj : {};
  const lutIds = Object.keys(lutSource)
    .map((k) => parseInt(k, 10))
    .filter((n) => Number.isFinite(n) && n >= 0);
  const maxId = Math.min(16384, Math.max(2100, maxVolLabel, ...(lutIds.length ? lutIds : [2100])));
  const effMin = Number.isFinite(Number(calMin)) ? Math.max(0.5, Number(calMin)) : 0.5;
  const effMax = Number.isFinite(Number(calMax)) && Number(calMax) >= effMin ? Number(calMax) : maxId;

  const R = new Array(maxId + 1).fill(0);
  const G = new Array(maxId + 1).fill(0);
  const B = new Array(maxId + 1).fill(0);
  const A = new Array(maxId + 1).fill(0);
  const I = new Array(maxId + 1);
  const labels = new Array(maxId + 1).fill("");

  for (let i = 0; i <= maxId; i++) {
    I[i] = i;
    const entry = lutSource[String(i)];
    if (i >= effMin && i <= effMax && isBrainFreeSurferLutEntry(i, entry)) {
      R[i] = entry.rgb[0] ?? 0;
      G[i] = entry.rgb[1] ?? 0;
      B[i] = entry.rgb[2] ?? 0;
      A[i] = 72;
      labels[i] = entry.name || `ROI_${i}`;
    } else {
      R[i] = 0;
      G[i] = 0;
      B[i] = 0;
      A[i] = 0;
      labels[i] = "";
    }
  }
  return { R, G, B, A, I, labels };
}

function buildAtlasDiscreteColormap(vol, meta) {
  const isFS = Boolean(meta?.isFreeSurferAtlas);
  const lutObj =
    meta?.lut && Object.keys(meta.lut).length > 0
      ? meta.lut
      : isFS
      ? state.dktLut || {}
      : {};
  const rawArr = meta?.rawAtlasVoxels || vol?.img || [];
  let maxVox = 1;
  for (let i = 0; i < rawArr.length; i++) {
    const v = Math.round(rawArr[i]);
    if (v > maxVox && v <= 65535) maxVox = v;
  }
  let maxLutId = 1;
  for (const k of Object.keys(lutObj)) {
    const n = parseInt(k, 10);
    if (Number.isFinite(n) && n > maxLutId && n <= 65535) maxLutId = n;
  }

  if (isFS) {
    const cmap = buildFreeSurferNiiVueLabelColormap(
      lutObj,
      Math.max(maxVox, maxLutId),
      vol?.cal_min ?? 0.5,
      vol?.cal_max ?? Infinity
    );
    return { ...cmap, maxLabelId: cmap.I.length - 1 };
  }

  const maxLabelId = Math.min(65535, Math.max(1, maxVox, maxLutId));
  const R = new Array(maxLabelId + 1).fill(0);
  const G = new Array(maxLabelId + 1).fill(0);
  const B = new Array(maxLabelId + 1).fill(0);
  const A = new Array(maxLabelId + 1).fill(0);
  const I = new Array(maxLabelId + 1);
  const labels = new Array(maxLabelId + 1).fill("");

  for (let i = 0; i <= maxLabelId; i++) {
    I[i] = i;
    if (i === 0) {
      A[0] = 0;
      continue;
    }
    const entry = lutObj[String(i)];
    if (entry && Array.isArray(entry.rgb)) {
      R[i] = entry.rgb[0] ?? 128;
      G[i] = entry.rgb[1] ?? 128;
      B[i] = entry.rgb[2] ?? 128;
      A[i] = 255;
      labels[i] = entry.name || entry.title || `ROI_${i}`;
    } else {
      const [dr, dg, db] = generateDeterministicRgb(i, "C");
      R[i] = dr;
      G[i] = dg;
      B[i] = db;
      A[i] = 255;
      labels[i] = `Region ${i}`;
    }
  }
  return { R, G, B, A, I, labels, maxLabelId };
}

function syncOverlayStyleButtonsUI() {
  const activeStyle = state.atlasOverlayStyle || "filled";
  document.querySelectorAll(".atlas-style-btn[data-atlas-style]").forEach((btn) => {
    const s = btn.getAttribute("data-atlas-style");
    btn.classList.toggle("active", s === activeStyle);
  });
}

async function applyAtlasColormapToVolume(vol) {
  if (!vol) return;
  let meta = state.layerMeta.get(vol.id);
  const isDiscreteAtlas = Boolean(meta?.isAtlas || meta?.isDktAtlas);
  const isFreeSurfer = meta ? Boolean(meta.isFreeSurferAtlas) : true;
  const aid = resolveAtlasTypeForVolume(vol);

  if (isFreeSurfer) {
    await ensureDktLutLoaded();
  } else if (meta && (!meta.lut || Object.keys(meta.lut).length === 0) && aid) {
    meta.lut = await ensureAtlasLutLoaded(aid);
  }

  // Register fallback 256-bin FreeSurferColorLUT once if needed by NiiVue
  if (state.nv && typeof state.nv.addColormap === "function" && !state._fsColormapRegistered) {
    const dktLutObj = state.dktLut || {};
    const cR = new Array(256).fill(0);
    const cG = new Array(256).fill(0);
    const cB = new Array(256).fill(0);
    const cA = new Array(256).fill(0);
    const cI = new Array(256).fill(0);
    const validEntries = Object.values(dktLutObj).filter((e) => e && isBrainFreeSurferLutEntry(e.id, e));
    for (let k = 0; k < 256; k++) {
      cI[k] = k;
      if (k > 48) {
        const pick = validEntries.length > 0 ? validEntries[k % validEntries.length] : null;
        cR[k] = pick ? pick.rgb[0] : (k * 73) % 200 + 55;
        cG[k] = pick ? pick.rgb[1] : (k * 151) % 200 + 55;
        cB[k] = pick ? pick.rgb[2] : (k * 199) % 200 + 55;
        cA[k] = Math.round(12 + ((k - 48) / 207) * 76);
      }
    }
    try {
      state.nv.addColormap("FreeSurferColorLUT", { R: cR, G: cG, B: cB, A: cA, I: cI });
      state._fsColormapRegistered = true;
    } catch (_) {
      // ignore
    }
  }

  if (isDiscreteAtlas || meta?.isMask || /dkt|destrieux|a2009s|aparc|aseg|atlas|thomas|tian/i.test(vol.name || "")) {
    if (!vol.hdr) vol.hdr = {};
    vol.hdr.intent_code = 1002;
    vol.hdr.scl_slope = 1;
    vol.hdr.scl_inter = 0;

    if (vol.img && vol.img.length) {
      if (meta && !meta.rawAtlasVoxels) {
        const snap = new Int32Array(vol.img.length);
        for (let i = 0; i < vol.img.length; i++) {
          snap[i] = Math.round(vol.img[i]);
        }
        meta.rawAtlasVoxels = snap;
      }
      const rawSrc = meta?.rawAtlasVoxels || meta?.unmaskedDktData || vol.img;
      const isolatedSet = meta?.visibleRegionIds instanceof Set ? meta.visibleRegionIds : null;
      const fsLut = meta?.lut || state.dktLut || {};
      let maxVox = 1;
      for (let i = 0; i < vol.img.length; i++) {
        const lbl = Math.round(rawSrc[i]);
        let keep = lbl > 0;
        if (keep && isFreeSurfer) {
          keep = isBrainFreeSurferLutEntry(lbl, fsLut[String(lbl)]);
        }
        if (keep && isolatedSet) {
          keep = isolatedSet.has(lbl);
        }
        if (!keep) {
          vol.img[i] = 0;
        } else {
          vol.img[i] = lbl;
          if (lbl > maxVox) maxVox = lbl;
        }
      }
      if (isFreeSurfer) {
        if (!Number.isFinite(vol.cal_min) || vol.cal_min < 0.5) vol.cal_min = 0.5;
        const fsMaxTarget = maxVox >= 11100 ? 12175 : Math.max(maxVox, getAtlasDefaultMaxLabel(vol));
        if (!meta?.userCustomWindow || !Number.isFinite(vol.cal_max) || vol.cal_max <= 0) {
          vol.cal_max = Math.max(vol.cal_max || 0, fsMaxTarget);
        }
      } else {
        vol.cal_min = 0;
        vol.cal_max = Math.max(1, getAtlasDefaultMaxLabel(vol), maxVox);
      }
    }

    const labelCmap = buildAtlasDiscreteColormap(vol, meta || { isFreeSurferAtlas: isFreeSurfer });
    labelCmap.A[0] = 0;
    vol.colormap = "FreeSurferColorLUT";
    if (typeof vol.setColormapLabel === "function") {
      vol.setColormapLabel(labelCmap);
    } else {
      vol.colormapLabel = labelCmap;
    }

    // Apply Filled / +Edge / Outline style & opacity
    const isVisible = meta ? Boolean(meta.visible) : true;
    const baseOp = meta?.userAtlasOpacity > 0 ? meta.userAtlasOpacity : meta?.savedOpacity > 0 ? meta.savedOpacity : 0.48;
    const style = state.atlasOverlayStyle || "filled";
    let outlineVal = 0;
    let targetOp = baseOp;
    if (style === "filled_outline") {
      outlineVal = 0.95;
      targetOp = Math.min(baseOp, 0.55);
    } else if (style === "outline") {
      outlineVal = 0.95;
      targetOp = 0.02;
    }
    const finalOp = isVisible ? targetOp : 0.0;
    const volIdx = getVolumeIndexById(vol.id);
    if (volIdx >= 0 && typeof state.nv?.setOpacity === "function") {
      state.nv.setOpacity(volIdx, finalOp);
    } else {
      vol.opacity = finalOp;
    }
    if (isVisible && state.nv) {
      if (typeof state.nv.setAtlasOutline === "function") {
        state.nv.setAtlasOutline(outlineVal);
      }
      if (state.nv.opts) {
        state.nv.opts.atlasOutline = outlineVal;
      }
      syncOverlayStyleButtonsUI();
    }
  } else {
    // Continuous structural volume branch: preserve existing masking + threshold logic
    vol.colormapLabel = null;
    vol.colormap = "FreeSurferColorLUT";
    if (vol.img && vol.img.length) {
      if (meta && !meta.unmaskedVolumeData) {
        meta.unmaskedVolumeData = vol.img.slice();
        meta.savedCalMinBeforeFsLut = vol.cal_min;
      }
      const origData = meta?.unmaskedVolumeData || vol.img;
      const maskVol = findActiveBrainMaskVolume() || getDktVolume();
      if (maskVol && maskVol !== vol && maskVol.img && maskVol.img.length === vol.img.length) {
        const mImg = maskVol.img;
        for (let i = 0; i < vol.img.length; i++) {
          vol.img[i] = mImg[i] > 0 ? origData[i] : 0;
        }
      }
      let maxVal = 0;
      for (let i = 0; i < origData.length; i += 4) {
        if (origData[i] > maxVal) maxVal = origData[i];
      }
      if (maxVal > 0 && (!vol.cal_min || vol.cal_min < maxVal * 0.28)) {
        vol.cal_min = Math.round(maxVal * 0.28);
      }
    }
  }
}

async function applyFreeSurferColorLutToVolume(vol) {
  return await applyAtlasColormapToVolume(vol);
}

async function setAtlasOverlayStyle(style) {
  const norm = style === "filled_outline" || style === "outline" ? style : "filled";
  state.atlasOverlayStyle = norm;
  syncOverlayStyleButtonsUI();
  const activeVol = getDktVolume();
  if (activeVol) {
    await applyAtlasColormapToVolume(activeVol);
    state.nv?.updateGLVolume?.();
    state.nv?.drawScene?.();
  } else if (state.nv) {
    const ov = norm === "filled" ? 0 : 0.95;
    if (typeof state.nv.setAtlasOutline === "function") state.nv.setAtlasOutline(ov);
    if (state.nv.opts) state.nv.opts.atlasOutline = ov;
    state.nv.drawScene?.();
  }
  syncDktOverlayControlsUI();
  syncAtlasStudioUI();
}

async function activateAtlasOverlay(volId, visible = true) {
  if (!state.nv) return;
  const idx = getVolumeIndexById(volId);
  if (idx < 0) return;
  const vol = state.nv.volumes[idx];
  const meta = state.layerMeta.get(volId);
  if (!meta) return;

  if (visible) {
    state.activeDktVolId = volId;
    state.activeAtlasVolId = volId;
    state.atlasStudioState.selectedAtlasVolId = volId;
    state.nv.volumes.forEach((otherVol, otherIdx) => {
      if (otherVol.id !== volId) {
        const om = state.layerMeta.get(otherVol.id);
        if (om && (om.isAtlas || om.isDktAtlas)) {
          om.visible = false;
          state.nv.setOpacity?.(otherIdx, 0);
        }
      }
    });
    meta.visible = true;
    await applyAtlasColormapToVolume(vol);
    const atlasSelect = document.getElementById("selectAtlasType");
    if (atlasSelect && meta.atlasId && Array.from(atlasSelect.options).some((o) => o.value === meta.atlasId)) {
      if (atlasSelect.value !== meta.atlasId) {
        atlasSelect.value = meta.atlasId;
        syncAtlasTypeSelectionUI(false, false);
      }
    }
  } else {
    meta.visible = false;
    state.nv.setOpacity?.(idx, 0);
  }

  state.nv.updateGLVolume?.();
  state.nv.drawScene?.();
  renderLayerList();
  syncDktOverlayControlsUI();
  syncAtlasStudioUI();
  updateDktReadoutUI(lookupDktRegionAtRAS(state.lastCrosshairRAS), null);
}

function computeClientAtlasRegions(vol, lutObj = {}) {
  const meta = vol ? state.layerMeta.get(vol.id) : null;
  const data = meta?.rawAtlasVoxels || vol?.img;
  if (!vol || !data || !vol.hdr?.affine) return [];
  const dims = getVolDims(vol);
  const nx = dims[1] || 0;
  const ny = dims[2] || 0;
  const nz = dims[3] || 0;
  if (!(nx > 0 && ny > 0 && nz > 0)) return [];

  const aff = vol.hdr.affine;
  const det =
    aff[0][0] * (aff[1][1] * aff[2][2] - aff[1][2] * aff[2][1]) -
    aff[0][1] * (aff[1][0] * aff[2][2] - aff[1][2] * aff[2][0]) +
    aff[0][2] * (aff[1][0] * aff[2][1] - aff[1][1] * aff[2][0]);
  const voxVol = Math.abs(det) || 1.0;

  const acc = new Map();
  const sliceStride = nx * ny;
  for (let k = 0; k < nz; k++) {
    const kOff = k * sliceStride;
    for (let j = 0; j < ny; j++) {
      const rowOff = kOff + j * nx;
      for (let i = 0; i < nx; i++) {
        const lbl = Math.round(data[rowOff + i]);
        if (lbl <= 0) continue;
        let rec = acc.get(lbl);
        if (!rec) {
          rec = { c: 0, si: 0, sj: 0, sk: 0 };
          acc.set(lbl, rec);
        }
        rec.c++;
        rec.si += i;
        rec.sj += j;
        rec.sk += k;
      }
    }
  }

  const out = [];
  for (const [lid, rec] of acc.entries()) {
    const entry = lutObj?.[String(lid)];
    const name = entry?.name || entry?.title || `Region ${lid}`;
    const shortName = entry?.shortName || entry?.abbr || name;
    const rgb = entry?.rgb || generateDeterministicRgb(lid);
    const hex = entry?.hex || `#${rgb.map((c) => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, "0")).join("")}`;
    let hemi = entry?.hemisphere || "C";
    if (!entry?.hemisphere) {
      const low = name.toLowerCase();
      if (low.startsWith("left") || low.startsWith("l-") || low.endsWith("-lh") || low.startsWith("ctx-lh")) hemi = "L";
      else if (low.startsWith("right") || low.startsWith("r-") || low.endsWith("-rh") || low.startsWith("ctx-rh")) hemi = "R";
      else if (meta?.atlasId === "thomas") hemi = lid >= 100 ? "R" : "L";
    }
    const centroidRAS = applyMat4Pt(aff, [rec.si / rec.c, rec.sj / rec.c, rec.sk / rec.c]).map((v) =>
      Math.round(v * 100) / 100
    );
    out.push({
      id: lid,
      name,
      shortName,
      hemisphere: hemi,
      voxelCount: rec.c,
      volumeMm3: Math.round(rec.c * voxVol * 100) / 100,
      centroidRAS,
      rgb,
      hex,
    });
  }
  out.sort((a, b) => a.id - b.id);
  return out;
}

// Find the active BET / skull-stripped brain mask volume in the workspace if present
function findActiveBrainMaskVolume() {
  if (!state.nv || !Array.isArray(state.nv.volumes)) return null;
  const betVol = state.nv.volumes.find((v) => state.layerMeta.get(v.id)?.isBetBrain);
  if (betVol) return betVol;
  const brainNamedVol = state.nv.volumes.find((v) => {
    const m = state.layerMeta.get(v.id);
    if (m?.isDktAtlas || m?.isAtlas) return false;
    return /_brain(\.nii(\.gz)?$|\b)/i.test(v.name || "");
  });
  return brainNamedVol || null;
}

// Mask ONLY FreeSurfer (DKT / Destrieux) cortical/whole-brain atlases against the subject BET brain mask
async function applyBrainMaskToDktVolume(dktVol = null, brainVolOverride = null) {
  const targetDktVol = dktVol || getDktVolume();
  if (!targetDktVol || !targetDktVol.img) return false;

  const dktMeta = state.layerMeta.get(targetDktVol.id);
  if (dktMeta && !dktMeta.isFreeSurferAtlas) {
    // Subcortical, thalamic (THOMAS), Tian, FSL, and custom atlases must NEVER be clipped by rough BET masks
    return false;
  }

  const dktLutObj = await ensureDktLutLoaded();
  const brainVol = brainVolOverride || findActiveBrainMaskVolume();
  if (!brainVol || !brainVol.img) return false;

  if (dktMeta && !dktMeta.unmaskedDktData) {
    dktMeta.unmaskedDktData = dktMeta.rawAtlasVoxels ? new Int16Array(dktMeta.rawAtlasVoxels) : new Int16Array(targetDktVol.img);
  }
  const sourceLabels = dktMeta?.unmaskedDktData || targetDktVol.img;

  const dDims = targetDktVol.hdr?.dims || [];
  const bDims = brainVol.hdr?.dims || [];
  const dNx = dDims[1] || 0;
  const dNy = dDims[2] || 0;
  const dNz = dDims[3] || 0;
  const bNx = bDims[1] || 0;
  const bNy = bDims[2] || 0;
  const bNz = bDims[3] || 0;

  const sameDims = dNx > 0 && dNx === bNx && dNy === bNy && dNz === bNz && targetDktVol.img.length === brainVol.img.length;
  let sameAffine = sameDims;
  if (sameDims && targetDktVol.hdr?.affine && brainVol.hdr?.affine) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 4; c++) {
        if (Math.abs((targetDktVol.hdr.affine[r][c] || 0) - (brainVol.hdr.affine[r][c] || 0)) > 1e-2) {
          sameAffine = false;
          break;
        }
      }
    }
  }

  if (sameDims && sameAffine) {
    const bImg = brainVol.img;
    const dImg = targetDktVol.img;
    for (let i = 0; i < dImg.length; i++) {
      const lbl = Math.round(sourceLabels[i]);
      dImg[i] = bImg[i] > 0 && isBrainFreeSurferLutEntry(lbl, dktLutObj[String(lbl)]) ? lbl : 0;
    }
  } else if (targetDktVol.hdr?.affine && brainVol.hdr?.affine && dNx > 0 && dNy > 0 && dNz > 0) {
    const dAff = targetDktVol.hdr.affine;
    const bInvAff = invertMat4(brainVol.hdr.affine);
    const dImg = targetDktVol.img;
    const sliceStride = dNx * dNy;
    for (let k = 0; k < dNz; k++) {
      for (let j = 0; j < dNy; j++) {
        const rowOffset = k * sliceStride + j * dNx;
        const baseX = dAff[0][1] * j + dAff[0][2] * k + dAff[0][3];
        const baseY = dAff[1][1] * j + dAff[1][2] * k + dAff[1][3];
        const baseZ = dAff[2][1] * j + dAff[2][2] * k + dAff[2][3];
        for (let i = 0; i < dNx; i++) {
          const idx = rowOffset + i;
          const origLabel = Math.round(sourceLabels[idx]);
          if (origLabel <= 0 || !isBrainFreeSurferLutEntry(origLabel, dktLutObj[String(origLabel)])) {
            dImg[idx] = 0;
            continue;
          }
          const ras = [
            dAff[0][0] * i + baseX,
            dAff[1][0] * i + baseY,
            dAff[2][0] * i + baseZ,
          ];
          const bVal = sampleVolumeAtRAS(brainVol, bInvAff, ras, true);
          dImg[idx] = bVal > 0 ? origLabel : 0;
        }
      }
    }
  }

  if (dktMeta) {
    dktMeta.rawAtlasVoxels = new Int16Array(targetDktVol.img.length);
    for (let i = 0; i < targetDktVol.img.length; i++) {
      dktMeta.rawAtlasVoxels[i] = Math.round(targetDktVol.img[i]);
    }
  }
  targetDktVol.cal_min = 0.5;
  await applyAtlasColormapToVolume(targetDktVol);
  return true;
}

async function addAtlasVolumeToWorkspace(niftiBytes, opts = {}) {
  const fileName = opts.fileName || "atlas.nii.gz";
  const inferred = inferModalityAndStyle(fileName);
  const atlasId = opts.atlasId || opts.explicitAtlasType || inferred.atlasId || "dkt";
  const isFS =
    opts.isFreeSurferAtlas !== undefined
      ? Boolean(opts.isFreeSurferAtlas)
      : atlasId === "dkt" || atlasId === "destrieux";

  // Replace existing volume ONLY if it shares exact atlasId + fileName (allows DKT + THOMAS + Tian S1..S4 simultaneously!)
  const existingIdx = state.nv.volumes.findIndex((v) => {
    const m = state.layerMeta.get(v.id);
    if (!m?.isDktAtlas && !m?.isAtlas) return false;
    return (m.atlasId === atlasId && atlasId !== "custom" && !String(atlasId).startsWith("custom_")) || v.name === fileName;
  });
  if (existingIdx >= 0) {
    const oldId = state.nv.volumes[existingIdx].id;
    state.nv.removeVolumeByIndex(existingIdx);
    state.layerMeta.delete(oldId);
  }

  let lutObj = opts.lut || null;
  if (isFS) {
    lutObj = await ensureDktLutLoaded();
  } else if (!lutObj || Object.keys(lutObj).length === 0) {
    lutObj = await ensureAtlasLutLoaded(atlasId);
  }

  const atlasTitle = opts.atlasName || inferred.modality || "Brain Atlas";
  const atlasShort = opts.atlasShortName || inferred.atlasShortName || atlasId.toUpperCase();
  const initialOp = opts.opacity ?? (isFS ? 0.38 : 0.52);

  const vol = await loadNVImageFromBytes(niftiBytes, {
    name: fileName,
    colormap: "gray",
    opacity: initialOp,
    cal_min: isFS ? 0.5 : 0,
    cal_max: isFS ? (atlasId === "destrieux" ? 12175 : 2035) : 255,
  });

  if (!vol.hdr) vol.hdr = {};
  vol.hdr.intent_code = 1002;
  vol.hdr.scl_slope = 1;
  vol.hdr.scl_inter = 0;

  let unmaskedDktData = null;
  if (opts.unmaskedBytes && isFS) {
    try {
      const unmaskedVol = await loadNVImageFromBytes(opts.unmaskedBytes, {
        name: `unmasked_${fileName}`,
        colormap: "gray",
        opacity: 0.0,
      });
      if (unmaskedVol?.img?.length === vol.img?.length) {
        unmaskedDktData = new Int16Array(unmaskedVol.img);
      }
    } catch (_) {
      // ignore
    }
  }

  const rawAtlasVoxels = new Int16Array(vol.img?.length || 0);
  if (vol.img) {
    for (let i = 0; i < vol.img.length; i++) {
      rawAtlasVoxels[i] = Math.round(vol.img[i]);
    }
  }
  if (!unmaskedDktData) {
    unmaskedDktData = new Int16Array(rawAtlasVoxels);
  }

  state.nv.addVolume(vol);
  registerVolumeMeta(vol, {
    label: opts.displayLabel || `${atlasTitle}`,
    modality: `${atlasShort} (${isFS ? "FreeSurferColorLUT" : "Discrete Label LUT"})`,
    opacity: initialOp,
    userAtlasOpacity: initialOp,
    visible: true,
    isMask: true,
    isAtlas: true,
    isDktAtlas: true,
    atlasType: atlasId,
    atlasId,
    atlasName: atlasTitle,
    atlasShortName: atlasShort,
    atlasCategory: opts.atlasCategory || inferred.atlasCategory || "cortical",
    atlasBadge: opts.atlasBadge || inferred.atlasBadge || atlasShort.toUpperCase(),
    isFreeSurferAtlas: isFS,
    lut: lutObj,
    regions: Array.isArray(opts.regions) ? opts.regions : null,
    rawAtlasVoxels,
    atlasEngine: opts.atlasEngine || null,
    atlasSpace: opts.atlasSpace || "subject",
    rawBytes: niftiBytes,
  });

  const meta = state.layerMeta.get(vol.id);
  if (meta) {
    meta.unmaskedDktData = unmaskedDktData;
    if (!meta.regions || meta.regions.length === 0) {
      meta.regions = computeClientAtlasRegions(vol, lutObj);
    }
  }

  if (isFS && findActiveBrainMaskVolume()) {
    await applyBrainMaskToDktVolume(vol);
  }

  await activateAtlasOverlay(vol.id, true);
  return vol;
}

// Backward-compatible DKT / Destrieux wrapper
async function addDktAtlasVolumeToWorkspace(dktBytes, fileName, displayLabel, unmaskedBytes = null, explicitAtlasType = null) {
  const detectedId = resolveAtlasTypeForVolume(null, explicitAtlasType || fileName);
  const reg = getAtlasRegistryEntry(detectedId);
  const inf = inferModalityAndStyle(fileName);
  const isFS = detectedId === "dkt" || detectedId === "destrieux";
  return await addAtlasVolumeToWorkspace(dktBytes, {
    fileName,
    displayLabel,
    unmaskedBytes,
    explicitAtlasType: detectedId,
    atlasId: detectedId,
    atlasName: reg?.name || inf.modality,
    atlasShortName: reg?.shortName || inf.atlasShortName || (detectedId === "destrieux" ? "Destrieux" : "DKT"),
    atlasCategory: reg?.category || inf.atlasCategory || "cortical",
    atlasBadge: reg?.badge || inf.atlasBadge || (detectedId === "destrieux" ? "DESTRIEUX" : "DKT"),
    isFreeSurferAtlas: isFS,
  });
}

async function loadDktAtlasFile(file) {
  if (!file) return;
  const inf = inferModalityAndStyle(file.name);
  const selectedId = document.getElementById("selectAtlasType")?.value || "dkt";
  const detectedId = inf.atlasId && inf.atlasId !== "dkt" ? inf.atlasId : /dkt/i.test(file.name) ? "dkt" : selectedId;
  const isFS = detectedId === "dkt" || detectedId === "destrieux";
  try {
    setCoregStatus("Loading Atlas...", true);
    showToast(`Loading Atlas file ${file.name}...`, 0);

    if (isFS && (!state.dktLut || Object.keys(state.dktLut).length === 0)) {
      await checkNeuroToolsStatus(false);
    }

    const ab = await file.arrayBuffer();
    let atlasBytes = new Uint8Array(ab);
    let fileName = file.name;
    let finalId = detectedId;
    let lutObj = isFS ? state.dktLut : await ensureAtlasLutLoaded(finalId);

    if (file.name.toLowerCase().endsWith(".mgz") || file.name.toLowerCase().endsWith(".mgh")) {
      const resp = await fetch("/api/convert-volume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sourceName: file.name,
          sourceBase64: uint8ArrayToBase64(atlasBytes),
          isDktAtlas: isFS,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.niftiBase64) {
        throw new Error(data.error || "Failed to convert FreeSurfer .mgz volume");
      }
      atlasBytes = base64ToUint8Array(data.niftiBase64);
      fileName = data.fileName;
      if (data.dktLut && isFS) lutObj = state.dktLut = data.dktLut;
      if (data.atlasType && isFS) finalId = data.atlasType;
    }

    const reg = getAtlasRegistryEntry(finalId);
    const shortTitle = reg?.shortName || inf.atlasShortName || finalId.toUpperCase();
    await addAtlasVolumeToWorkspace(atlasBytes, {
      fileName,
      displayLabel: `${shortTitle} (${file.name})`,
      atlasId: finalId,
      atlasName: reg?.name || `${shortTitle} Atlas`,
      atlasShortName: shortTitle,
      atlasCategory: reg?.category || inf.atlasCategory || "cortical",
      atlasBadge: reg?.badge || inf.atlasBadge || shortTitle.toUpperCase(),
      isFreeSurferAtlas: isFS,
      lut: lutObj,
    });
    setCoregStatus("Ready", false);
    showToast(`Loaded ${shortTitle} "${file.name}"! Hover cursor or move crosshair to inspect stacked atlases.`);
  } catch (err) {
    console.error("Failed to load Atlas file:", err);
    setCoregStatus("Ready", false);
    showToast(`Load Atlas Error: ${err.message}`, 4500);
  }
}

// Find the active Brain Atlas NVImage volume in workspace
function getDktVolume() {
  if (!state.nv) return null;
  const activeId = state.activeAtlasVolId || state.activeDktVolId;
  if (activeId) {
    const idx = getVolumeIndexById(activeId);
    if (idx >= 0) return state.nv.volumes[idx];
  }
  const preferredType = document.getElementById("selectAtlasType")?.value || state.selectedAtlasType || "dkt";
  const preferred = state.nv.volumes.find((v) => {
    const m = state.layerMeta.get(v.id);
    return (m?.isAtlas || m?.isDktAtlas) && resolveAtlasTypeForVolume(v) === preferredType;
  });
  if (preferred) {
    state.activeDktVolId = preferred.id;
    state.activeAtlasVolId = preferred.id;
    return preferred;
  }
  const found = state.nv.volumes.find((v) => {
    const m = state.layerMeta.get(v.id);
    return m?.isAtlas || m?.isDktAtlas;
  });
  if (found) {
    state.activeDktVolId = found.id;
    state.activeAtlasVolId = found.id;
    return found;
  }
  return null;
}

function lookupAtlasRegionAtRAS(vol, ras) {
  if (!vol || !vol.hdr?.affine) return null;
  const meta = state.layerMeta.get(vol.id);
  const srcArr = meta?.rawAtlasVoxels || vol.img;
  if (!srcArr || !srcArr.length) return null;

  const dims = getVolDims(vol);
  const nx = dims[1] || 0;
  const ny = dims[2] || 0;
  const nz = dims[3] || 0;
  if (!(nx > 0 && ny > 0 && nz > 0)) return null;

  const invAff = invertMat4(vol.hdr.affine);
  const ijk = applyMat4Pt(invAff, ras);
  const i = Math.round(ijk[0]);
  const j = Math.round(ijk[1]);
  const k = Math.round(ijk[2]);
  if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return null;

  const labelId = Math.round(srcArr[i + nx * (j + ny * k)]);
  if (!labelId || labelId <= 0) return null;

  const isFS = meta ? Boolean(meta.isFreeSurferAtlas) : true;
  const lutObj = meta?.lut && Object.keys(meta.lut).length > 0 ? meta.lut : isFS ? state.dktLut : {};
  let entry = lutObj?.[String(labelId)] || null;
  if (isFS && !isBrainFreeSurferLutEntry(labelId, entry)) return null;

  if (!entry) {
    const [r, g, b] = generateDeterministicRgb(labelId);
    const hex = `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`;
    entry = { id: labelId, name: `Region ${labelId}`, shortName: `R${labelId}`, title: `Region ${labelId}`, hemisphere: "C", rgb: [r, g, b], hex };
  }

  const badge = getAtlasBadgeInfo(meta);
  const shortAtlas = meta?.atlasShortName || getAtlasShortTitle(vol);
  return {
    volId: vol.id,
    atlasId: meta?.atlasId || meta?.atlasType || "atlas",
    atlasTitle: shortAtlas,
    atlasBadge: badge.text,
    atlasBadgeClass: badge.cls,
    labelId,
    name: entry.name || entry.title || `Region ${labelId}`,
    shortName: entry.shortName || entry.name || `Region ${labelId}`,
    title: entry.title || entry.name || `Region ${labelId}`,
    hemisphere: entry.hemisphere || "C",
    rgb: entry.rgb || [56, 189, 248],
    hex: entry.hex || "#38bdf8",
    entry,
    activeOverlay: vol.id === (state.activeAtlasVolId || state.activeDktVolId),
    ras,
  };
}

function lookupAllAtlasesAtRAS(ras) {
  const vols = getLoadedAtlasVolumes();
  if (!vols.length || !Array.isArray(ras)) return [];
  const activeId = state.activeAtlasVolId || state.activeDktVolId;
  const sorted = [...vols].sort((a, b) => (a.id === activeId ? -1 : b.id === activeId ? 1 : 0));
  const hits = [];
  for (const v of sorted) {
    const hit = lookupAtlasRegionAtRAS(v, ras);
    if (hit) hits.push(hit);
  }
  return hits;
}

// Backward-compatible primary lookup (used by legacy hover/inspector)
function lookupDktRegionAtRAS(ras) {
  const activeVol = getDktVolume();
  if (!activeVol) return null;
  const hits = lookupAllAtlasesAtRAS(ras);
  const primary = hits[0] || null;
  if (!primary) {
    return { labelId: 0, entry: null, ras, atlasTitle: getAtlasShortTitle(activeVol), hits };
  }
  return {
    labelId: primary.labelId,
    entry: primary.entry,
    ras,
    atlasTitle: primary.atlasTitle,
    primary,
    hits,
  };
}

async function jumpToAtlasRegion(volId, regionId) {
  const idx = getVolumeIndexById(volId);
  if (idx < 0 || !state.nv) return false;
  const vol = state.nv.volumes[idx];
  const meta = state.layerMeta.get(volId);
  if (!meta?.regions) {
    meta.regions = computeClientAtlasRegions(vol, meta?.lut || state.dktLut);
  }
  const region = meta?.regions?.find((r) => Number(r.id) === Number(regionId));
  if (!region?.centroidRAS) return false;

  const centroid = region.centroidRAS;
  state.lastCrosshairRAS = [...centroid];
  if (typeof state.nv.mm2frac === "function" && state.nv.scene?.crosshairPos) {
    const frac = state.nv.mm2frac(centroid);
    if (frac && frac.length >= 3) {
      state.nv.scene.crosshairPos = [frac[0], frac[1], frac[2]];
    }
  }
  state.nv.drawScene?.();
  updateCrosshairReadout({ mm: centroid });
  showToast(`Jumped to ${region.shortName || region.name} (${meta.atlasShortName || "Atlas"} #${region.id}) at RAS [${centroid.map((v) => v.toFixed(1)).join(", ")}]`, 2600);
  return true;
}

async function setAtlasVisibleRegions(volId, regionIdSetOrNull) {
  const idx = getVolumeIndexById(volId);
  if (idx < 0) return;
  const vol = state.nv.volumes[idx];
  const meta = state.layerMeta.get(volId);
  if (!meta) return;
  meta.visibleRegionIds = regionIdSetOrNull instanceof Set ? new Set(regionIdSetOrNull) : null;
  await applyAtlasColormapToVolume(vol);
  state.nv?.updateGLVolume?.();
  state.nv?.drawScene?.();
  syncAtlasStudioUI();
}

async function toggleAtlasRegionVisibility(volId, regionId, checked) {
  const idx = getVolumeIndexById(volId);
  if (idx < 0) return;
  const vol = state.nv.volumes[idx];
  const meta = state.layerMeta.get(volId);
  if (!meta) return;
  if (!meta.regions) {
    meta.regions = computeClientAtlasRegions(vol, meta.lut || state.dktLut);
  }
  const rid = Number(regionId);
  if (!(meta.visibleRegionIds instanceof Set)) {
    // Switching from "all visible" to specific subset
    const allIds = new Set(meta.regions.map((r) => Number(r.id)));
    if (!checked) allIds.delete(rid);
    meta.visibleRegionIds = checked ? null : allIds;
  } else {
    if (checked) meta.visibleRegionIds.add(rid);
    else meta.visibleRegionIds.delete(rid);
    if (meta.regions.length > 0 && meta.visibleRegionIds.size >= meta.regions.length) {
      meta.visibleRegionIds = null;
    }
  }
  await applyAtlasColormapToVolume(vol);
  state.nv?.updateGLVolume?.();
  state.nv?.drawScene?.();
  syncAtlasStudioUI();
}

async function copySelectedAtlasRoisToDrawingLayer(volId = null, explicitIds = null) {
  const targetVol = volId ? state.nv?.volumes?.[getVolumeIndexById(volId)] : getDktVolume();
  if (!targetVol) {
    showToast("Load an atlas volume first.");
    return null;
  }
  const meta = state.layerMeta.get(targetVol.id);
  const rawArr = meta?.rawAtlasVoxels || targetVol.img;
  if (!rawArr || !targetVol.hdr?.affine) return null;

  let targetIds = null;
  let singleRegionInfo = null;
  if (Array.isArray(explicitIds) && explicitIds.length > 0) {
    targetIds = new Set(explicitIds.map(Number));
  } else if (meta?.visibleRegionIds instanceof Set && meta.visibleRegionIds.size > 0) {
    targetIds = new Set(meta.visibleRegionIds);
  } else {
    const hit = lookupAtlasRegionAtRAS(targetVol, state.lastCrosshairRAS);
    if (hit && hit.labelId > 0) {
      targetIds = new Set([hit.labelId]);
      singleRegionInfo = hit;
    }
  }

  if (!targetIds || targetIds.size === 0) {
    showToast("Isolate/check ROI(s) in Atlas Studio or move crosshair onto an atlas region first.");
    return null;
  }

  const dims = getVolDims(targetVol);
  const nx = dims[1] || 0;
  const ny = dims[2] || 0;
  const nz = dims[3] || 0;
  const vox = new Uint8Array(nx * ny * nz);
  let count = 0;
  for (let i = 0; i < vox.length; i++) {
    const val = Math.round(rawArr[i]);
    if (targetIds.has(val)) {
      vox[i] = 1;
      count++;
    }
  }
  if (count === 0) {
    showToast("Selected ROI(s) have 0 voxels on this volume grid.");
    return null;
  }

  const firstId = Array.from(targetIds)[0];
  const lutEntry = (meta?.lut || state.dktLut || {})[String(firstId)];
  const atlasShort = meta?.atlasShortName || "Atlas";
  const layerName =
    targetIds.size === 1
      ? `${atlasShort}_${(singleRegionInfo?.shortName || lutEntry?.shortName || lutEntry?.name || `ROI_${firstId}`).replace(/[^a-zA-Z0-9_-]+/g, "_")}`
      : `${atlasShort}_${targetIds.size}ROIs`;
  const colorHex = singleRegionInfo?.hex || lutEntry?.hex || "#f43f5e";

  const created = await createDrawingLayer({
    name: layerName,
    mode: "mask",
    colorHex,
    opacity: 0.65,
    visible: true,
    voxels: vox,
    explicitDims: [...dims],
    explicitAffine: cloneMat4(targetVol.hdr.affine),
  });
  showToast(`Copied ${targetIds.size} ROI(s) (${count.toLocaleString()} vox) to Drawing Layer "${layerName}" (ready to save as .voi / .nii.gz)!`, 4500);
  return created;
}

function syncDktOverlayControlsUI() {
  const chk = document.getElementById("chkDktColorOverlay");
  const slider = document.getElementById("sliderDktOverlayOpacity");
  const pctEl = document.getElementById("valDktOverlayOpacity");
  const minInput = document.getElementById("inputDktCalMin");
  const maxInput = document.getElementById("inputDktCalMax");
  const autoBtn = document.getElementById("btnDktAutoWindow");
  const dktVol = getDktVolume();
  const defaultMax = getAtlasDefaultMaxLabel(dktVol);
  syncOverlayStyleButtonsUI();

  if (!dktVol) {
    if (chk) {
      chk.checked = false;
      chk.disabled = true;
    }
    if (slider) {
      slider.value = "48";
      slider.disabled = true;
    }
    if (pctEl) pctEl.textContent = "48%";
    if (minInput) {
      minInput.value = "0.5";
      minInput.disabled = true;
    }
    if (maxInput) {
      maxInput.value = defaultMax.toFixed(1);
      maxInput.disabled = true;
    }
    if (autoBtn) {
      autoBtn.disabled = true;
    }
    return;
  }

  const meta = state.layerMeta.get(dktVol.id);
  const isOverlayOn = Boolean(meta ? meta.visible : dktVol.opacity > 0);
  const savedAlpha = meta?.userAtlasOpacity > 0 ? meta.userAtlasOpacity : meta?.savedOpacity > 0 ? meta.savedOpacity : 0.48;
  const activeAlpha = isOverlayOn ? (meta?.userAtlasOpacity > 0 ? meta.userAtlasOpacity : dktVol.opacity) : 0;

  if (chk) {
    chk.disabled = false;
    chk.checked = isOverlayOn;
  }
  if (slider) {
    slider.disabled = false;
    slider.value = String(Math.round((isOverlayOn ? activeAlpha : savedAlpha) * 100));
  }
  if (pctEl) {
    pctEl.textContent = isOverlayOn ? `${Math.round(activeAlpha * 100)}%` : "OFF";
  }
  if (minInput) {
    minInput.disabled = false;
    const calMinVal = dktVol.cal_min !== undefined ? Number(dktVol.cal_min).toFixed(1) : "0.5";
    if (document.activeElement !== minInput) {
      minInput.value = calMinVal;
    }
  }
  if (maxInput) {
    maxInput.disabled = false;
    const calMaxVal = dktVol.cal_max !== undefined ? Number(dktVol.cal_max).toFixed(1) : defaultMax.toFixed(1);
    if (document.activeElement !== maxInput) {
      maxInput.value = calMaxVal;
    }
  }
  if (autoBtn) {
    autoBtn.disabled = false;
    autoBtn.title = `Reset ${getAtlasShortTitle(dktVol)} window to full brain label range (0.5 – ${defaultMax})`;
  }
}

function resetDktInspectorUI() {
  const swatch = document.getElementById("dktColorSwatch");
  const titleEl = document.getElementById("dktRegionTitle");
  const subEl = document.getElementById("dktRegionSub");
  const tooltip = document.getElementById("dktHoverTooltip");
  const hudPill = document.getElementById("dktHudPill");
  const atlasTitle = getAtlasShortTitle();

  if (swatch) swatch.style.background = "#475569";
  if (titleEl) titleEl.textContent = `${atlasTitle}: Not Loaded`;
  if (subEl) subEl.textContent = `Generate or load an atlas (DKT, THOMAS, Tian, FSL), then hover over brain`;
  if (tooltip) tooltip.hidden = true;
  if (hudPill) hudPill.hidden = true;
  syncDktOverlayControlsUI();
}

function updateDktReadoutUI(dktInfo, cursorEvent = null) {
  const swatch = document.getElementById("dktColorSwatch");
  const titleEl = document.getElementById("dktRegionTitle");
  const subEl = document.getElementById("dktRegionSub");
  const tooltip = document.getElementById("dktHoverTooltip");
  const hudPill = document.getElementById("dktHudPill");
  const wrapper = document.getElementById("canvasWrapper");

  if (!dktInfo) {
    resetDktInspectorUI();
    return;
  }

  const hits = Array.isArray(dktInfo.hits) ? dktInfo.hits : dktInfo.labelId > 0 && dktInfo.entry ? [
    {
      atlasTitle: dktInfo.atlasTitle || getAtlasShortTitle(getDktVolume()),
      atlasBadge: (dktInfo.atlasTitle || "DKT").toUpperCase(),
      labelId: dktInfo.labelId,
      name: dktInfo.entry.name || dktInfo.entry.title,
      shortName: dktInfo.entry.shortName || dktInfo.entry.title || dktInfo.entry.name,
      title: dktInfo.entry.title || dktInfo.entry.name,
      hex: dktInfo.entry.hex || "#38bdf8",
    },
  ] : [];

  const ras = dktInfo.ras;
  const rasStr = ras ? `RAS [${ras[0].toFixed(1)}, ${ras[1].toFixed(1)}, ${ras[2].toFixed(1)}] mm` : "";
  const activeAtlasTitle = dktInfo.atlasTitle || getAtlasShortTitle(getDktVolume());

  if (hits.length > 0) {
    const primary = hits[0];
    if (swatch) swatch.style.background = primary.hex || "#38bdf8";
    if (titleEl) titleEl.textContent = primary.title || primary.name;
    if (subEl) {
      const stackedText = hits
        .map((h) => `${h.atlasBadge}: ${h.shortName || h.title} (#${h.labelId})`)
        .join(" · ");
      subEl.textContent = `${stackedText} · ${rasStr}`;
    }

    if (hudPill) {
      hudPill.hidden = false;
      hudPill.innerHTML = hits
        .slice(0, 3)
        .map(
          (h, idx) => `
            <span class="dkt-swatch" style="background:${sanitizeCssHex(h.hex)};"></span>
            <span style="${idx === 0 ? "font-weight:700;" : "color:var(--text-secondary);"}">${hits.length > 1 ? `${escapeDrawHtml(h.atlasBadge)}: ` : ""}${escapeDrawHtml(h.shortName || h.title)}</span>
            <span style="color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">#${Number(h.labelId)}</span>
          `
        )
        .join(`<span style="color:var(--text-muted);margin:0 2px;">|</span>`);
    }

    if (tooltip && cursorEvent && wrapper) {
      const rect = wrapper.getBoundingClientRect();
      const relX = cursorEvent.clientX - rect.left;
      const relY = cursorEvent.clientY - rect.top;
      const estH = 38 + hits.length * 18;
      const maxX = Math.max(10, rect.width - 315);
      const maxY = Math.max(10, rect.height - estH);
      const tx = Math.min(maxX, Math.max(10, relX + 16));
      const ty = Math.min(maxY, Math.max(10, relY + 16));

      tooltip.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:4px;min-width:0;width:100%;">
          ${hits
            .map(
              (h) => `
                <div style="display:flex;align-items:center;gap:6px;min-width:0;">
                  <span class="dkt-swatch" style="background:${sanitizeCssHex(h.hex)};width:10px;height:10px;"></span>
                  <span class="atlas-stacked-badge">${escapeDrawHtml(h.atlasBadge)}</span>
                  <span class="dkt-region-title" style="flex:1;">${escapeDrawHtml(h.shortName || h.title)}</span>
                  <span style="color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">#${Number(h.labelId)}</span>
                </div>
              `
            )
            .join("")}
          <div class="dkt-region-sub">${escapeDrawHtml(rasStr)}</div>
        </div>
      `;
      tooltip.style.transform = `translate3d(${Math.round(tx)}px, ${Math.round(ty)}px, 0)`;
      tooltip.hidden = false;
    }
  } else {
    if (swatch) swatch.style.background = "#334155";
    if (titleEl) titleEl.textContent = "Outside Atlas Labels (Background)";
    if (subEl) subEl.textContent = `${activeAtlasTitle} #0 · ${rasStr}`;
    if (tooltip) tooltip.hidden = true;
    if (hudPill) {
      hudPill.hidden = false;
      const loadedCnt = getLoadedAtlasVolumes().length;
      hudPill.innerHTML = `
        <span class="dkt-swatch" style="background:#475569;"></span>
        <span style="color:var(--text-secondary);">${loadedCnt > 1 ? `${loadedCnt} Atlases Loaded` : `${escapeDrawHtml(activeAtlasTitle)} Active`} — Hover over brain</span>
      `;
    }
  }
}

// Convert mouse hover position on NiiVue canvas -> 3D RAS world coordinate -> stacked multi-atlas lookup
function handleCanvasMouseHover(e) {
  if (!state.nv) return;
  if (getLoadedAtlasVolumes().length === 0) return;

  if (state.zoom2d.isDraggingPan || state.drawing?.isStroking) {
    const tooltip = document.getElementById("dktHoverTooltip");
    if (tooltip) tooltip.hidden = true;
    return;
  }

  const canvas = document.getElementById("niivueCanvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  const cssX = e.clientX - rect.left;
  const cssY = e.clientY - rect.top;
  const scaleX = (canvas.width || rect.width) / rect.width;
  const scaleY = (canvas.height || rect.height) / rect.height;

  let frac = null;
  if (typeof state.nv.canvasPos2frac === "function") {
    frac = state.nv.canvasPos2frac([cssX * scaleX, cssY * scaleY]);
    if ((!frac || frac[0] < 0) && Math.abs(scaleX - 1) > 0.05) {
      frac = state.nv.canvasPos2frac([cssX, cssY]);
    }
  }

  if (!frac || frac[0] < 0 || frac[1] < 0 || frac[2] < 0) {
    const tooltip = document.getElementById("dktHoverTooltip");
    if (tooltip) tooltip.hidden = true;
    return;
  }

  const mm = state.nv.frac2mm(frac);
  if (!mm) return;
  const ras = [Number(mm[0]), Number(mm[1]), Number(mm[2])];

  const badge = document.getElementById("rasCoordsBadge");
  if (badge) {
    badge.textContent = `RAS: ${ras[0].toFixed(1)}, ${ras[1].toFixed(1)}, ${ras[2].toFixed(1)} mm`;
  }

  const dktInfo = lookupDktRegionAtRAS(ras);
  updateDktReadoutUI(dktInfo, e);
}

// Update Crosshair RAS & Voxel Intensity Table across ALL loaded volumes
function updateCrosshairReadout(data) {
  if (!data || !data.mm) return;
  const ras = [Number(data.mm[0]), Number(data.mm[1]), Number(data.mm[2])];
  state.lastCrosshairRAS = ras;

  const badge = document.getElementById("rasCoordsBadge");
  if (badge) {
    badge.textContent = `RAS: ${ras[0].toFixed(1)}, ${ras[1].toFixed(1)}, ${ras[2].toFixed(1)} mm`;
  }

  if (getLoadedAtlasVolumes().length > 0) {
    const dktInfo = lookupDktRegionAtRAS(ras);
    updateDktReadoutUI(dktInfo, null);
  }
  if (typeof renderAtlasStudioCrosshairFooter === "function") {
    renderAtlasStudioCrosshairFooter();
  }

  const tableEl = document.getElementById("voxelReadoutTable");
  if (!tableEl || !state.nv) return;

  tableEl.innerHTML = state.nv.volumes
    .map((vol) => {
      const meta = state.layerMeta.get(vol.id);
      const invAff = invertMat4(vol.hdr.affine);
      const val = sampleVolumeAtRAS(vol, invAff, ras, Boolean(meta?.isMask));
      const vox = applyMat4Pt(invAff, ras).map((v) => Math.round(v));
      if (meta?.isAtlas || meta?.isDktAtlas) {
        const hit = lookupAtlasRegionAtRAS(vol, ras);
        const atlasText = hit ? `${hit.shortName || hit.title} (#${hit.labelId})` : `0 (BG)`;
        return `
          <div class="voxel-row">
            <span class="voxel-name" title="${meta?.label || vol.name}">${meta?.label || vol.name}</span>
            <span class="voxel-val" style="color:${hit?.hex || "#94a3b8"};">${atlasText}</span>
          </div>
        `;
      }
      if (meta?.isDrawingLayer) {
        const dLayer = state.drawing?.layers?.find((l) => l.id === meta.drawingLayerId);
        const lid = Math.round(val);
        let drawText = "0 (BG)";
        let drawHex = "#94a3b8";
        if (lid > 0 && dLayer) {
          if (dLayer.mode === "multilabel") {
            const p = dLayer.labelPalette?.[lid - 1];
            drawText = p ? `#${lid} ${p.name}` : `Label #${lid}`;
            drawHex = p?.hex || dLayer.colorHex;
          } else {
            drawText = `Inside (${lid})`;
            drawHex = dLayer.colorHex;
          }
        }
        return `
          <div class="voxel-row">
            <span class="voxel-name" title="${meta?.label || vol.name}">${meta?.label || vol.name}</span>
            <span style="color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">[${vox.join(",")}]</span>
            <span class="voxel-val" style="color:${drawHex};">${drawText}</span>
          </div>
        `;
      }
      return `
        <div class="voxel-row">
          <span class="voxel-name" title="${meta?.label || vol.name}">${meta?.label || vol.name}</span>
          <span style="color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">[${vox.join(",")}]</span>
          <span class="voxel-val">${val.toFixed(2)}</span>
        </div>
      `;
    })
    .join("");
}

// ============================================================================
// 2D Slice Zoom & Pan Subsystem (Ctrl/Cmd+Wheel, Pinch, Shift/Middle/Pan Drag)
// ============================================================================
const IN_PLANE_WORLD_AXES = {
  0: [0, 1], // Axial -> X, Y
  1: [0, 2], // Coronal -> X, Z
  2: [1, 2], // Sagittal -> Y, Z
};

function getCurrentPan2D() {
  const raw = state.nv?.scene?.pan2Dxyzmm;
  if (raw && raw.length >= 4) {
    const z = Number(raw[3]);
    return [
      Number(raw[0]) || 0,
      Number(raw[1]) || 0,
      Number(raw[2]) || 0,
      Number.isFinite(z) && z >= 1 ? z : 1.0,
    ];
  }
  return [
    state.zoom2d.panMM[0] || 0,
    state.zoom2d.panMM[1] || 0,
    state.zoom2d.panMM[2] || 0,
    state.zoom2d.zoom || 1.0,
  ];
}

function getCanvasDevicePos(clientX, clientY) {
  const canvas = document.getElementById("niivueCanvas");
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) return null;
  if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
    return null;
  }
  const cssX = clientX - rect.left;
  const cssY = clientY - rect.top;
  const scaleX = (canvas.width || rect.width) / rect.width;
  const scaleY = (canvas.height || rect.height) / rect.height;
  return {
    canvas,
    rect,
    cssX,
    cssY,
    devX: cssX * scaleX,
    devY: cssY * scaleY,
    scaleX,
    scaleY,
  };
}

function isPointInLtwh(x, y, ltwh) {
  if (!ltwh || ltwh.length < 4) return false;
  const [l, t, w, h] = ltwh;
  if (!w || !h) return false;
  const x0 = w < 0 ? l + w : l;
  const y0 = h < 0 ? t + h : t;
  const aw = Math.abs(w);
  const ah = Math.abs(h);
  return x >= x0 && x <= x0 + aw && y >= y0 && y <= y0 + ah;
}

function get2DSliceTileAtClientPos(clientX, clientY) {
  if (!state.nv) return null;
  const activeSliceType = state.nv.opts?.sliceType ?? state.nv.sliceType;
  if (activeSliceType === SLICE_TYPE.RENDER) return null;
  const pos = getCanvasDevicePos(clientX, clientY);
  if (!pos) return null;

  const tiles = state.nv.screenSlices;
  if (Array.isArray(tiles) && tiles.length > 0) {
    for (const tile of tiles) {
      if (isPointInLtwh(pos.devX, pos.devY, tile.leftTopWidthHeight)) {
        if (tile.axCorSag === 3) return null; // Never hijack 3D Render tile
        return { tile, axCorSag: tile.axCorSag, pos, useDevice: true };
      }
    }
    if (Math.abs(pos.scaleX - 1) > 0.05) {
      for (const tile of tiles) {
        if (isPointInLtwh(pos.cssX, pos.cssY, tile.leftTopWidthHeight)) {
          if (tile.axCorSag === 3) return null;
          return { tile, axCorSag: tile.axCorSag, pos, useDevice: false };
        }
      }
    }
    return null;
  }

  // Fallback for single-slice / mock contexts without populated screenSlices
  return { tile: null, axCorSag: -1, pos, useDevice: true };
}

function sampleCanvasPtRAS(ptX, ptY, altCssX = null, altCssY = null) {
  if (!state.nv || typeof state.nv.canvasPos2frac !== "function" || typeof state.nv.frac2mm !== "function") {
    return null;
  }
  let frac = state.nv.canvasPos2frac([ptX, ptY]);
  if ((!frac || frac[0] < 0) && altCssX !== null && altCssY !== null) {
    frac = state.nv.canvasPos2frac([altCssX, altCssY]);
  }
  if (!frac || frac[0] < 0 || frac[1] < 0 || frac[2] < 0) return null;
  const mm = state.nv.frac2mm(frac);
  if (!mm || mm.length < 3) return null;
  const out = [Number(mm[0]), Number(mm[1]), Number(mm[2])];
  return out.every(Number.isFinite) ? out : null;
}

// Sample a valid point inside a tile and linearly extrapolate RAS to any target point in that tile
function measureLinearRASInTile(tile, targetX, targetY) {
  const ltwh = tile?.leftTopWidthHeight;
  if (!ltwh || ltwh.length < 4) return null;
  const [l, t, w, h] = ltwh;
  if (w < 8 || h < 8) return null;

  const anchorCandidates = [
    [targetX, targetY],
    [l + w * 0.5, t + h * 0.5],
    [l + w * 0.38, t + h * 0.5],
    [l + w * 0.62, t + h * 0.5],
    [l + w * 0.5, t + h * 0.38],
    [l + w * 0.5, t + h * 0.62],
  ];

  let basePt = null;
  let baseMM = null;
  for (const [ax, ay] of anchorCandidates) {
    const mm = sampleCanvasPtRAS(ax, ay);
    if (mm) {
      basePt = [ax, ay];
      baseMM = mm;
      break;
    }
  }
  if (!baseMM) return null;
  if (Math.hypot(targetX - basePt[0], targetY - basePt[1]) < 0.5) {
    return { mm: baseMM, jacX: null, jacY: null };
  }

  const step = Math.min(20, Math.max(6, Math.min(w, h) * 0.12));
  const dxSign = basePt[0] + step <= l + w - 2 ? 1 : -1;
  const dySign = basePt[1] + step <= t + h - 2 ? 1 : -1;
  const mmX = sampleCanvasPtRAS(basePt[0] + dxSign * step, basePt[1]);
  const mmY = sampleCanvasPtRAS(basePt[0], basePt[1] + dySign * step);

  if (!mmX || !mmY) return null;
  const jacX = [
    (mmX[0] - baseMM[0]) / (dxSign * step),
    (mmX[1] - baseMM[1]) / (dxSign * step),
    (mmX[2] - baseMM[2]) / (dxSign * step),
  ];
  const jacY = [
    (mmY[0] - baseMM[0]) / (dySign * step),
    (mmY[1] - baseMM[1]) / (dySign * step),
    (mmY[2] - baseMM[2]) / (dySign * step),
  ];
  const dx = targetX - basePt[0];
  const dy = targetY - basePt[1];
  return {
    mm: [
      baseMM[0] + jacX[0] * dx + jacY[0] * dy,
      baseMM[1] + jacX[1] * dx + jacY[1] * dy,
      baseMM[2] + jacX[2] * dx + jacY[2] * dy,
    ],
    jacX,
    jacY,
  };
}

// Measure constrained world RAS coordinates at either a specific tile cursor point or across all 2D tile centers
function measureConstrainedWorldRAS(cursorTileInfo = null) {
  if (!state.nv) return null;
  const sums = [0, 0, 0];
  const counts = [0, 0, 0];

  if (cursorTileInfo?.tile) {
    const t = cursorTileInfo.tile;
    const x = cursorTileInfo.useDevice ? cursorTileInfo.pos.devX : cursorTileInfo.pos.cssX;
    const y = cursorTileInfo.useDevice ? cursorTileInfo.pos.devY : cursorTileInfo.pos.cssY;
    const res = measureLinearRASInTile(t, x, y);
    if (res) {
      const axes = IN_PLANE_WORLD_AXES[t.axCorSag] || [0, 1, 2];
      axes.forEach((ax) => {
        sums[ax] += res.mm[ax];
        counts[ax] += 1;
      });
    }
  } else {
    const tiles = state.nv.screenSlices || [];
    for (const t of tiles) {
      if (!t || t.axCorSag < 0 || t.axCorSag > 2 || !t.leftTopWidthHeight) continue;
      const [l, top, w, h] = t.leftTopWidthHeight;
      const res = measureLinearRASInTile(t, l + w * 0.5, top + h * 0.5);
      if (!res) continue;
      const axes = IN_PLANE_WORLD_AXES[t.axCorSag] || [0, 1];
      axes.forEach((ax) => {
        sums[ax] += res.mm[ax];
        counts[ax] += 1;
      });
    }
  }

  return {
    mm: counts.map((c, idx) => (c > 0 ? sums[idx] / c : null)),
    counts,
  };
}

function setNvPanRaw(panVec4) {
  if (!state.nv) return;
  if (typeof state.nv.setPan2Dxyzmm === "function") {
    state.nv.setPan2Dxyzmm([...panVec4]);
  } else if (state.nv.scene) {
    state.nv.scene.pan2Dxyzmm = [...panVec4];
    state.nv.drawScene?.();
  }
}

// Calibrate d(sampledWorldMM_i)/d(pan2Dxyzmm_i) once using NiiVue's live slice transform
function getOrCalibratePanSlope(zoom, cursorTileInfo = null) {
  const z = Math.max(1.0, zoom || 1.0);
  if (state.zoom2d.panCalib) {
    const { baseSlope, scalesByInvZoom } = state.zoom2d.panCalib;
    return scalesByInvZoom ? baseSlope.map((s) => s / z) : [...baseSlope];
  }

  const fallback = [-1 / z, -1 / z, -1 / z];
  if (!state.nv || !state.nv.volumes?.length || !Array.isArray(state.nv.screenSlices) || state.nv.screenSlices.length === 0) {
    return fallback;
  }

  const saved = getCurrentPan2D();
  const z1 = Math.max(1.25, z);
  setNvPanRaw([0, 0, 0, z1]);
  const mA = measureConstrainedWorldRAS(null);
  setNvPanRaw([2, 2, 2, z1]);
  const mB = measureConstrainedWorldRAS(null);

  if (!mA || !mB) {
    setNvPanRaw(saved);
    return fallback;
  }

  const slope1 = [0, 1, 2].map((ax) => {
    if (mA.mm[ax] !== null && mB.mm[ax] !== null) {
      const s = (mB.mm[ax] - mA.mm[ax]) / 2.0;
      if (Math.abs(s) > 1e-4) return s;
    }
    return null;
  });

  // Check whether sensitivity scales inversely with zoom (1/z) or stays constant
  let scalesByInvZoom = true;
  let validPairIdx = slope1.findIndex((s) => s !== null);
  if (validPairIdx >= 0) {
    const z2 = z1 * 2.0;
    setNvPanRaw([0, 0, 0, z2]);
    const mA2 = measureConstrainedWorldRAS(null);
    setNvPanRaw([2, 2, 2, z2]);
    const mB2 = measureConstrainedWorldRAS(null);
    if (mA2?.mm[validPairIdx] !== null && mB2?.mm[validPairIdx] !== null) {
      const s2 = (mB2.mm[validPairIdx] - mA2.mm[validPairIdx]) / 2.0;
      if (Math.abs(s2) > 1e-4) {
        const ratio = Math.abs(slope1[validPairIdx] / s2);
        scalesByInvZoom = ratio > 1.35;
      }
    }
  }

  // Restore original pan before returning
  setNvPanRaw(saved);

  const observedSign = validPairIdx >= 0 ? Math.sign(slope1[validPairIdx]) || -1 : -1;
  const baseSlope = [0, 1, 2].map((ax) => {
    const measured = slope1[ax];
    if (measured !== null) {
      return scalesByInvZoom ? measured * z1 : measured;
    }
    return scalesByInvZoom ? observedSign : observedSign / z1;
  });

  if (slope1.every((s) => s !== null)) {
    state.zoom2d.panCalib = { baseSlope, scalesByInvZoom };
  }
  return scalesByInvZoom ? baseSlope.map((s) => s / z) : [...baseSlope];
}

function applyPan2Dxyzmm(panVec4) {
  const zRaw = Number(panVec4?.[3]);
  const zoom = Math.max(1.0, Math.min(10.0, Number.isFinite(zRaw) ? zRaw : 1.0));
  const px = Math.max(-400, Math.min(400, Number(panVec4?.[0]) || 0));
  const py = Math.max(-400, Math.min(400, Number(panVec4?.[1]) || 0));
  const pz = Math.max(-400, Math.min(400, Number(panVec4?.[2]) || 0));
  const nextVec = [px, py, pz, Number(zoom.toFixed(4))];

  state.zoom2d.panMM = [nextVec[0], nextVec[1], nextVec[2]];
  state.zoom2d.zoom = nextVec[3];

  setNvPanRaw(nextVec);
  syncZoom2dUI();
}

function set2DZoom(targetZoom, focusRAS = null, cursorClientPos = null) {
  const newZoom = Math.max(1.0, Math.min(10.0, Number(targetZoom) || 1.0));
  if (newZoom <= 1.001) {
    applyPan2Dxyzmm([0, 0, 0, 1.0]);
    return;
  }

  const curPan = getCurrentPan2D();
  if (!state.nv || !state.nv.volumes?.length) {
    applyPan2Dxyzmm([0, 0, 0, newZoom]);
    return;
  }

  const cursorTileInfo = cursorClientPos
    ? get2DSliceTileAtClientPos(cursorClientPos.x, cursorClientPos.y)
    : null;
  const targetFocus =
    Array.isArray(focusRAS) && focusRAS.length >= 3 && focusRAS.every(Number.isFinite)
      ? focusRAS
      : state.lastCrosshairRAS;

  // Set provisional pan, then apply closed-form world-RAS anchor correction
  const oldSlope = getOrCalibratePanSlope(curPan[3] || 1.0, cursorTileInfo);
  const slope = getOrCalibratePanSlope(newZoom, cursorTileInfo);
  setNvPanRaw([curPan[0], curPan[1], curPan[2], newZoom]);

  const measured = measureConstrainedWorldRAS(cursorTileInfo);
  if (measured && targetFocus) {
    const nextPan = [curPan[0], curPan[1], curPan[2]];
    for (let ax = 0; ax < 3; ax++) {
      if (measured.mm[ax] !== null && Math.abs(slope[ax]) > 1e-5) {
        const errMM = targetFocus[ax] - measured.mm[ax];
        nextPan[ax] = curPan[ax] + errMM / slope[ax];
      } else if (Math.abs(slope[ax]) > 1e-5) {
        // Keep unconstrained axis (e.g. orthogonal axis during wheel zoom on one MPR tile) anchored without jump
        nextPan[ax] = curPan[ax] * (oldSlope[ax] / slope[ax]);
      }
    }
    applyPan2Dxyzmm([nextPan[0], nextPan[1], nextPan[2], newZoom]);
    return;
  }

  applyPan2Dxyzmm([curPan[0], curPan[1], curPan[2], newZoom]);
}

function step2DZoom(multiplier) {
  const curZoom = getCurrentPan2D()[3] || 1.0;
  let nextZoom = curZoom * multiplier;
  if (nextZoom < 1.03 && multiplier < 1) {
    nextZoom = 1.0;
  }
  set2DZoom(nextZoom, state.lastCrosshairRAS, null);
}

function reset2DZoom() {
  state.zoom2d.panMM = [0, 0, 0];
  state.zoom2d.zoom = 1.0;
  setNvPanRaw([0, 0, 0, 1.0]);
  syncZoom2dUI();
}

function togglePan2dMode(forceState = null) {
  const nextActive = forceState !== null ? Boolean(forceState) : !state.zoom2d.panModeActive;
  if (nextActive && state.drawing?.paintEnabled && typeof setDrawingPaintEnabled === "function") {
    setDrawingPaintEnabled(false);
    showToast("Paused Paint Mode while 2D Pan Mode is active.", 2200);
  }
  state.zoom2d.panModeActive = nextActive;

  const panBtn = document.getElementById("btnTogglePan2dMode");
  if (panBtn) {
    panBtn.classList.toggle("active", nextActive);
    panBtn.setAttribute("aria-pressed", String(nextActive));
  }

  const canvas = document.getElementById("niivueCanvas");
  if (canvas) {
    canvas.classList.toggle("pan-mode-active", nextActive);
    if (!nextActive) canvas.classList.remove("is-panning");
  }

  if (state.nv?.opts) {
    state.nv.opts.dragMode = nextActive ? DRAG_MODE.pan : DRAG_MODE.contrast;
  }
  syncZoom2dUI();
}

function syncZoom2dUI() {
  const hud = document.getElementById("zoom2dHud");
  const select = document.getElementById("selectZoom2dLevel");
  const zoomInBtn = document.getElementById("btnZoom2dIn");
  const zoomOutBtn = document.getElementById("btnZoom2dOut");
  const resetBtn = document.getElementById("btnZoom2dReset");
  const panBtn = document.getElementById("btnTogglePan2dMode");

  const [px, py, pz, zoom] = getCurrentPan2D();
  const isAtDefault = zoom <= 1.001 && Math.hypot(px, py, pz) < 0.25;
  const activeSliceType = state.nv?.opts?.sliceType ?? state.nv?.sliceType;
  const isRenderOnly = activeSliceType === SLICE_TYPE.RENDER;

  if (hud) {
    hud.classList.toggle("is-3d-only", Boolean(isRenderOnly));
  }
  if (panBtn) {
    panBtn.classList.toggle("active", state.zoom2d.panModeActive);
    panBtn.setAttribute("aria-pressed", String(state.zoom2d.panModeActive));
  }
  if (zoomOutBtn) {
    zoomOutBtn.disabled = zoom <= 1.001;
  }
  if (zoomInBtn) {
    zoomInBtn.disabled = zoom >= 9.99;
  }
  if (resetBtn) {
    resetBtn.disabled = isAtDefault && !state.zoom2d.panModeActive;
  }

  if (select) {
    // Match an existing preset option (within 1%) or maintain a single dynamic custom option
    let customOpt = select.querySelector('option[data-custom-zoom="true"]');
    const presetOpts = Array.from(select.options).filter((o) => !o.hasAttribute("data-custom-zoom"));
    const matched = presetOpts.find((o) => Math.abs(parseFloat(o.value) - zoom) < 0.015);
    if (matched) {
      if (customOpt) customOpt.remove();
      select.value = matched.value;
    } else {
      const pctLabel = `${Math.round(zoom * 100)}%`;
      if (!customOpt) {
        customOpt = document.createElement("option");
        customOpt.setAttribute("data-custom-zoom", "true");
        select.appendChild(customOpt);
      }
      customOpt.value = String(zoom);
      customOpt.textContent = pctLabel;
      select.value = String(zoom);
    }
  }
}

function wire2DZoomAndPanListeners() {
  const nvCanvas = document.getElementById("niivueCanvas");

  document.getElementById("btnTogglePan2dMode")?.addEventListener("click", () => {
    togglePan2dMode();
  });
  document.getElementById("btnZoom2dIn")?.addEventListener("click", () => {
    step2DZoom(1.25);
  });
  document.getElementById("btnZoom2dOut")?.addEventListener("click", () => {
    step2DZoom(0.8);
  });
  document.getElementById("btnZoom2dReset")?.addEventListener("click", () => {
    reset2DZoom();
    if (state.zoom2d.panModeActive) {
      togglePan2dMode(false);
    }
  });
  document.getElementById("selectZoom2dLevel")?.addEventListener("change", (e) => {
    const val = parseFloat(e.target.value);
    if (Number.isFinite(val)) {
      set2DZoom(val, state.lastCrosshairRAS, null);
    }
  });

  if (!nvCanvas) return;

  // Ctrl/Cmd+Scroll (and browser trackpad pinch, which emits ctrlKey + wheel) -> 2D Zoom
  // Plain Scroll passes through untouched to NiiVue slice stepping
  nvCanvas.addEventListener(
    "wheel",
    (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (!state.nv || !state.nv.volumes?.length) return;
      const tileInfo = get2DSliceTileAtClientPos(e.clientX, e.clientY);
      if (!tileInfo) return; // Over 3D render tile or in 3D-only mode -> keep native 3D zoom

      e.preventDefault();
      e.stopImmediatePropagation();

      const curZoom = getCurrentPan2D()[3] || 1.0;
      const clampedDelta = Math.max(-120, Math.min(120, e.deltaY));
      const factor = Math.exp(-clampedDelta * 0.0045);
      const nextZoom = Math.max(1.0, Math.min(10.0, curZoom * factor));

      const xPt = tileInfo.useDevice ? tileInfo.pos.devX : tileInfo.pos.cssX;
      const yPt = tileInfo.useDevice ? tileInfo.pos.devY : tileInfo.pos.cssY;
      const cursorRAS =
        sampleCanvasPtRAS(tileInfo.pos.devX, tileInfo.pos.devY, tileInfo.pos.cssX, tileInfo.pos.cssY) ||
        measureLinearRASInTile(tileInfo.tile, xPt, yPt)?.mm ||
        state.lastCrosshairRAS;

      set2DZoom(nextZoom, cursorRAS, { x: e.clientX, y: e.clientY });
    },
    { capture: true, passive: false }
  );

  // 1:1 World-mm 2D Pan Drag (Pan Mode + Left Drag, Shift + Left Drag, or Middle-Click Drag) over 2D tiles
  let activePanDrag = null;

  nvCanvas.addEventListener(
    "mousedown",
    (e) => {
      if (!state.nv || !state.nv.volumes?.length) return;
      const wantsPan = e.button === 1 || (e.button === 0 && (e.shiftKey || state.zoom2d.panModeActive));
      if (!wantsPan) return;

      const tileInfo = get2DSliceTileAtClientPos(e.clientX, e.clientY);
      if (!tileInfo) return; // Over 3D tile -> allow NiiVue 3D camera rotation

      e.preventDefault();
      e.stopImmediatePropagation();

      const startPan = getCurrentPan2D();
      const slope = getOrCalibratePanSlope(startPan[3], tileInfo);
      let jacX = null;
      let jacY = null;
      if (tileInfo.tile) {
        const ptX = tileInfo.useDevice ? tileInfo.pos.devX : tileInfo.pos.cssX;
        const ptY = tileInfo.useDevice ? tileInfo.pos.devY : tileInfo.pos.cssY;
        const lin = measureLinearRASInTile(tileInfo.tile, ptX, ptY);
        if (lin?.jacX && lin?.jacY) {
          // Convert Jacobian from tile coordinate units to CSS px
          const coordScaleX = tileInfo.useDevice ? tileInfo.pos.scaleX : 1;
          const coordScaleY = tileInfo.useDevice ? tileInfo.pos.scaleY : 1;
          jacX = lin.jacX.map((v) => v * coordScaleX);
          jacY = lin.jacY.map((v) => v * coordScaleY);
        }
      }

      activePanDrag = {
        startClientX: e.clientX,
        startClientY: e.clientY,
        startPan,
        slope,
        axes: tileInfo.tile ? IN_PLANE_WORLD_AXES[tileInfo.axCorSag] || [0, 1] : [0, 1],
        jacX,
        jacY,
      };
      state.zoom2d.isDraggingPan = true;
      nvCanvas.classList.add("is-panning");
      const tooltip = document.getElementById("dktHoverTooltip");
      if (tooltip) tooltip.hidden = true;
    },
    { capture: true }
  );

  window.addEventListener("mousemove", (e) => {
    if (!activePanDrag) return;
    const dx = e.clientX - activePanDrag.startClientX;
    const dy = e.clientY - activePanDrag.startClientY;
    const nextPan = [...activePanDrag.startPan];

    if (activePanDrag.jacX && activePanDrag.jacY) {
      for (const ax of activePanDrag.axes) {
        const s = activePanDrag.slope[ax];
        if (Math.abs(s) > 1e-5) {
          const dWorldMM = activePanDrag.jacX[ax] * dx + activePanDrag.jacY[ax] * dy;
          nextPan[ax] = activePanDrag.startPan[ax] - dWorldMM / s;
        }
      }
    } else {
      const mmPerPx = 0.5 / Math.max(1, activePanDrag.startPan[3] || 1);
      const ax0 = activePanDrag.axes[0] ?? 0;
      const ax1 = activePanDrag.axes[1] ?? 1;
      nextPan[ax0] = activePanDrag.startPan[ax0] - (dx * mmPerPx) / (activePanDrag.slope[ax0] || -1);
      nextPan[ax1] = activePanDrag.startPan[ax1] + (dy * mmPerPx) / (activePanDrag.slope[ax1] || -1);
    }

    applyPan2Dxyzmm(nextPan);
  });

  const endPanDrag = () => {
    if (!activePanDrag) return;
    activePanDrag = null;
    state.zoom2d.isDraggingPan = false;
    nvCanvas.classList.remove("is-panning");
  };
  window.addEventListener("mouseup", endPanDrag);
  window.addEventListener("blur", endPanDrag);
}

// ============================================================================
// Shared Non-Blocking Floating Pop-Out Window Framework
// ============================================================================
function normalizePopoutZIndices() {
  const ordered = Array.from(state.popouts.values()).sort(
    (a, b) => (parseInt(a.el.style.zIndex || "1205", 10)) - (parseInt(b.el.style.zIndex || "1205", 10))
  );
  let z = 1205;
  ordered.forEach((p) => {
    z += 2;
    p.el.style.zIndex = String(z);
  });
  state.popoutTopZ = z;
}

function getTopVisiblePopoutId() {
  let topId = null;
  let topZ = -1;
  state.popouts.forEach((p, id) => {
    if (!p.el.hidden && !p.isMinimized) {
      const z = parseInt(p.el.style.zIndex || "1205", 10);
      if (z >= topZ) {
        topZ = z;
        topId = id;
      }
    }
  });
  return topId;
}

function setPopoutDomContent(container, content) {
  if (!container || content === undefined || content === null) return;
  if (typeof content === "string") {
    container.innerHTML = content;
  } else if (typeof Node !== "undefined" && content instanceof Node) {
    container.innerHTML = "";
    container.appendChild(content);
  }
}

function createPopoutWindow(options = {}) {
  const id = String(options.id || `popout_${state.popouts.size + 1}`);
  const existing = state.popouts.get(id);
  if (existing) {
    if (options.title !== undefined) existing.setTitle(options.title);
    if (options.badge !== undefined) existing.setBadge(options.badge);
    if (options.content !== undefined) existing.setContent(options.content);
    if (options.footer !== undefined) existing.setFooter(options.footer);
    if (options.autoOpen) existing.open();
    return existing;
  }

  const layer = document.getElementById("popoutLayer") || document.body;
  const dock = document.getElementById("popoutDock");

  const minWidth = Math.max(240, Number(options.minWidth) || 320);
  const minHeight = Math.max(160, Number(options.minHeight) || 220);
  const vw = typeof window !== "undefined" && window.innerWidth ? window.innerWidth : 1280;
  const vh = typeof window !== "undefined" && window.innerHeight ? window.innerHeight : 800;

  let curWidth = Math.min(vw - 16, Math.max(minWidth, Number(options.width) || 500));
  let curHeight = Math.min(vh - 16, Math.max(minHeight, Number(options.height) || 440));
  const cascade = (state.popouts.size % 5) * 28;
  let curLeft =
    options.x !== undefined
      ? Number(options.x)
      : Math.max(12, Math.round((vw - curWidth) * 0.5 + cascade));
  let curTop =
    options.y !== undefined
      ? Number(options.y)
      : Math.max(64, Math.round((vh - curHeight) * 0.42 + cascade));

  const el = document.createElement("section");
  el.className = "popout-window";
  el.id = `popout_${id}`;
  el.setAttribute("role", "dialog");
  el.setAttribute("aria-modal", "false");
  el.setAttribute("aria-label", options.title || "Tool Window");
  el.hidden = true;

  const headerEl = document.createElement("header");
  headerEl.className = "popout-header";

  const titleWrap = document.createElement("div");
  titleWrap.className = "popout-title-wrap";

  const iconEl = document.createElement("span");
  iconEl.className = "popout-title-icon";
  iconEl.setAttribute("aria-hidden", "true");
  iconEl.textContent = options.icon || "◆";

  const titleEl = document.createElement("h4");
  titleEl.className = "popout-title";
  titleEl.textContent = options.title || "Floating Module";

  const badgeEl = document.createElement("span");
  badgeEl.className = "popout-badge";
  if (options.badge) {
    badgeEl.textContent = options.badge;
  } else {
    badgeEl.hidden = true;
  }

  titleWrap.appendChild(iconEl);
  titleWrap.appendChild(titleEl);
  titleWrap.appendChild(badgeEl);

  const actionsEl = document.createElement("div");
  actionsEl.className = "popout-header-actions";

  const minBtn = document.createElement("button");
  minBtn.type = "button";
  minBtn.className = "popout-ctrl-btn min-btn";
  minBtn.title = "Minimize to bottom-left dock";
  minBtn.setAttribute("aria-label", "Minimize window");
  minBtn.textContent = "−";

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "popout-ctrl-btn close-btn";
  closeBtn.title = "Close window (Esc)";
  closeBtn.setAttribute("aria-label", "Close window");
  closeBtn.textContent = "✕";

  actionsEl.appendChild(minBtn);
  actionsEl.appendChild(closeBtn);
  headerEl.appendChild(titleWrap);
  headerEl.appendChild(actionsEl);

  const bodyEl = document.createElement("div");
  bodyEl.className = "popout-body";
  setPopoutDomContent(bodyEl, options.content);

  const footerEl = document.createElement("footer");
  footerEl.className = "popout-footer";
  if (options.footer) {
    setPopoutDomContent(footerEl, options.footer);
  } else {
    footerEl.hidden = true;
  }

  const resizeHandle = document.createElement("div");
  resizeHandle.className = "popout-resize-handle";
  resizeHandle.title = "Drag to resize";

  el.appendChild(headerEl);
  el.appendChild(bodyEl);
  el.appendChild(footerEl);
  el.appendChild(resizeHandle);
  layer.appendChild(el);

  const applyGeom = () => {
    el.style.width = `${Math.round(curWidth)}px`;
    el.style.height = `${Math.round(curHeight)}px`;
    el.style.left = `${Math.round(curLeft)}px`;
    el.style.top = `${Math.round(curTop)}px`;
  };

  const handle = {
    id,
    el,
    headerEl,
    bodyEl,
    footerEl,
    titleEl,
    badgeEl,
    dockPillEl: null,
    isMinimized: false,
    isOpen() {
      return !el.hidden && !handle.isMinimized;
    },
    clampToViewport() {
      const curVW = typeof window !== "undefined" && window.innerWidth ? window.innerWidth : 1280;
      const curVH = typeof window !== "undefined" && window.innerHeight ? window.innerHeight : 800;
      curWidth = Math.max(Math.min(minWidth, curVW - 16), Math.min(curWidth, curVW - 16));
      curHeight = Math.max(Math.min(minHeight, curVH - 16), Math.min(curHeight, curVH - 16));
      curLeft = Math.max(8, Math.min(curLeft, Math.max(8, curVW - curWidth - 8)));
      curTop = Math.max(8, Math.min(curTop, Math.max(8, curVH - curHeight - 8)));
      applyGeom();
    },
    bringToFront() {
      if (state.popoutTopZ >= 8000) {
        normalizePopoutZIndices();
      }
      state.popoutTopZ += 2;
      el.style.zIndex = String(state.popoutTopZ);
      state.focusedPopoutId = id;
      state.popouts.forEach((p, pid) => {
        p.el.classList.toggle("is-focused", pid === id);
      });
      options.onFocus?.(handle);
    },
    setTitle(newTitle) {
      titleEl.textContent = newTitle || "Floating Module";
      el.setAttribute("aria-label", titleEl.textContent);
      if (handle.dockPillEl) {
        const lbl = handle.dockPillEl.querySelector(".popout-dock-label");
        if (lbl) lbl.textContent = titleEl.textContent;
      }
    },
    setBadge(text) {
      if (text) {
        badgeEl.textContent = text;
        badgeEl.hidden = false;
      } else {
        badgeEl.textContent = "";
        badgeEl.hidden = true;
      }
    },
    setContent(content) {
      setPopoutDomContent(bodyEl, content);
    },
    setFooter(footer) {
      if (footer) {
        setPopoutDomContent(footerEl, footer);
        footerEl.hidden = false;
      } else {
        footerEl.innerHTML = "";
        footerEl.hidden = true;
      }
    },
    open() {
      handle.isMinimized = false;
      if (handle.dockPillEl) {
        handle.dockPillEl.remove();
        handle.dockPillEl = null;
      }
      el.hidden = false;
      handle.clampToViewport();
      handle.bringToFront();
      options.onOpen?.(handle);
    },
    minimize() {
      handle.isMinimized = true;
      el.hidden = true;
      el.classList.remove("is-focused");
      if (state.focusedPopoutId === id) {
        state.focusedPopoutId = getTopVisiblePopoutId();
        if (state.focusedPopoutId) {
          state.popouts.get(state.focusedPopoutId)?.el.classList.add("is-focused");
        }
      }
      if (dock && !handle.dockPillEl) {
        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = "popout-dock-pill";
        pill.title = `Restore ${titleEl.textContent}`;
        const dot = document.createElement("span");
        dot.className = "popout-dock-dot";
        const lbl = document.createElement("span");
        lbl.className = "popout-dock-label";
        lbl.textContent = titleEl.textContent;
        pill.appendChild(dot);
        pill.appendChild(lbl);
        pill.addEventListener("click", () => handle.restore());
        dock.appendChild(pill);
        handle.dockPillEl = pill;
      }
      options.onMinimize?.(handle);
    },
    restore() {
      handle.open();
      options.onRestore?.(handle);
    },
    close() {
      handle.isMinimized = false;
      el.hidden = true;
      el.classList.remove("is-focused");
      if (handle.dockPillEl) {
        handle.dockPillEl.remove();
        handle.dockPillEl = null;
      }
      if (state.focusedPopoutId === id) {
        state.focusedPopoutId = getTopVisiblePopoutId();
        if (state.focusedPopoutId) {
          state.popouts.get(state.focusedPopoutId)?.el.classList.add("is-focused");
        }
      }
      options.onClose?.(handle);
    },
    toggle() {
      if (handle.isOpen()) {
        handle.close();
      } else {
        handle.open();
      }
    },
    destroy() {
      handle.close();
      el.remove();
      state.popouts.delete(id);
    },
  };

  // Bring to front on any interaction inside the window
  el.addEventListener("pointerdown", () => handle.bringToFront(), { capture: true });
  minBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    handle.minimize();
  });
  closeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    handle.close();
  });

  // Header pointer drag
  headerEl.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest(".popout-header-actions")) return;
    e.preventDefault();
    handle.bringToFront();
    const startX = e.clientX;
    const startY = e.clientY;
    const initLeft = curLeft;
    const initTop = curTop;
    el.classList.add("is-dragging");
    if (typeof headerEl.setPointerCapture === "function") {
      try {
        headerEl.setPointerCapture(e.pointerId);
      } catch (_) {}
    }

    const onMove = (mv) => {
      curLeft = initLeft + (mv.clientX - startX);
      curTop = initTop + (mv.clientY - startY);
      handle.clampToViewport();
    };
    const onUp = () => {
      el.classList.remove("is-dragging");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });

  // Corner resize drag
  resizeHandle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    handle.bringToFront();
    const startX = e.clientX;
    const startY = e.clientY;
    const initW = curWidth;
    const initH = curHeight;
    el.classList.add("is-resizing");
    if (typeof resizeHandle.setPointerCapture === "function") {
      try {
        resizeHandle.setPointerCapture(e.pointerId);
      } catch (_) {}
    }

    const onMove = (mv) => {
      const curVW = window.innerWidth || 1280;
      const curVH = window.innerHeight || 800;
      curWidth = Math.max(minWidth, Math.min(curVW - curLeft - 8, initW + (mv.clientX - startX)));
      curHeight = Math.max(minHeight, Math.min(curVH - curTop - 8, initH + (mv.clientY - startY)));
      applyGeom();
    };
    const onUp = () => {
      el.classList.remove("is-resizing");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });

  handle.clampToViewport();
  state.popouts.set(id, handle);
  if (options.autoOpen) handle.open();
  return handle;
}

function getPopoutWindow(id) {
  return state.popouts.get(String(id)) || null;
}

function closeAllPopouts() {
  state.popouts.forEach((p) => p.close());
}

function clampAllPopoutsToViewport() {
  state.popouts.forEach((p) => {
    if (!p.el.hidden) p.clampToViewport();
  });
}

// ============================================================================
// Phase 2: Drawing Layers, Assisted Segmentation & MRIcroGL .voi / NIfTI Export
// ============================================================================
const DRAW_COLOR_SWATCHES = [
  "#f43f5e",
  "#f59e0b",
  "#10b981",
  "#38bdf8",
  "#a855f7",
  "#ec4899",
  "#eab308",
  "#06b6d4",
];

const DEFAULT_MULTILABEL_PALETTE = [
  { id: 1, name: "Lesion / SOZ", hex: "#f43f5e" },
  { id: 2, name: "Resection Cavity", hex: "#10b981" },
  { id: 3, name: "Eloquent Cortex", hex: "#38bdf8" },
  { id: 4, name: "Perilesional Margin", hex: "#f59e0b" },
  { id: 5, name: "Vascular Landmark", hex: "#a855f7" },
  { id: 6, name: "Artifact / Exclude", hex: "#ec4899" },
];

const DRAW_TOOL_DEFS = [
  { id: "pen", label: "Pen", key: "B", icon: "✏", title: "Freehand 2D/3D Brush (Shortcut: B)" },
  { id: "filled_pen", label: "Filled", key: "F", icon: "⬟", title: "Closed-Contour Filled Pen (Shortcut: F)" },
  { id: "eraser", label: "Eraser", key: "X", icon: "⌫", title: "Eraser (Shortcut: X, or Right-Click-Drag anytime)" },
  { id: "flood_fill", label: "Bucket", key: "G", icon: "🪣", title: "2D Flood Fill inside closed contour/region (Shortcut: G)" },
  { id: "region_grow", label: "Grow", key: "W", icon: "✦", title: "Intensity-Guided Region Grow click-to-segment (Shortcut: W)" },
];

function escapeDrawHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function sanitizeCssHex(hex, fallback = "#38bdf8") {
  const s = String(hex || "").trim();
  return /^#[0-9a-f]{6}$/i.test(s) ? s : fallback;
}

function hexToRgb(hex) {
  const m = String(hex || "").trim().match(/^#?([0-9a-f]{6})$/i);
  if (!m) return [244, 63, 94];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function normalizeVolDims3(dims) {
  if (!Array.isArray(dims)) return [1, 1, 1];
  if (dims.length >= 4) return [Math.max(1, dims[1] | 0), Math.max(1, dims[2] | 0), Math.max(1, dims[3] | 0)];
  if (dims.length >= 3) return [Math.max(1, dims[0] | 0), Math.max(1, dims[1] | 0), Math.max(1, dims[2] | 0)];
  return [1, 1, 1];
}

function stripDrawingFilenameExt(name) {
  return String(name || "drawing")
    .replace(/\.(nii\.gz|nii|voi|mgz|mgh)$/i, "")
    .trim() || "drawing";
}

function buildUint8Nifti1Bytes(nx, ny, nz, affine4x4, voxelsUint8 = null) {
  const totalVoxels = Math.max(1, (nx | 0) * (ny | 0) * (nz | 0));
  const buf = new ArrayBuffer(352 + totalVoxels);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  const aff = Array.isArray(affine4x4) && affine4x4.length >= 3 ? affine4x4 : identityMat4();

  const vx = Math.hypot(aff[0][0] || 0, aff[1][0] || 0, aff[2][0] || 0) || 1;
  const vy = Math.hypot(aff[0][1] || 0, aff[1][1] || 0, aff[2][1] || 0) || 1;
  const vz = Math.hypot(aff[0][2] || 0, aff[1][2] || 0, aff[2][2] || 0) || 1;

  view.setInt32(0, 348, true); // sizeof_hdr
  view.setInt16(40, 3, true); // dim[0]
  view.setInt16(42, nx | 0, true);
  view.setInt16(44, ny | 0, true);
  view.setInt16(46, nz | 0, true);
  view.setInt16(48, 1, true);
  view.setInt16(50, 1, true);
  view.setInt16(52, 1, true);
  view.setInt16(54, 1, true);

  view.setInt16(70, 2, true); // DT_UINT8
  view.setInt16(72, 8, true); // bitpix

  view.setFloat32(76, 1.0, true); // pixdim[0]
  view.setFloat32(80, vx, true);
  view.setFloat32(84, vy, true);
  view.setFloat32(88, vz, true);
  view.setFloat32(92, 1.0, true);
  view.setFloat32(96, 1.0, true);
  view.setFloat32(100, 1.0, true);
  view.setFloat32(104, 1.0, true);

  view.setFloat32(108, 352.0, true); // vox_offset
  view.setFloat32(112, 1.0, true); // scl_slope
  view.setFloat32(116, 0.0, true); // scl_inter
  bytes[123] = 2; // NIFTI_UNITS_MM

  view.setInt16(252, 0, true); // qform_code
  view.setInt16(254, 1, true); // sform_code
  for (let c = 0; c < 4; c++) {
    view.setFloat32(280 + c * 4, Number(aff[0][c]) || 0, true);
    view.setFloat32(296 + c * 4, Number(aff[1][c]) || 0, true);
    view.setFloat32(312 + c * 4, Number(aff[2][c]) || 0, true);
  }
  bytes[344] = 0x6e; // 'n'
  bytes[345] = 0x2b; // '+'
  bytes[346] = 0x31; // '1'
  bytes[347] = 0x00; // '\0'

  if (voxelsUint8 && voxelsUint8.length) {
    const copyLen = Math.min(totalVoxels, voxelsUint8.length);
    bytes.set(voxelsUint8.subarray(0, copyLen), 352);
  }
  return bytes;
}

function resolveSliceVoxelAxes(affine4x4, axCorSag) {
  const aff = Array.isArray(affine4x4) && affine4x4.length >= 3 ? affine4x4 : identityMat4();
  // Target world axis: Axial -> Z (2), Coronal -> Y (1), Sagittal -> X (0)
  let targetWorldAxis = 2;
  if (axCorSag === 1) targetWorldAxis = 1;
  else if (axCorSag === 2) targetWorldAxis = 0;

  let sliceAxis = 2;
  let bestScore = -1;
  for (let col = 0; col < 3; col++) {
    const norm = Math.hypot(aff[0][col] || 0, aff[1][col] || 0, aff[2][col] || 0) || 1;
    const score = Math.abs(aff[targetWorldAxis][col] || 0) / norm;
    if (score > bestScore) {
      bestScore = score;
      sliceAxis = col;
    }
  }
  const otherAxes = [0, 1, 2].filter((a) => a !== sliceAxis);
  return { sliceAxis, uAxis: otherAxes[0], vAxis: otherAxes[1] };
}

function computeLayerStatsData(voxels, affine) {
  const perLabelCounts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
  let voxelCount = 0;
  if (voxels && voxels.length) {
    for (let i = 0; i < voxels.length; i++) {
      const v = voxels[i];
      if (v > 0) {
        voxelCount++;
        if (v >= 1 && v <= 6) perLabelCounts[v]++;
      }
    }
  }
  const voxelMm3 = Math.abs(det3x3(affine || identityMat4())) || 1;
  const volumeCc = (voxelCount * voxelMm3) / 1000.0;
  return { voxelCount, volumeCc, voxelMm3, perLabelCounts };
}

function applySparseDiffToVoxels(voxels, changesInput, newVal = 1, overwrite = true) {
  if (!voxels || !changesInput) return null;
  const indices = [];
  const prev = [];
  const next = [];

  const handleEntry = (idx, targetVal) => {
    if (idx < 0 || idx >= voxels.length) return;
    const oldVal = voxels[idx];
    const clampedNext = Math.max(0, Math.min(255, targetVal | 0));
    if (!overwrite && clampedNext !== 0 && oldVal !== 0) return;
    if (oldVal === clampedNext) return;
    indices.push(idx);
    prev.push(oldVal);
    next.push(clampedNext);
    voxels[idx] = clampedNext;
  };

  if (changesInput instanceof Map) {
    for (const [idx, val] of changesInput.entries()) {
      handleEntry(idx | 0, val);
    }
  } else {
    for (const idx of changesInput) {
      handleEntry(idx | 0, newVal);
    }
  }

  if (indices.length === 0) return null;
  return {
    indices: Uint32Array.from(indices),
    prev: Uint8Array.from(prev),
    next: Uint8Array.from(next),
  };
}

function stampBrushSegmentVoxels(
  voxels,
  dims,
  p0,
  p1,
  value,
  brushSize,
  brush3D,
  sliceAxis,
  overwrite,
  diffMap
) {
  const [nx, ny, nz] = normalizeVolDims3(dims);
  const val = Math.max(0, Math.min(255, value | 0));
  const r = Math.max(0, Math.min(8, Math.round(brushSize || 1) - 1));
  const r2 = r * r + 0.25;
  const sliceStride = nx * ny;
  const ax = sliceAxis === 0 || sliceAxis === 1 || sliceAxis === 2 ? sliceAxis : 2;

  const dx = p1[0] - p0[0];
  const dy = p1[1] - p0[1];
  const dz = p1[2] - p0[2];
  const steps = Math.max(1, Math.ceil(Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) * 2));

  const ozMin = !brush3D && ax === 2 ? 0 : -r;
  const ozMax = !brush3D && ax === 2 ? 0 : r;
  const oyMin = !brush3D && ax === 1 ? 0 : -r;
  const oyMax = !brush3D && ax === 1 ? 0 : r;
  const oxMin = !brush3D && ax === 0 ? 0 : -r;
  const oxMax = !brush3D && ax === 0 ? 0 : r;

  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const cx = Math.round(p0[0] + dx * t);
    const cy = Math.round(p0[1] + dy * t);
    const cz = Math.round(p0[2] + dz * t);

    for (let oz = ozMin; oz <= ozMax; oz++) {
      const z = cz + oz;
      if (z < 0 || z >= nz) continue;
      for (let oy = oyMin; oy <= oyMax; oy++) {
        const y = cy + oy;
        if (y < 0 || y >= ny) continue;
        for (let ox = oxMin; ox <= oxMax; ox++) {
          if (ox * ox + oy * oy + oz * oz > r2) continue;
          const x = cx + ox;
          if (x < 0 || x >= nx) continue;
          const idx = z * sliceStride + y * nx + x;
          if (!overwrite && val !== 0 && voxels[idx] !== 0) continue;
          if (voxels[idx] !== val || diffMap.has(idx)) {
            diffMap.set(idx, val);
          }
        }
      }
    }
  }
}

function fillClosedPolygon2DOnSlice(
  voxels,
  dims,
  contourVoxels,
  sliceAxis,
  sliceIdx,
  uAxis,
  vAxis,
  value,
  overwrite,
  diffMap
) {
  if (!Array.isArray(contourVoxels) || contourVoxels.length === 0) return;
  const dimArr = normalizeVolDims3(dims);
  const [nx, ny, nz] = dimArr;
  const sliceStride = nx * ny;
  const strides = [1, nx, sliceStride];
  const sIdx = Math.max(0, Math.min(dimArr[sliceAxis] - 1, Math.round(sliceIdx)));
  const val = Math.max(0, Math.min(255, value | 0));
  const nu = dimArr[uAxis];
  const nv = dimArr[vAxis];

  // Snap polygon vertices onto fixed sliceIdx and stamp closed perimeter
  const snapped = contourVoxels.map((pt) => {
    const copy = [Math.round(pt[0]), Math.round(pt[1]), Math.round(pt[2])];
    copy[sliceAxis] = sIdx;
    return copy;
  });

  for (let i = 0; i < snapped.length; i++) {
    const pA = snapped[i];
    const pB = snapped[(i + 1) % snapped.length];
    stampBrushSegmentVoxels(voxels, [nx, ny, nz], pA, pB, val, 1, false, sliceAxis, overwrite, diffMap);
  }
  if (snapped.length < 3) return;

  const poly = snapped.map((p) => [p[uAxis], p[vAxis]]);
  let minV = nv;
  let maxV = -1;
  for (const [, py] of poly) {
    if (py < minV) minV = py;
    if (py > maxV) maxV = py;
  }
  minV = Math.max(0, Math.floor(minV));
  maxV = Math.min(nv - 1, Math.ceil(maxV));

  const baseOffset = sIdx * strides[sliceAxis];
  const uStride = strides[uAxis];
  const vStride = strides[vAxis];
  const xs = [];

  for (let v = minV; v <= maxV; v++) {
    const scanY = v + 0.50013;
    xs.length = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0];
      const yi = poly[i][1];
      const xj = poly[j][0];
      const yj = poly[j][1];
      const intersects = (yi <= scanY && yj > scanY) || (yj <= scanY && yi > scanY);
      if (intersects) {
        const xHit = xi + ((scanY - yi) / (yj - yi)) * (xj - xi);
        xs.push(xHit);
      }
    }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const uStart = Math.max(0, Math.ceil(xs[k] - 0.5));
      const uEnd = Math.min(nu - 1, Math.floor(xs[k + 1] + 0.5));
      for (let u = uStart; u <= uEnd; u++) {
        const idx = baseOffset + u * uStride + v * vStride;
        if (!overwrite && val !== 0 && voxels[idx] !== 0) continue;
        if (voxels[idx] !== val) {
          diffMap.set(idx, val);
        }
      }
    }
  }
}

function floodFill2DOnSlice(voxels, dims, seedVox, sliceAxis, uAxis, vAxis, fillValue, overwrite = true) {
  const diffMap = new Map();
  if (!voxels || !seedVox) return diffMap;
  const dimArr = normalizeVolDims3(dims);
  const [nx, ny] = dimArr;
  const strides = [1, nx, nx * ny];

  const sIdx = Math.max(0, Math.min(dimArr[sliceAxis] - 1, Math.round(seedVox[sliceAxis])));
  const su = Math.round(seedVox[uAxis]);
  const sv = Math.round(seedVox[vAxis]);
  const nu = dimArr[uAxis];
  const nv = dimArr[vAxis];
  if (su < 0 || su >= nu || sv < 0 || sv >= nv) return diffMap;

  const baseOffset = sIdx * strides[sliceAxis];
  const uStride = strides[uAxis];
  const vStride = strides[vAxis];
  const val = Math.max(0, Math.min(255, fillValue | 0));

  const startIdx = baseOffset + su * uStride + sv * vStride;
  const targetVal = voxels[startIdx];
  if (targetVal === val) return diffMap;
  if (!overwrite && val !== 0 && targetVal !== 0) return diffMap;

  const totalSlice = nu * nv;
  const visited = new Uint8Array(totalSlice);
  const queue = new Int32Array(totalSlice);
  let qHead = 0;
  let qTail = 0;

  const startPos = sv * nu + su;
  visited[startPos] = 1;
  queue[qTail++] = startPos;

  while (qHead < qTail) {
    const pos = queue[qHead++];
    const u = pos % nu;
    const v = (pos - u) / nu;
    const idx = baseOffset + u * uStride + v * vStride;
    diffMap.set(idx, val);

    if (u > 0) {
      const nPos = pos - 1;
      if (!visited[nPos] && voxels[idx - uStride] === targetVal) {
        visited[nPos] = 1;
        queue[qTail++] = nPos;
      }
    }
    if (u + 1 < nu) {
      const nPos = pos + 1;
      if (!visited[nPos] && voxels[idx + uStride] === targetVal) {
        visited[nPos] = 1;
        queue[qTail++] = nPos;
      }
    }
    if (v > 0) {
      const nPos = pos - nu;
      if (!visited[nPos] && voxels[idx - vStride] === targetVal) {
        visited[nPos] = 1;
        queue[qTail++] = nPos;
      }
    }
    if (v + 1 < nv) {
      const nPos = pos + nu;
      if (!visited[nPos] && voxels[idx + vStride] === targetVal) {
        visited[nPos] = 1;
        queue[qTail++] = nPos;
      }
    }
  }
  return diffMap;
}

function clearSliceDiffOnVoxels(voxels, dims, sliceAxis, sliceIdx, targetLabel = 0) {
  const diffMap = new Map();
  if (!voxels) return diffMap;
  const dimArr = normalizeVolDims3(dims);
  const [nx] = dimArr;
  const strides = [1, nx, nx * dimArr[1]];
  const otherAxes = [0, 1, 2].filter((a) => a !== sliceAxis);
  const uAxis = otherAxes[0];
  const vAxis = otherAxes[1];
  const sIdx = Math.max(0, Math.min(dimArr[sliceAxis] - 1, Math.round(sliceIdx)));
  const nu = dimArr[uAxis];
  const nv = dimArr[vAxis];
  const baseOffset = sIdx * strides[sliceAxis];
  const uStride = strides[uAxis];
  const vStride = strides[vAxis];

  for (let v = 0; v < nv; v++) {
    const vBase = baseOffset + v * vStride;
    for (let u = 0; u < nu; u++) {
      const idx = vBase + u * uStride;
      const cur = voxels[idx];
      if (cur > 0 && (targetLabel === 0 || cur === targetLabel)) {
        diffMap.set(idx, 0);
      }
    }
  }
  return diffMap;
}

function buildDrawingDiscreteColormap(layer) {
  const R = new Array(256).fill(0);
  const G = new Array(256).fill(0);
  const B = new Array(256).fill(0);
  const A = new Array(256).fill(0);
  const I = new Array(256);

  for (let i = 0; i < 256; i++) {
    I[i] = i;
  }

  const baseRgb = hexToRgb(layer?.colorHex || "#f43f5e");
  const palette = Array.isArray(layer?.labelPalette) && layer.labelPalette.length
    ? layer.labelPalette
    : DEFAULT_MULTILABEL_PALETTE;

  // Bins 0..2 remain RGBA (0,0,0,0) as a transparent guard band for empty background voxels (f = 0).
  if (layer?.mode === "multilabel") {
    for (let k = 3; k < 256; k++) {
      const fVal = 0.5 + 6.0 * (k / 255.0);
      const lbl = Math.max(1, Math.min(6, Math.round(fVal)));
      const entry = palette[(lbl - 1) % palette.length];
      const [r, g, b] = hexToRgb(entry?.hex || layer.colorHex);
      R[k] = r;
      G[k] = g;
      B[k] = b;
      A[k] = 255;
    }
  } else {
    for (let k = 3; k < 256; k++) {
      R[k] = baseRgb[0];
      G[k] = baseRgb[1];
      B[k] = baseRgb[2];
      A[k] = 255;
    }
  }
  return { R, G, B, A, I, min: 0, max: 0 };
}

function applyDrawingLayerColormap(layer, vol) {
  if (!layer || !vol) return;
  const isMulti = layer.mode === "multilabel";
  const calMin = 0.5;
  const calMax = isMulti ? 6.5 : 1.5;
  const globalMax = isMulti ? 6 : 1;

  const cmapKey = `__draw_${layer.id}`;
  const cmap = buildDrawingDiscreteColormap(layer);
  if (state.nv && typeof state.nv.addColormap === "function") {
    state.nv.addColormap(cmapKey, cmap);
  }

  vol.colormapLabel = null;
  vol.colormapType = 0;
  vol.colorbarVisible = false;
  if (vol.hdr) {
    vol.hdr.cal_min = calMin;
    vol.hdr.cal_max = calMax;
  }
  vol.trustCalMinMax = true;
  vol.colormap = cmapKey;

  // Re-assert display window after setColormap()'s internal calMinMax() recalculation.
  vol.cal_min = calMin;
  vol.cal_max = calMax;
  vol.robust_min = calMin;
  vol.robust_max = calMax;
  vol.global_min = 0;
  vol.global_max = globalMax;
}

function getDrawingLayerById(layerId) {
  return state.drawing.layers.find((l) => l.id === layerId) || null;
}

function getActiveDrawingLayer() {
  return getDrawingLayerById(state.drawing.activeLayerId) || null;
}

function getBaseDrawingReferenceVolume() {
  if (!state.nv || !state.nv.volumes?.length) return null;
  const refIdx = getVolumeIndexById(state.refVolId);
  if (refIdx >= 0) {
    const rv = state.nv.volumes[refIdx];
    const rm = state.layerMeta.get(rv.id);
    if (rm && !rm.isDrawingLayer && !rm.isSyntheticVerification) return rv;
  }
  return (
    state.nv.volumes.find((v) => {
      const m = state.layerMeta.get(v.id);
      return m && !m.isDrawingLayer && !m.isSyntheticVerification;
    }) || null
  );
}

function refreshDrawingStats(layer) {
  if (!layer) return;
  const stats = computeLayerStatsData(layer.voxels, layer.affine);
  layer.voxelCount = stats.voxelCount;
  layer.volumeCc = stats.volumeCc;
  layer.perLabelCounts = stats.perLabelCounts;
  layer.voxelMm3 = stats.voxelMm3;
}

function commitDrawingLayerToGL(layer, skipRenderList = false) {
  if (!layer || !state.nv) return;
  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    const vol = state.nv.volumes[volIdx];
    if (vol.img && vol.img.length === layer.voxels.length) {
      vol.img.set(layer.voxels);
    }
    applyDrawingLayerColormap(layer, vol);
    if (typeof vol.calculateRAS === "function") {
      vol.calculateRAS();
    }
  }
  refreshDrawingStats(layer);
  state.nv.updateGLVolume();
  state.nv.drawScene();
  syncDrawActiveBanner();
  if (!skipRenderList) {
    renderLayerList();
  } else {
    renderDrawingStudioUI();
  }
}

function applyDiffDirectionToLayer(layer, entry, isUndo) {
  if (!layer || !entry || !layer.voxels) return;
  const srcArr = isUndo ? entry.prev : entry.next;
  for (let i = 0; i < entry.indices.length; i++) {
    layer.voxels[entry.indices[i]] = srcArr[i];
  }
}

function undoDrawingLayer(layer = null) {
  const target = layer || getActiveDrawingLayer();
  if (!target || !target.undoStack?.length) {
    showToast("Nothing to undo on active drawing layer.", 1800);
    return false;
  }
  const entry = target.undoStack.pop();
  applyDiffDirectionToLayer(target, entry, true);
  target.redoStack.push(entry);
  commitDrawingLayerToGL(target);
  showToast(`Undid ${entry.indices.length} voxel change(s).`, 1500);
  return true;
}

function redoDrawingLayer(layer = null) {
  const target = layer || getActiveDrawingLayer();
  if (!target || !target.redoStack?.length) {
    showToast("Nothing to redo on active drawing layer.", 1800);
    return false;
  }
  const entry = target.redoStack.pop();
  applyDiffDirectionToLayer(target, entry, false);
  target.undoStack.push(entry);
  commitDrawingLayerToGL(target);
  showToast(`Redid ${entry.indices.length} voxel change(s).`, 1500);
  return true;
}

function getActivePenValueForLayer(layer) {
  if (!layer) return 1;
  if (state.drawing.tool === "eraser") return 0;
  if (layer.mode === "multilabel") return Math.max(1, Math.min(6, layer.activeLabel || 1));
  return 1;
}

async function createDrawingLayer(options = {}) {
  if (!state.nv) return null;
  const baseVol = getBaseDrawingReferenceVolume();
  let nx, ny, nz, affine;

  if (options.explicitDims && options.explicitAffine) {
    [nx, ny, nz] = normalizeVolDims3(options.explicitDims);
    affine = cloneMat4(options.explicitAffine);
  } else if (baseVol) {
    [nx, ny, nz] = normalizeVolDims3(getVolDims(baseVol));
    affine = cloneMat4(baseVol.hdr.affine);
  } else {
    showToast("Load an anatomical NIfTI/DICOM volume first to create a drawing layer.", 3200);
    return null;
  }

  const totalVox = nx * ny * nz;
  const voxels =
    options.voxels instanceof Uint8Array && options.voxels.length === totalVox
      ? options.voxels
      : new Uint8Array(totalVox);

  const counter = state.drawing.nextLayerCounter++;
  const colorHex =
    options.colorHex || DRAW_COLOR_SWATCHES[(counter - 1) % DRAW_COLOR_SWATCHES.length];
  const layerName = (options.name || `VOI_${counter}`).trim() || `VOI_${counter}`;
  const mode = options.mode === "multilabel" ? "multilabel" : "binary";
  const opacity = Number.isFinite(Number(options.opacity)) ? Math.max(0, Math.min(1, Number(options.opacity))) : 0.72;
  const visible = options.visible !== undefined ? Boolean(options.visible) : true;
  const labelPalette = Array.isArray(options.labelPalette) && options.labelPalette.length === 6
    ? options.labelPalette.map((p, idx) => ({
        id: idx + 1,
        name: String(p?.name || `ROI ${idx + 1}`),
        hex: p?.hex || DEFAULT_MULTILABEL_PALETTE[idx].hex,
      }))
    : DEFAULT_MULTILABEL_PALETTE.map((p) => ({ ...p }));

  const safeFileStem = layerName.replace(/[^a-zA-Z0-9._-]/g, "_") || `VOI_${counter}`;
  const niftiBytes = buildUint8Nifti1Bytes(nx, ny, nz, affine, voxels);
  const vol = await loadNVImageFromBytes(niftiBytes, {
    name: `${safeFileStem}.nii`,
    opacity: visible ? opacity : 0.0,
    cal_min: 0.5,
    cal_max: mode === "multilabel" ? 6.5 : 1.5,
    colorbarVisible: false,
  });

  if (vol.hdr) {
    vol.hdr.affine = cloneMat4(affine);
    vol.hdr.sform_code = 1;
  }

  const layer = {
    id: options.id || `draw_${Date.now()}_${counter}`,
    volId: vol.id,
    name: layerName,
    mode,
    colorHex,
    activeLabel: Math.max(1, Math.min(6, Number(options.activeLabel) || 1)),
    labelPalette,
    visible,
    opacity,
    dims: [nx, ny, nz],
    affine,
    voxels,
    undoStack: [],
    redoStack: [],
    voxelCount: 0,
    volumeCc: 0,
    voxelMm3: 1,
    perLabelCounts: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 },
  };

  applyDrawingLayerColormap(layer, vol);
  state.nv.addVolume(vol);
  registerVolumeMeta(vol, {
    label: layerName,
    modality: "Drawing VOI",
    opacity: visible ? opacity : 0.0,
    visible,
    isMask: true,
    isDktAtlas: false,
    isDrawingLayer: true,
    drawingLayerId: layer.id,
  });
  const volMeta = state.layerMeta.get(vol.id);
  if (volMeta) {
    volMeta.savedOpacity = opacity;
    volMeta.visible = visible;
  }

  refreshDrawingStats(layer);
  state.drawing.layers.push(layer);
  state.drawing.activeLayerId = layer.id;

  if (typeof vol.calculateRAS === "function") {
    vol.calculateRAS();
  }
  state.nv.updateGLVolume();
  state.nv.drawScene();
  renderLayerList();
  syncDrawHeaderButton();
  syncDrawActiveBanner();
  return layer;
}

function selectActiveDrawingLayer(layerId) {
  const layer = getDrawingLayerById(layerId);
  if (!layer) return;
  state.drawing.activeLayerId = layer.id;
  syncDrawActiveBanner();
  renderLayerList();
}

function deleteDrawingLayer(layerId) {
  const idxInDraw = state.drawing.layers.findIndex((l) => l.id === layerId);
  if (idxInDraw < 0) return;
  const layer = state.drawing.layers[idxInDraw];
  state.drawing.layers.splice(idxInDraw, 1);

  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    state.nv.removeVolumeByIndex(volIdx);
    state.layerMeta.delete(layer.volId);
  }

  if (state.drawing.activeLayerId === layerId) {
    const next = state.drawing.layers[Math.min(idxInDraw, state.drawing.layers.length - 1)] || null;
    state.drawing.activeLayerId = next ? next.id : null;
  }
  if (state.drawing.layers.length === 0 && state.drawing.paintEnabled) {
    setDrawingPaintEnabled(false);
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  renderLayerList();
  syncDrawHeaderButton();
  syncDrawActiveBanner();
}

function setDrawingLayerVisibility(layerId, visible) {
  const layer = getDrawingLayerById(layerId);
  if (!layer) return;
  layer.visible = Boolean(visible);
  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    const meta = state.layerMeta.get(layer.volId);
    if (meta) meta.visible = layer.visible;
    state.nv.setOpacity(volIdx, layer.visible ? layer.opacity : 0.0);
    state.nv.drawScene();
  }
  renderLayerList();
}

function setDrawingLayerOpacity(layerId, opacity) {
  const layer = getDrawingLayerById(layerId);
  if (!layer) return;
  const clamped = Math.max(0, Math.min(1, Number(opacity)));
  layer.opacity = clamped;
  layer.visible = clamped > 0;
  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    const meta = state.layerMeta.get(layer.volId);
    if (meta) {
      meta.visible = clamped > 0;
      if (clamped > 0) meta.savedOpacity = clamped;
    }
    state.nv.setOpacity(volIdx, clamped);
    state.nv.drawScene();
  }
  renderLayerList();
}

function renameDrawingLayer(layerId, newName) {
  const layer = getDrawingLayerById(layerId);
  if (!layer) return;
  const clean = String(newName || "").trim() || layer.name;
  layer.name = clean;
  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    const vol = state.nv.volumes[volIdx];
    const meta = state.layerMeta.get(layer.volId);
    vol.name = `${clean.replace(/[^a-zA-Z0-9._-]/g, "_")}.nii`;
    if (meta) meta.label = clean;
  }
  syncDrawActiveBanner();
  renderLayerList();
}

function setDrawingLayerColor(layerId, hex) {
  const layer = getDrawingLayerById(layerId);
  if (!layer) return;
  layer.colorHex = hex;
  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    applyDrawingLayerColormap(layer, state.nv.volumes[volIdx]);
    state.nv.updateGLVolume();
    state.nv.drawScene();
  }
  syncDrawActiveBanner();
  renderLayerList();
}

function setDrawingLayerMode(layerId, mode) {
  const layer = getDrawingLayerById(layerId);
  if (!layer) return;
  layer.mode = mode === "multilabel" ? "multilabel" : "binary";
  const volIdx = getVolumeIndexById(layer.volId);
  if (volIdx >= 0) {
    applyDrawingLayerColormap(layer, state.nv.volumes[volIdx]);
    state.nv.updateGLVolume();
    state.nv.drawScene();
  }
  syncDrawActiveBanner();
  renderLayerList();
}

function clearSliceOnActiveDrawingLayer() {
  const layer = getActiveDrawingLayer();
  if (!layer) return;
  const invAff = invertMat4(layer.affine);
  const vox = applyMat4Pt(invAff, state.lastCrosshairRAS || [0, 0, 0]);
  const ax = state.drawing.lastSliceAxis;
  const sIdx = Math.max(0, Math.min(layer.dims[ax] - 1, Math.round(vox[ax])));
  const diffMap = clearSliceDiffOnVoxels(layer.voxels, layer.dims, ax, sIdx, 0);
  const entry = applySparseDiffToVoxels(layer.voxels, diffMap, 0, true);
  if (!entry) {
    showToast("Current slice is already empty.", 1800);
    return;
  }
  layer.undoStack.push(entry);
  if (layer.undoStack.length > 30) layer.undoStack.shift();
  layer.redoStack = [];
  commitDrawingLayerToGL(layer);
  showToast(`Cleared ${entry.indices.length} voxel(s) on slice ${sIdx}.`, 1800);
}

function clearActiveDrawingLayer() {
  const layer = getActiveDrawingLayer();
  if (!layer) return;
  const diffMap = new Map();
  for (let i = 0; i < layer.voxels.length; i++) {
    if (layer.voxels[i] > 0) diffMap.set(i, 0);
  }
  const entry = applySparseDiffToVoxels(layer.voxels, diffMap, 0, true);
  if (!entry) {
    showToast("Layer is already empty.", 1800);
    return;
  }
  layer.undoStack.push(entry);
  if (layer.undoStack.length > 30) layer.undoStack.shift();
  layer.redoStack = [];
  commitDrawingLayerToGL(layer);
  showToast(`Cleared entire layer "${layer.name}" (${entry.indices.length} voxels).`, 2000);
}

function resetDrawingWorkspace() {
  state.drawing.paintEnabled = false;
  state.drawing.isStroking = false;
  state.drawing.layers = [];
  state.drawing.activeLayerId = null;
  state.drawing.nextLayerCounter = 1;
  state.drawing.regionGrow.sourceVolId = null;

  const nvCanvas = document.getElementById("niivueCanvas");
  nvCanvas?.classList.remove("draw-mode-active", "draw-brush-hover");
  clearStrokeOverlayCanvas();
  syncDrawHeaderButton();
  syncDrawActiveBanner();
  renderDrawingStudioUI();
}

function pushUpdatedVoxelsAsUndo(layer, nextVoxels) {
  const diffMap = new Map();
  for (let i = 0; i < layer.voxels.length; i++) {
    if (layer.voxels[i] !== nextVoxels[i]) {
      diffMap.set(i, nextVoxels[i]);
    }
  }
  const entry = applySparseDiffToVoxels(layer.voxels, diffMap, 0, true);
  if (!entry) return null;
  layer.undoStack.push(entry);
  if (layer.undoStack.length > 30) layer.undoStack.shift();
  layer.redoStack = [];
  commitDrawingLayerToGL(layer);
  return entry;
}

async function runDrawingMorphOp(op, extraParams = {}) {
  const layer = getActiveDrawingLayer();
  if (!layer) {
    showToast("Select or create a drawing layer first.", 2200);
    return;
  }
  try {
    setCoregStatus(`Running ${op.replace(/_/g, " ")}...`, true);
    const targetLabel =
      op === "interpolate_slices" || op === "fill_holes" || op === "largest_island" || op === "dilate" || op === "erode"
        ? layer.mode === "multilabel"
          ? 0
          : 1
        : Math.max(1, getActivePenValueForLayer(layer));

    const resp = await fetch("/api/drawing-morph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        op,
        voxelsBase64: uint8ArrayToBase64(layer.voxels),
        dims: layer.dims,
        affine: layer.affine,
        targetLabel,
        overwrite: state.drawing.overwriteLabels,
        axis: state.drawing.interpAxis,
        ...extraParams,
      }),
    });
    const res = await resp.json();
    if (!resp.ok || !res.voxelsBase64) {
      throw new Error(res.error || "Morphological operation failed");
    }
    const nextBytes = base64ToUint8Array(res.voxelsBase64);
    pushUpdatedVoxelsAsUndo(layer, nextBytes);
    setCoregStatus("Ready", false);
    showToast(res.message || `Applied ${op} cleanly.`, 2600);
  } catch (err) {
    console.error("Drawing morph error:", err);
    setCoregStatus("Ready", false);
    showToast(`Drawing tool error: ${err.message}`, 3600);
  }
}

async function runBackendRegionGrow(layer, seedVoxel, sliceAxis) {
  if (!layer || !state.nv) return;
  const candidateId = state.drawing.regionGrow.sourceVolId || state.refVolId;
  let srcVol = candidateId && getVolumeIndexById(candidateId) >= 0 ? state.nv.volumes[getVolumeIndexById(candidateId)] : getBaseDrawingReferenceVolume();
  if (!srcVol) {
    showToast("No anatomical source volume available for Region Growing.", 2600);
    return;
  }
  const srcMeta = state.layerMeta.get(srcVol.id);
  let srcBytes = srcMeta?.rawBytes || null;
  if (!srcBytes && srcVol.img) {
    const sDims = normalizeVolDims3(getVolDims(srcVol));
    srcBytes = buildUint8Nifti1Bytes(sDims[0], sDims[1], sDims[2], srcVol.hdr.affine, new Uint8Array(srcVol.img));
  }
  if (!srcBytes) {
    showToast("Selected source volume cannot be read for Region Growing.", 2600);
    return;
  }
  const penVal = getActivePenValueForLayer(layer) || 1;
  await runDrawingMorphOp("region_grow", {
    targetLabel: penVal,
    seedVoxel: seedVoxel.map((v) => Math.round(v)),
    sliceAxis,
    scope: state.drawing.regionGrow.scope,
    tolerancePct: state.drawing.regionGrow.tolerancePct,
    maxRadiusMm: state.drawing.regionGrow.maxRadiusMm,
    sourceName: srcVol.name || "source.nii.gz",
    sourceBase64: uint8ArrayToBase64(srcBytes),
    sourceAffine: srcVol.hdr?.affine || identityMat4(),
  });
}

function buildDrawingLabelsExportMetadata(layer) {
  if (!layer || layer.mode !== "multilabel") return null;
  refreshDrawingStats(layer);
  const labels = (layer.labelPalette || []).map((p, i) => {
    const vox = layer.perLabelCounts?.[i + 1] || 0;
    return {
      id: i + 1,
      name: p.name,
      hex: p.hex,
      voxelCount: vox,
      volumeCc: Number(((vox * (layer.voxelMm3 || 1)) / 1000.0).toFixed(4)),
    };
  });
  return {
    layerName: layer.name,
    mode: "multilabel",
    dims: layer.dims,
    labels,
  };
}

function mergeVisibleDrawingLayersVoxels(layersList) {
  const visibleLayers = (layersList || []).filter((l) => l && l.visible);
  if (visibleLayers.length === 0) return null;
  const first = visibleLayers[0];
  const merged = new Uint8Array(first.voxels.length);
  const labels = [];
  let labelId = 1;
  for (const layer of visibleLayers) {
    if (layer.voxels.length !== merged.length) continue;
    let count = 0;
    for (let i = 0; i < merged.length; i++) {
      if (layer.voxels[i] > 0) {
        merged[i] = labelId;
        count++;
      }
    }
    labels.push({
      id: labelId,
      name: layer.name,
      hex: layer.colorHex,
      voxelCount: count,
      volumeCc: Number(((count * (layer.voxelMm3 || 1)) / 1000.0).toFixed(4)),
    });
    labelId++;
    if (labelId > 255) break;
  }
  return {
    voxels: merged,
    dims: [...first.dims],
    affine: cloneMat4(first.affine),
    labelsMetadata: {
      layerName: "Merged_VOIs",
      mode: "multilabel",
      dims: [...first.dims],
      labels,
    },
  };
}

async function performDrawingSaveRequest({
  voxels,
  dims,
  affine,
  format = "voi",
  fileName = "drawing",
  mode = "download",
  labelsMetadata = null,
}) {
  const fmt = ["voi", "nii.gz", "nii"].includes(format) ? format : "voi";
  const cleanStem = stripDrawingFilenameExt(fileName).replace(/[^a-zA-Z0-9._-]/g, "_") || "drawing_voi";
  const suggestedName = `${cleanStem}.${fmt}`;

  let fileHandle = null;
  let pickerFailedNonAbort = false;
  if (mode === "save_as" && typeof window.showSaveFilePicker === "function") {
    try {
      fileHandle = await window.showSaveFilePicker({ suggestedName });
    } catch (err) {
      if (err && err.name === "AbortError") {
        showToast("Save cancelled.", 1800);
        return false;
      }
      fileHandle = null;
      pickerFailedNonAbort = true;
    }
  }

  const effectiveMode = fileHandle ? "download" : mode;
  const resp = await fetch("/api/drawing-save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      voxelsBase64: uint8ArrayToBase64(voxels),
      dims,
      affine,
      format: fmt,
      fileName: cleanStem,
      mode: effectiveMode,
      labelsMetadata,
    }),
  });
  const res = await resp.json();
  if (!resp.ok) {
    throw new Error(res.error || "Failed to export drawing layer");
  }

  if (fileHandle) {
    const outBytes = base64ToUint8Array(res.fileBase64);
    const writable = await fileHandle.createWritable();
    await writable.write(outBytes);
    await writable.close();
    if (labelsMetadata) {
      downloadTextFile(
        `${stripDrawingFilenameExt(fileHandle.name || res.fileName)}_labels.json`,
        JSON.stringify(labelsMetadata, null, 2)
      );
    }
    showToast(`Saved ${fileHandle.name || res.fileName}!`, 3500);
    return true;
  }

  if (res.cancelled) {
    showToast("Save cancelled.", 1800);
    return false;
  }
  if (res.saved) {
    showToast(`Saved ${res.fileName} to ${res.savedPath}`, 4200);
    return true;
  }

  const outBytes = base64ToUint8Array(res.fileBase64);
  if (mode === "save_as" && !pickerFailedNonAbort && res.needsFallback && typeof window.showSaveFilePicker === "function") {
    try {
      const fh = await window.showSaveFilePicker({ suggestedName: res.fileName });
      const writable = await fh.createWritable();
      await writable.write(outBytes);
      await writable.close();
      if (labelsMetadata) {
        downloadTextFile(
          `${stripDrawingFilenameExt(fh.name || res.fileName)}_labels.json`,
          JSON.stringify(labelsMetadata, null, 2)
        );
      }
      showToast(`Saved ${fh.name || res.fileName}!`, 3500);
      return true;
    } catch (err) {
      if (err && err.name === "AbortError") {
        showToast("Save cancelled.", 1800);
        return false;
      }
    }
  }

  downloadBlobBytes(res.fileName, outBytes);
  if (labelsMetadata) {
    downloadTextFile(`${stripDrawingFilenameExt(res.fileName)}_labels.json`, JSON.stringify(labelsMetadata, null, 2));
  }
  showToast(`Downloaded ${res.fileName}`, 3200);
  return true;
}

async function exportDrawingLayer(layerId = null, mode = "download", formatOverride = null, fileStemOverride = null) {
  const layer = layerId ? getDrawingLayerById(layerId) : getActiveDrawingLayer();
  if (!layer) {
    showToast("No drawing layer selected to save.", 2200);
    return false;
  }
  const patientId = (document.getElementById("patientIdInput")?.value?.trim() || "sub-01").replace(/[^a-zA-Z0-9._-]/g, "_");
  const safeLayer = layer.name.replace(/[^a-zA-Z0-9._-]/g, "_");
  const stem = (fileStemOverride || `${patientId}_${safeLayer}`).trim() || `${patientId}_VOI`;
  const fmt = formatOverride || state.drawing.exportFormat || "voi";
  const labelsMetadata = buildDrawingLabelsExportMetadata(layer);
  try {
    setCoregStatus("Exporting VOI...", true);
    return await performDrawingSaveRequest({
      voxels: layer.voxels,
      dims: layer.dims,
      affine: layer.affine,
      format: fmt,
      fileName: stem,
      mode,
      labelsMetadata,
    });
  } catch (err) {
    console.error("Export drawing error:", err);
    showToast(`Export error: ${err.message}`, 3600);
    return false;
  } finally {
    setCoregStatus("Ready", false);
  }
}

async function mergeAndExportVisibleDrawingLayers(mode = "save_as", formatOverride = null) {
  const merged = mergeVisibleDrawingLayersVoxels(state.drawing.layers);
  if (!merged) {
    showToast("Turn ON visibility for at least one drawing layer to merge.", 2400);
    return false;
  }
  const patientId = (document.getElementById("patientIdInput")?.value?.trim() || "sub-01").replace(/[^a-zA-Z0-9._-]/g, "_");
  const fmt = formatOverride || state.drawing.exportFormat || "voi";
  try {
    setCoregStatus("Merging VOIs...", true);
    return await performDrawingSaveRequest({
      voxels: merged.voxels,
      dims: merged.dims,
      affine: merged.affine,
      format: fmt,
      fileName: `${patientId}_merged_VOIs`,
      mode,
      labelsMetadata: merged.labelsMetadata,
    });
  } catch (err) {
    console.error("Merge drawing export error:", err);
    showToast(`Merge export error: ${err.message}`, 3600);
    return false;
  } finally {
    setCoregStatus("Ready", false);
  }
}

async function importBytesAsDrawingLayer(rawBytes, fileName, options = {}) {
  const baseVol = getBaseDrawingReferenceVolume();
  const payload = {
    sourceName: fileName,
    sourceBase64: uint8ArrayToBase64(rawBytes),
  };
  if (baseVol && !options.useNativeGrid) {
    payload.refDims = normalizeVolDims3(getVolDims(baseVol));
    payload.refAffine = baseVol.hdr.affine;
  }
  if (options.sourceAffine) {
    payload.sourceAffine = options.sourceAffine;
  }
  if (options.thresholdMin !== undefined && options.thresholdMin !== null) {
    payload.thresholdMin = options.thresholdMin;
  }

  const resp = await fetch("/api/drawing-import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const res = await resp.json();
  if (!resp.ok || !res.voxelsBase64) {
    throw new Error(res.error || `Failed to import ${fileName}`);
  }

  const voxels = base64ToUint8Array(res.voxelsBase64);
  const layer = await createDrawingLayer({
    id: options.id,
    name: options.name || stripDrawingFilenameExt(fileName),
    mode: options.mode || res.suggestedMode || "binary",
    colorHex: options.colorHex,
    opacity: options.opacity,
    visible: options.visible,
    activeLabel: options.activeLabel,
    labelPalette: options.labelPalette,
    voxels,
    explicitDims: res.dims,
    explicitAffine: res.affine,
  });

  if (options.openStudio) {
    openDrawingStudioPopout();
  }
  if (!options.quiet && layer) {
    showToast(
      `Imported "${layer.name}" (${layer.volumeCc.toFixed(2)} cm³${res.resampled ? ", nearest-neighbor resampled to Ref grid" : ""}).`,
      3400
    );
  }
  return layer;
}

async function importFileAsDrawingLayer(file, options = {}) {
  if (!file) return null;
  const ab = await file.arrayBuffer();
  return importBytesAsDrawingLayer(new Uint8Array(ab), file.name, options);
}

async function convertVolumeToDrawingLayer(volId) {
  const volIdx = getVolumeIndexById(volId);
  if (volIdx < 0) return null;
  const vol = state.nv.volumes[volIdx];
  const meta = state.layerMeta.get(volId);
  if (!vol || !meta || meta.isDrawingLayer) return null;

  let rawBytes = meta.rawBytes || null;
  if (!rawBytes && vol.img) {
    const dims = normalizeVolDims3(getVolDims(vol));
    rawBytes = buildUint8Nifti1Bytes(dims[0], dims[1], dims[2], vol.hdr.affine, new Uint8Array(vol.img));
  }
  if (!rawBytes) {
    showToast("Unable to extract volume bytes to convert into a drawing layer.", 3000);
    return null;
  }

  const thresholdMin = !meta.isMask
    ? Number.isFinite(vol.cal_min) && vol.cal_min > 0
      ? vol.cal_min
      : Math.max(0.5, (vol.global_max || vol.cal_max || 1) * 0.4)
    : null;

  const layer = await importBytesAsDrawingLayer(rawBytes, `${meta.label || vol.name}.nii.gz`, {
    name: `${meta.label || stripDrawingFilenameExt(vol.name)}_ROI`,
    sourceAffine: vol.hdr?.affine || identityMat4(),
    thresholdMin,
    openStudio: true,
    quiet: true,
  });
  if (layer) {
    showToast(
      `Converted "${meta.label}" into editable Drawing Layer "${layer.name}" (${layer.volumeCc.toFixed(2)} cm³).`,
      3400
    );
  }
  return layer;
}

async function rehydrateRestoredDrawingLayers(manifestEntries, drawingBytesByName) {
  if (!Array.isArray(manifestEntries) || manifestEntries.length === 0) return;
  for (const dEntry of manifestEntries) {
    const rawBytes = drawingBytesByName?.get(dEntry.fileName) || null;
    if (!rawBytes) continue;
    try {
      await importBytesAsDrawingLayer(rawBytes, dEntry.fileName, {
        id: dEntry.id,
        name: dEntry.name || stripDrawingFilenameExt(dEntry.fileName),
        mode: dEntry.mode || "binary",
        colorHex: dEntry.colorHex,
        opacity: dEntry.opacity,
        visible: dEntry.visible,
        activeLabel: dEntry.activeLabel,
        labelPalette: dEntry.labelPalette,
        quiet: true,
      });
    } catch (err) {
      console.warn("Failed to restore drawing layer:", dEntry.fileName, err);
    }
  }
}

// ============================================================================
// Drawing Studio Pop-Out UI, Banner, Keyboard Shortcuts & Brush Overlay Canvas
// ============================================================================
function syncDrawHeaderButton() {
  const btn = document.getElementById("btnToggleDrawStudio");
  if (!btn) return;
  const popOpen = Boolean(state.drawing.popout && state.drawing.popout.isOpen());
  btn.classList.toggle("active", popOpen);
  btn.classList.toggle("paint-on", state.drawing.paintEnabled);
  btn.setAttribute("aria-expanded", String(popOpen));
}

function syncDrawActiveBanner() {
  const banner = document.getElementById("drawActiveBanner");
  if (!banner) return;
  const activeLayer = getActiveDrawingLayer();
  if (!state.drawing.paintEnabled || !activeLayer) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  const dot = document.getElementById("drawBannerDot");
  const text = document.getElementById("drawBannerText");
  let dotHex = activeLayer.colorHex || "#f43f5e";
  if (activeLayer.mode === "multilabel") {
    dotHex = activeLayer.labelPalette?.[activeLayer.activeLabel - 1]?.hex || dotHex;
  }
  if (dot) dot.style.background = dotHex;
  const toolDef = DRAW_TOOL_DEFS.find((t) => t.id === state.drawing.tool) || DRAW_TOOL_DEFS[0];
  if (text) {
    text.textContent = `Paint: ${activeLayer.name} • ${toolDef.label} (${state.drawing.brushRadius}px ${
      state.drawing.brush3D ? "3D" : "2D"
    })`;
  }
}

async function setDrawingPaintEnabled(forceState = null) {
  const next = forceState !== null ? Boolean(forceState) : !state.drawing.paintEnabled;
  if (next) {
    if (state.drawing.layers.length === 0) {
      const created = await createDrawingLayer();
      if (!created) return false;
    }
    if (state.zoom2d.panModeActive) {
      togglePan2dMode(false);
    }
    state.drawing.paintEnabled = true;
  } else {
    state.drawing.paintEnabled = false;
    state.drawing.isStroking = false;
    clearStrokeOverlayCanvas();
  }

  const nvCanvas = document.getElementById("niivueCanvas");
  if (nvCanvas) {
    nvCanvas.classList.toggle("draw-mode-active", state.drawing.paintEnabled);
    if (!state.drawing.paintEnabled) nvCanvas.classList.remove("draw-brush-hover");
  }
  syncDrawHeaderButton();
  syncDrawActiveBanner();
  renderDrawingStudioUI();
  return state.drawing.paintEnabled;
}

function renderDrawingStudioUI() {
  const pop = state.drawing.popout;
  if (!pop || !pop.bodyEl) return;
  const container = pop.bodyEl;
  const activeLayer = getActiveDrawingLayer();
  const baseVol = getBaseDrawingReferenceVolume();
  const baseDims = baseVol ? normalizeVolDims3(getVolDims(baseVol)) : null;
  const baseLabel = baseVol ? state.layerMeta.get(baseVol.id)?.label || baseVol.name : "None loaded";

  const patientId = (document.getElementById("patientIdInput")?.value?.trim() || "sub-01").replace(/[^a-zA-Z0-9._-]/g, "_");
  const defaultStem = activeLayer
    ? `${patientId}_${activeLayer.name.replace(/[^a-zA-Z0-9._-]/g, "_")}`
    : `${patientId}_VOI`;
  const existingFilenameInput = container.querySelector("#drawExportFileInput");
  const keepCustomStem =
    existingFilenameInput &&
    existingFilenameInput.dataset.forLayerId === (activeLayer?.id || "") &&
    existingFilenameInput.value.trim()
      ? existingFilenameInput.value.trim()
      : defaultStem;

  const nonDrawingVols = (state.nv?.volumes || []).filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isDrawingLayer && !m.isSyntheticVerification;
  });
  const rgSrcId = state.drawing.regionGrow.sourceVolId || state.refVolId || nonDrawingVols[0]?.id || "";
  const isGrowTool = state.drawing.tool === "region_grow";

  const layerRowsHtml =
    state.drawing.layers.length === 0
      ? `<div class="draw-empty-note">No drawing layers yet. Click <strong>+ New Layer</strong> or <strong>Import .voi</strong> to begin.</div>`
      : state.drawing.layers
          .map((l) => {
            const isActive = activeLayer && l.id === activeLayer.id;
            const swColor =
              l.mode === "multilabel"
                ? l.labelPalette?.[l.activeLabel - 1]?.hex || l.colorHex
                : l.colorHex;
            return `
              <div class="draw-layer-item ${isActive ? "active" : ""}" data-draw-select-layer="${l.id}">
                <input type="checkbox" data-draw-layer-vis="${l.id}" ${l.visible ? "checked" : ""} title="Show / Hide layer" />
                <span class="draw-layer-swatch-dot" style="background:${swColor};"></span>
                <input type="text" class="draw-layer-name-input" data-draw-layer-rename="${l.id}" value="${escapeDrawHtml(l.name)}" title="Rename layer" />
                <span class="draw-vol-pill" title="${l.voxelCount} voxels">${l.volumeCc.toFixed(2)} cm³</span>
                <button type="button" class="icon-btn" data-draw-layer-del="${l.id}" title="Delete drawing layer">✕</button>
              </div>
            `;
          })
          .join("");

  let activeColorHtml = "";
  if (activeLayer) {
    if (activeLayer.mode === "binary") {
      activeColorHtml = `
        <div class="draw-color-swatches">
          ${DRAW_COLOR_SWATCHES.map(
            (hex) =>
              `<button type="button" class="draw-color-swatch ${
                activeLayer.colorHex.toLowerCase() === hex ? "active" : ""
              }" data-draw-color="${hex}" style="background:${hex};" title="Set layer color ${hex}"></button>`
          ).join("")}
          <input type="color" class="draw-color-picker" id="drawCustomColorInput" value="${activeLayer.colorHex}" title="Pick custom layer color" />
        </div>
      `;
    } else {
      activeColorHtml = `
        <div class="draw-multilabel-list">
          ${activeLayer.labelPalette
            .map((p, idx) => {
              const lblId = idx + 1;
              const isAct = activeLayer.activeLabel === lblId;
              const vox = activeLayer.perLabelCounts?.[lblId] || 0;
              const cc = ((vox * (activeLayer.voxelMm3 || 1)) / 1000).toFixed(2);
              return `
                <div class="draw-multilabel-row ${isAct ? "active" : ""}" data-draw-pick-label="${lblId}">
                  <span class="draw-label-num" style="background:${p.hex};">${lblId}</span>
                  <input type="text" class="draw-layer-name-input" data-draw-label-name="${lblId}" value="${escapeDrawHtml(p.name)}" />
                  <span class="draw-vol-pill">${cc} cm³</span>
                </div>
              `;
            })
            .join("")}
        </div>
      `;
    }
  }

  container.innerHTML = `
    <div class="draw-studio-body">
      <div class="draw-studio-topstrip">
        <button id="btnDrawPaintToggle" type="button" class="draw-paint-toggle-btn ${
          state.drawing.paintEnabled ? "active" : ""
        }" title="Toggle interactive 2D slice painting ON / OFF (Shortcut: E)">
          <span class="draw-paint-dot"></span>
          <span>Paint Mode: ${state.drawing.paintEnabled ? "ON" : "OFF"} (<kbd>E</kbd>)</span>
        </button>
        <button id="btnDrawNewLayer" type="button" class="btn btn-primary btn-sm" title="Create a new drawing / VOI layer aligned to Reference">${"+"} New Layer</button>
        <button id="btnDrawImportVoi" type="button" class="btn btn-secondary btn-sm" title="Import .voi / .nii / .nii.gz mask as editable drawing layer">Import .voi</button>
      </div>

      <div class="draw-grid-badge">
        <span>Grid: ${escapeDrawHtml(baseLabel)}</span>
        <span>${baseDims ? `${baseDims[0]}×${baseDims[1]}×${baseDims[2]}` : "No Ref Grid"}</span>
      </div>

      <div class="draw-section-card">
        <div class="draw-section-header">
          <span>Drawing Layers (${state.drawing.layers.length})</span>
          ${
            activeLayer
              ? `<span style="color:#bae6fd;font-weight:600;text-transform:none;">${activeLayer.voxelCount} vox (${activeLayer.volumeCc.toFixed(2)} cm³)</span>`
              : ""
          }
        </div>
        <div class="draw-layer-list">${layerRowsHtml}</div>
        ${
          activeLayer
            ? `
          <div class="draw-inline-row" style="justify-content:space-between;margin-top:2px;">
            <span style="color:var(--text-muted);">Label Mode:</span>
            <div class="draw-segmented">
              <button type="button" class="draw-seg-btn ${activeLayer.mode === "binary" ? "active" : ""}" data-draw-mode="binary">Binary (0/1)</button>
              <button type="button" class="draw-seg-btn ${activeLayer.mode === "multilabel" ? "active" : ""}" data-draw-mode="multilabel">Multi-Label (1–6)</button>
            </div>
          </div>
          ${activeColorHtml}
        `
            : ""
        }
      </div>

      <div class="draw-section-card">
        <div class="draw-section-header">
          <span>Brush &amp; Segmentation Tools</span>
          <label style="display:inline-flex;align-items:center;gap:4px;font-weight:500;text-transform:none;cursor:pointer;">
            <input type="checkbox" id="chkDrawOverwrite" ${state.drawing.overwriteLabels ? "checked" : ""} />
            <span>Overwrite</span>
          </label>
        </div>

        <div class="draw-tool-grid">
          ${DRAW_TOOL_DEFS.map(
            (t) => `
              <button type="button" class="draw-tool-btn ${state.drawing.tool === t.id ? "active" : ""}"
                data-draw-tool="${t.id}" title="${t.title}">
                <span style="font-size:13px;line-height:1;">${t.icon}</span>
                <span>${t.label}</span>
              </button>
            `
          ).join("")}
        </div>

        <div class="draw-brush-row">
          <div class="draw-size-pills" title="Brush Radius (Shortcuts: [ / ])">
            <span style="color:var(--text-muted);align-self:center;margin-right:2px;">Size:</span>
            ${[1, 2, 3, 4, 5]
              .map(
                (sz) =>
                  `<button type="button" class="draw-size-pill ${
                    state.drawing.brushRadius === sz ? "active" : ""
                  }" data-draw-size="${sz}">${sz}</button>`
              )
              .join("")}
          </div>
          <div class="draw-segmented">
            <button type="button" class="draw-seg-btn ${!state.drawing.brush3D ? "active" : ""}" data-draw-3d="0" title="Paint on active 2D slice only">2D Disk</button>
            <button type="button" class="draw-seg-btn ${state.drawing.brush3D ? "active" : ""}" data-draw-3d="1" title="Paint as a 3D spherical brush across neighboring slices">3D Sphere</button>
          </div>
        </div>

        ${
          activeLayer
            ? `
          <div class="draw-inline-row" style="justify-content:space-between;">
            <span style="color:var(--text-muted);">Layer α:</span>
            <input type="range" id="sliderDrawLayerOpacity" min="0" max="100" value="${Math.round(
              activeLayer.opacity * 100
            )}" style="flex:1;" />
            <span id="valDrawLayerOpacity" class="opacity-pct">${Math.round(activeLayer.opacity * 100)}%</span>
          </div>
        `
            : ""
        }

        <div class="draw-region-grow-card" ${isGrowTool ? "" : "hidden"}>
          <div class="draw-section-header" style="color:#7dd3fc;">
            <span>Region Grow Settings</span>
            <div class="draw-segmented">
              <button type="button" class="draw-seg-btn ${state.drawing.regionGrow.scope === "3d" ? "active" : ""}" data-draw-rg-scope="3d">3D Vol</button>
              <button type="button" class="draw-seg-btn ${state.drawing.regionGrow.scope === "2d" ? "active" : ""}" data-draw-rg-scope="2d">2D Slice</button>
            </div>
          </div>
          <div class="draw-inline-row">
            <span style="color:var(--text-muted);">Source:</span>
            <select id="selectDrawRgSource" class="select-compact" style="flex:1;">
              ${
                nonDrawingVols.length === 0
                  ? `<option value="">No source volume</option>`
                  : nonDrawingVols
                      .map((v) => {
                        const m = state.layerMeta.get(v.id);
                        return `<option value="${v.id}" ${v.id === rgSrcId ? "selected" : ""}>${escapeDrawHtml(
                          m?.label || v.name
                        )}</option>`;
                      })
                      .join("")
              }
            </select>
          </div>
          <div class="draw-inline-row">
            <span style="color:var(--text-muted);width:64px;">Tol: <strong id="valDrawRgTol">${state.drawing.regionGrow.tolerancePct}%</strong></span>
            <input type="range" id="sliderDrawRgTol" min="2" max="55" step="1" value="${state.drawing.regionGrow.tolerancePct}" style="flex:1;" />
          </div>
          <div class="draw-inline-row">
            <span style="color:var(--text-muted);width:64px;">Max: <strong id="valDrawRgRad">${state.drawing.regionGrow.maxRadiusMm}mm</strong></span>
            <input type="range" id="sliderDrawRgRad" min="5" max="80" step="1" value="${state.drawing.regionGrow.maxRadiusMm}" style="flex:1;" />
          </div>
          <div style="font-size:10px;color:#bae6fd;">Turn Paint Mode ON &amp; click inside structure on any 2D slice.</div>
        </div>

        <div class="draw-actions-grid" style="grid-template-columns:repeat(4,minmax(0,1fr));">
          <button id="btnDrawUndo" type="button" class="btn-xs" ${
            activeLayer?.undoStack?.length ? "" : "disabled"
          } title="Undo last stroke (Cmd/Ctrl+Z)">↺ Undo</button>
          <button id="btnDrawRedo" type="button" class="btn-xs" ${
            activeLayer?.redoStack?.length ? "" : "disabled"
          } title="Redo (Cmd/Ctrl+Shift+Z)">↻ Redo</button>
          <button id="btnDrawClearSlice" type="button" class="btn-xs" ${
            activeLayer ? "" : "disabled"
          } title="Clear active slice on this layer">Clear Slice</button>
          <button id="btnDrawClearLayer" type="button" class="btn-xs btn-danger-outline" ${
            activeLayer ? "" : "disabled"
          } title="Clear all voxels on this layer">Clear All</button>
        </div>
      </div>

      <div class="draw-section-card">
        <div class="draw-section-header">
          <span>3D Cleanup &amp; Interpolation</span>
          <select id="selectDrawInterpAxis" class="select-compact" title="Interpolation slice axis">
            <option value="auto" ${state.drawing.interpAxis === "auto" ? "selected" : ""}>Axis: Auto</option>
            <option value="2" ${state.drawing.interpAxis === "2" ? "selected" : ""}>Axial (Z)</option>
            <option value="1" ${state.drawing.interpAxis === "1" ? "selected" : ""}>Coronal (Y)</option>
            <option value="0" ${state.drawing.interpAxis === "0" ? "selected" : ""}>Sagittal (X)</option>
          </select>
        </div>
        <div class="draw-actions-grid">
          <button id="btnDrawInterpSlices" type="button" class="btn btn-secondary btn-sm" ${
            activeLayer ? "" : "disabled"
          } title="Smoothly interpolate between painted key slices using signed distance">Interpolate Slices</button>
          <button id="btnDrawFillHoles" type="button" class="btn btn-secondary btn-sm" ${
            activeLayer ? "" : "disabled"
          } title="Fill enclosed 3D interior holes">Fill 3D Holes</button>
          <button id="btnDrawLargestIsland" type="button" class="btn btn-secondary btn-sm" ${
            activeLayer ? "" : "disabled"
          } title="Keep only the largest 3D connected component">Largest Island</button>
          <div style="display:flex;gap:4px;">
            <button id="btnDrawDilate" type="button" class="btn btn-secondary btn-sm" style="flex:1;" ${
              activeLayer ? "" : "disabled"
            } title="Expand ROI boundary by 1 voxel">Dilate +1</button>
            <button id="btnDrawErode" type="button" class="btn btn-secondary btn-sm" style="flex:1;" ${
              activeLayer ? "" : "disabled"
            } title="Shrink ROI boundary by 1 voxel">Erode -1</button>
          </div>
        </div>
      </div>

      <div class="draw-section-card">
        <div class="draw-section-header"><span>Save / Export VOI &amp; NIfTI</span></div>
        <div class="draw-export-row">
          <input id="drawExportFileInput" type="text" class="draw-export-name-input"
            data-for-layer-id="${escapeDrawHtml(activeLayer?.id || "")}"
            value="${escapeDrawHtml(keepCustomStem)}" placeholder="patient_VOI" />
          <select id="selectDrawExportFormat" class="select-compact" title="Export File Format">
            <option value="voi" ${state.drawing.exportFormat === "voi" ? "selected" : ""}>.voi (MRIcroGL)</option>
            <option value="nii.gz" ${state.drawing.exportFormat === "nii.gz" ? "selected" : ""}>.nii.gz</option>
            <option value="nii" ${state.drawing.exportFormat === "nii" ? "selected" : ""}>.nii</option>
          </select>
        </div>
        <div class="draw-export-row">
          <button id="btnDrawSaveAs" type="button" class="btn btn-primary btn-sm" style="flex:1;" ${
            activeLayer ? "" : "disabled"
          } title="Choose destination folder & filename (local or mounted network share)">Save As...</button>
          <button id="btnDrawDownload" type="button" class="btn btn-secondary btn-sm" ${
            activeLayer ? "" : "disabled"
          } title="Download file immediately">Download</button>
          <button id="btnDrawMergeExport" type="button" class="btn btn-outline btn-sm" ${
            state.drawing.layers.some((l) => l.visible) ? "" : "disabled"
          } title="Merge all visible drawing layers into one multi-label file + labels JSON">Merge Visible</button>
        </div>
      </div>
    </div>
  `;

  bindDrawingStudioEvents(container);
}

function bindDrawingStudioEvents(container) {
  if (!container || container._drawDelegated) return;
  container._drawDelegated = true;

  container.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (btn) {
      if (btn.id === "btnDrawPaintToggle") {
        setDrawingPaintEnabled();
        return;
      }
      if (btn.id === "btnDrawNewLayer") {
        createDrawingLayer();
        return;
      }
      if (btn.id === "btnDrawImportVoi") {
        document.getElementById("voiImportFileInput")?.click();
        return;
      }
      if (btn.id === "btnDrawUndo") {
        undoDrawingLayer();
        return;
      }
      if (btn.id === "btnDrawRedo") {
        redoDrawingLayer();
        return;
      }
      if (btn.id === "btnDrawClearSlice") {
        clearSliceOnActiveDrawingLayer();
        return;
      }
      if (btn.id === "btnDrawClearLayer") {
        clearActiveDrawingLayer();
        return;
      }
      if (btn.id === "btnDrawInterpSlices") {
        runDrawingMorphOp("interpolate_slices");
        return;
      }
      if (btn.id === "btnDrawFillHoles") {
        runDrawingMorphOp("fill_holes");
        return;
      }
      if (btn.id === "btnDrawLargestIsland") {
        runDrawingMorphOp("largest_island");
        return;
      }
      if (btn.id === "btnDrawDilate") {
        runDrawingMorphOp("dilate");
        return;
      }
      if (btn.id === "btnDrawErode") {
        runDrawingMorphOp("erode");
        return;
      }
      if (btn.id === "btnDrawSaveAs" || btn.id === "btnDrawDownload") {
        const stem = container.querySelector("#drawExportFileInput")?.value?.trim() || null;
        const fmt = container.querySelector("#selectDrawExportFormat")?.value || state.drawing.exportFormat;
        exportDrawingLayer(state.drawing.activeLayerId, btn.id === "btnDrawSaveAs" ? "save_as" : "download", fmt, stem);
        return;
      }
      if (btn.id === "btnDrawMergeExport") {
        const fmt = container.querySelector("#selectDrawExportFormat")?.value || state.drawing.exportFormat;
        mergeAndExportVisibleDrawingLayers("save_as", fmt);
        return;
      }

      const delId = btn.getAttribute("data-draw-layer-del");
      if (delId) {
        deleteDrawingLayer(delId);
        return;
      }
      const modeVal = btn.getAttribute("data-draw-mode");
      if (modeVal && state.drawing.activeLayerId) {
        setDrawingLayerMode(state.drawing.activeLayerId, modeVal);
        return;
      }
      const swatchHex = btn.getAttribute("data-draw-color");
      if (swatchHex && state.drawing.activeLayerId) {
        setDrawingLayerColor(state.drawing.activeLayerId, swatchHex);
        return;
      }
      const toolVal = btn.getAttribute("data-draw-tool");
      if (toolVal) {
        state.drawing.tool = toolVal;
        syncDrawActiveBanner();
        renderDrawingStudioUI();
        return;
      }
      const sizeVal = btn.getAttribute("data-draw-size");
      if (sizeVal) {
        state.drawing.brushRadius = Math.max(1, Math.min(5, parseInt(sizeVal, 10) || 1));
        syncDrawActiveBanner();
        renderDrawingStudioUI();
        return;
      }
      const d3Val = btn.getAttribute("data-draw-3d");
      if (d3Val !== null) {
        state.drawing.brush3D = d3Val === "1";
        syncDrawActiveBanner();
        renderDrawingStudioUI();
        return;
      }
      const rgScope = btn.getAttribute("data-draw-rg-scope");
      if (rgScope) {
        state.drawing.regionGrow.scope = rgScope === "2d" ? "2d" : "3d";
        renderDrawingStudioUI();
        return;
      }
    }

    const pickLblRow = e.target.closest("[data-draw-pick-label]");
    if (pickLblRow && !e.target.matches("input")) {
      const lblId = parseInt(pickLblRow.getAttribute("data-draw-pick-label"), 10);
      const act = getActiveDrawingLayer();
      if (act && Number.isFinite(lblId)) {
        act.activeLabel = lblId;
        syncDrawActiveBanner();
        renderDrawingStudioUI();
      }
      return;
    }

    const layerItem = e.target.closest("[data-draw-select-layer]");
    if (layerItem && !e.target.matches("input,button")) {
      selectActiveDrawingLayer(layerItem.getAttribute("data-draw-select-layer"));
    }
  });

  container.addEventListener("change", (e) => {
    const t = e.target;
    const visId = t.getAttribute("data-draw-layer-vis");
    if (visId) {
      setDrawingLayerVisibility(visId, t.checked);
      return;
    }
    const renId = t.getAttribute("data-draw-layer-rename");
    if (renId) {
      renameDrawingLayer(renId, t.value);
      return;
    }
    const lblNameId = t.getAttribute("data-draw-label-name");
    if (lblNameId) {
      const act = getActiveDrawingLayer();
      const idx = parseInt(lblNameId, 10) - 1;
      if (act && act.labelPalette?.[idx]) {
        act.labelPalette[idx].name = t.value.trim() || `ROI ${idx + 1}`;
        const volIdx = getVolumeIndexById(act.volId);
        if (volIdx >= 0) applyDrawingLayerColormap(act, state.nv.volumes[volIdx]);
      }
      return;
    }
    if (t.id === "drawCustomColorInput" && state.drawing.activeLayerId) {
      setDrawingLayerColor(state.drawing.activeLayerId, t.value);
      return;
    }
    if (t.id === "chkDrawOverwrite") {
      state.drawing.overwriteLabels = Boolean(t.checked);
      return;
    }
    if (t.id === "selectDrawRgSource") {
      state.drawing.regionGrow.sourceVolId = t.value || null;
      return;
    }
    if (t.id === "selectDrawInterpAxis") {
      state.drawing.interpAxis = t.value || "auto";
      return;
    }
    if (t.id === "selectDrawExportFormat") {
      state.drawing.exportFormat = t.value || "voi";
      return;
    }
  });

  container.addEventListener("input", (e) => {
    const t = e.target;
    if (t.id === "sliderDrawLayerOpacity" && state.drawing.activeLayerId) {
      const val = parseFloat(t.value) / 100;
      const pctEl = container.querySelector("#valDrawLayerOpacity");
      if (pctEl) pctEl.textContent = `${Math.round(val * 100)}%`;
      setDrawingLayerOpacity(state.drawing.activeLayerId, val);
      return;
    }
    if (t.id === "sliderDrawRgTol") {
      state.drawing.regionGrow.tolerancePct = parseInt(t.value, 10) || 15;
      const vEl = container.querySelector("#valDrawRgTol");
      if (vEl) vEl.textContent = `${state.drawing.regionGrow.tolerancePct}%`;
      return;
    }
    if (t.id === "sliderDrawRgRad") {
      state.drawing.regionGrow.maxRadiusMm = parseInt(t.value, 10) || 35;
      const vEl = container.querySelector("#valDrawRgRad");
      if (vEl) vEl.textContent = `${state.drawing.regionGrow.maxRadiusMm}mm`;
    }
  });
}

function openDrawingStudioPopout() {
  if (!state.drawing.popout) {
    const wrapperRect = document.getElementById("canvasWrapper")?.getBoundingClientRect();
    const initX = wrapperRect ? Math.max(16, Math.round(wrapperRect.right - 368)) : 820;
    const initY = wrapperRect ? Math.max(64, Math.round(wrapperRect.top + 12)) : 74;
    state.drawing.popout = createPopoutWindow({
      id: "drawing-studio",
      title: "Drawing & VOI Studio",
      width: 356,
      height: 615,
      x: initX,
      y: initY,
      minWidth: 316,
      minHeight: 400,
      autoOpen: true,
      onClose: () => {
        if (state.drawing.paintEnabled) {
          setDrawingPaintEnabled(false);
        }
        syncDrawHeaderButton();
      },
      onMinimize: () => {
        syncDrawHeaderButton();
      },
    });
  } else {
    state.drawing.popout.open();
  }
  renderDrawingStudioUI();
  syncDrawHeaderButton();
  return state.drawing.popout;
}

function toggleDrawingStudioPopout(forceOpen = null) {
  const existing = state.drawing.popout;
  const currentlyOpen = Boolean(existing && existing.isOpen() && !existing.isMinimized());
  const nextOpen = forceOpen !== null ? Boolean(forceOpen) : !currentlyOpen;
  if (nextOpen) {
    openDrawingStudioPopout();
  } else if (existing) {
    existing.close();
  }
}

// ============================================================================
// Floating Brain Atlas Studio Pop-Out (Multi-Atlas, THOMAS/Tian/FSL, ROI Browser)
// ============================================================================
function syncAtlasHeaderButton() {
  const btn = document.getElementById("btnToggleAtlasStudio");
  if (!btn) return;
  const pop = state.atlasStudioPopout;
  const active = Boolean(pop && pop.isOpen() && !pop.isMinimized());
  btn.classList.toggle("active", active);
  btn.setAttribute("aria-expanded", String(active));
}

function renderAtlasStudioCrosshairFooter() {
  const pop = state.atlasStudioPopout;
  if (!pop || !pop.isOpen()) return;
  const listEl = pop.bodyEl.querySelector("#atlasStudioStackedList");
  const rasEl = pop.bodyEl.querySelector("#atlasStudioRasLabel");
  if (!listEl) return;
  const ras = state.lastCrosshairRAS || [0, 0, 0];
  if (rasEl) {
    rasEl.textContent = `RAS [${ras[0].toFixed(1)}, ${ras[1].toFixed(1)}, ${ras[2].toFixed(1)}] mm`;
  }
  const hits = lookupAllAtlasesAtRAS(ras);
  if (!hits.length) {
    listEl.innerHTML = `<div style="color:var(--text-muted);font-size:10.5px;">${
      getLoadedAtlasVolumes().length ? "Crosshair outside loaded atlas labels." : "No atlas loaded yet."
    }</div>`;
    return;
  }
  listEl.innerHTML = hits
    .map(
      (h) => `
        <div class="atlas-stacked-row" title="${escapeDrawHtml(h.name)} (#${Number(h.labelId)})">
          <span class="atlas-region-swatch" style="background:${sanitizeCssHex(h.hex)};"></span>
          <span class="atlas-stacked-badge">${escapeDrawHtml(h.atlasBadge)}</span>
          <span style="flex:1;min-width:0;color:#f8fafc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeDrawHtml(h.shortName || h.name)}</span>
          <span style="color:var(--text-muted);font-family:var(--font-mono);font-size:10px;">#${Number(h.labelId)}</span>
        </div>
      `
    )
    .join("");
}

function buildAtlasStudioDOM(bodyEl) {
  bodyEl.className = "popout-body atlas-studio-body";
  bodyEl.innerHTML = `
    <div class="atlas-studio-section">
      <div class="atlas-studio-sec-hdr">
        <span>Loaded Atlases (Single Overlay / Stacked Hover)</span>
        <div id="atlasStudioStyleGroup" class="atlas-style-group" role="group" aria-label="Overlay Render Style">
          <button type="button" class="atlas-style-btn active" data-atlas-style="filled">Filled</button>
          <button type="button" class="atlas-style-btn" data-atlas-style="filled_outline">+Edge</button>
          <button type="button" class="atlas-style-btn" data-atlas-style="outline">Edge</button>
        </div>
      </div>
      <div id="atlasStudioLoadedList" class="atlas-studio-list"></div>
    </div>

    <div class="atlas-studio-section">
      <div class="atlas-studio-sec-hdr">
        <span>Generate / Warp Atlas</span>
        <span id="atlasStudioCachePill" class="atlas-cache-pill is-idle" title="Per-T1w MNI152 transform cache status">Cache Idle</span>
      </div>
      <div class="atlas-import-grid">
        <select id="atlasStudioGenSelect" class="select-full" title="Choose catalog atlas"></select>
        <div class="atlas-import-row">
          <select id="atlasStudioGenT1" class="select-full" style="flex:1;" title="Target T1w MRI"></select>
          <select id="atlasStudioGenMode" class="select-full" style="flex:1;" title="Registration / segmentation pipeline">
            <option value="fnirt_nonlinear">FNIRT Nonlinear (Cached)</option>
            <option value="flirt_12dof">FLIRT 12-DOF (~2s)</option>
          </select>
        </div>
        <div id="atlasStudioHipsOpts" class="hips-options-box" style="margin-bottom:0;" hidden>
          <div id="atlasStudioHipsFlagsRow" class="atlas-import-grid">
            <div class="hips-options-row">
              <label for="atlasStudioHipsContrast">Contrast:</label>
              <select id="atlasStudioHipsContrast" class="select-compact" style="flex:1;">
                <option value="t1" selected>Standard T1w (-t1)</option>
                <option value="wmn">WMn / FGATIR (Native)</option>
              </select>
            </div>
            <div class="hips-options-row" style="justify-content:space-between;">
              <label class="hips-check-label">
                <input type="checkbox" id="atlasStudioHipsForce" />
                <span>Force Recompute</span>
              </label>
              <span id="atlasStudioAutoDmBadge" class="hips-dm-badge" hidden>Auto -dm</span>
            </div>
          </div>
          <div class="hips-options-row">
            <input id="atlasStudioHipsCustomImage" type="text" class="hips-custom-img-input"
              placeholder="Custom Docker image (auto-detect if blank)"
              title="Optional local HIPS-THOMAS image name/tag (e.g. anagrammarian/sthomas:latest)" />
            <button id="atlasStudioBtnCheckHips" type="button" class="btn-xs" title="Check Docker with custom image">Check</button>
          </div>
        </div>
        <button id="atlasStudioBtnGenerate" type="button" class="btn btn-dkt-seg btn-sm">Generate Atlas in Subject Space</button>
        <div id="atlasStudioJobProgress" class="hips-job-strip" style="margin-top:2px;" hidden>
          <div class="hips-job-top">
            <span id="atlasStudioJobStage" class="hips-job-stage">Running HIPS-THOMAS...</span>
            <span id="atlasStudioJobElapsed" class="hips-job-elapsed">00:00</span>
            <button id="atlasStudioBtnCancelJob" type="button" class="btn-xs btn-danger-outline">Cancel</button>
          </div>
          <div id="atlasStudioJobLog" class="hips-job-log">Starting container...</div>
        </div>
      </div>
    </div>

    <div class="atlas-studio-section">
      <div class="atlas-studio-sec-hdr">
        <span>Import HIPS-THOMAS / Custom Atlas + LUT</span>
      </div>
      <div class="atlas-import-grid">
        <div class="atlas-import-row">
          <label class="btn btn-secondary btn-sm" style="flex:1;justify-content:center;" title="Select single sthomas_LR_labels.nii.gz or both thomasfull_L.nii.gz + thomasfull_R.nii.gz or per-nucleus files">
            + HIPS-THOMAS Files
            <input id="atlasStudioHipsInput" type="file" accept=".nii,.nii.gz" multiple hidden />
          </label>
          <label class="btn btn-secondary btn-sm" style="flex:1;justify-content:center;" title="Select an entire HIPS-THOMAS output folder (left/ &amp; right/)">
            + HIPS Folder
            <input id="atlasStudioHipsFolderInput" type="file" webkitdirectory directory multiple hidden />
          </label>
        </div>
        <div class="atlas-import-row">
          <label class="btn btn-secondary btn-xs" title="Select custom atlas (.nii, .nii.gz, .mgz)">
            Atlas NIfTI
            <input id="atlasStudioCustomNifti" type="file" accept=".nii,.nii.gz,.mgz,.mgh" hidden />
          </label>
          <span id="atlasStudioCustomNiftiName" class="atlas-import-fname">None</span>
          <label class="btn btn-secondary btn-xs" title="Optional LUT (.json, .csv, .tsv, .txt, .ctbl, .xml)">
            + LUT
            <input id="atlasStudioCustomLut" type="file" accept=".json,.csv,.tsv,.txt,.ctbl,.xml" hidden />
          </label>
          <span id="atlasStudioCustomLutName" class="atlas-import-fname">Auto</span>
        </div>
        <div class="atlas-import-row">
          <select id="atlasStudioCustomSpace" class="select-compact" style="flex:1;">
            <option value="subject" selected>Space: Subject Native</option>
            <option value="mni152_fnirt">Space: MNI152 -&gt; Subject (FNIRT)</option>
            <option value="mni152_flirt">Space: MNI152 -&gt; Subject (FLIRT)</option>
          </select>
          <button id="atlasStudioBtnImportCustom" type="button" class="btn btn-primary btn-xs">Import Custom</button>
        </div>
      </div>
    </div>

    <div class="atlas-studio-section">
      <div class="atlas-studio-sec-hdr">
        <span id="atlasStudioRegionsTitle">Regions / Nuclei Browser</span>
        <div style="display:flex;gap:4px;">
          <button id="atlasStudioBtnAllRois" type="button" class="btn-xs" title="Show all regions">All</button>
          <button id="atlasStudioBtnClearRois" type="button" class="btn-xs" title="Clear isolation to 0 visible ROIs before isolating one">None</button>
        </div>
      </div>
      <div class="atlas-region-toolbar">
        <input id="atlasStudioSearch" type="search" class="atlas-search-input" placeholder="Search nuclei / ROI..." />
        <div class="atlas-hemi-group" role="group" aria-label="Hemisphere filter">
          <button type="button" class="atlas-hemi-btn active" data-hemi="all">All</button>
          <button type="button" class="atlas-hemi-btn" data-hemi="L">L</button>
          <button type="button" class="atlas-hemi-btn" data-hemi="R">R</button>
        </div>
      </div>
      <div id="atlasStudioRegionList" class="atlas-region-list"></div>
      <button id="atlasStudioBtnCopyToDraw" type="button" class="btn btn-secondary btn-sm"
        title="Copy isolated ROI(s) (or region under crosshair) into an editable Drawing Layer for .voi / NIfTI export">
        Copy Selected ROI(s) to Drawing Layer (.voi)
      </button>
    </div>

    <div class="atlas-studio-section">
      <div class="atlas-studio-sec-hdr">
        <span>Crosshair Stacked Hit</span>
        <span id="atlasStudioRasLabel" style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);">RAS [0, 0, 0]</span>
      </div>
      <div id="atlasStudioStackedList" class="atlas-stacked-list"></div>
    </div>
  `;

  bodyEl.querySelector("#atlasStudioStyleGroup")?.addEventListener("click", (e) => {
    const b = e.target.closest(".atlas-style-btn[data-atlas-style]");
    if (!b) return;
    setAtlasOverlayStyle(b.getAttribute("data-atlas-style"));
  });

  bodyEl.querySelector("#atlasStudioGenSelect")?.addEventListener("change", (e) => {
    const val = e.target.value;
    const sidebarSel = document.getElementById("selectAtlasType");
    if (sidebarSel && Array.from(sidebarSel.options).some((o) => o.value === val)) {
      sidebarSel.value = val;
      syncAtlasTypeSelectionUI(false, false);
    }
    syncAtlasStudioUI();
  });

  bodyEl.querySelector("#atlasStudioGenMode")?.addEventListener("change", (e) => {
    const targetAid = String(bodyEl.querySelector("#atlasStudioGenSelect")?.value || state.selectedAtlasType || "thomas").toLowerCase();
    const val = e.target.value;
    if (val) {
      if (!state.atlasPreferredModeById) state.atlasPreferredModeById = new Map();
      state.atlasPreferredModeById.set(targetAid, val);
    }
    const sideSel = document.getElementById("selectAtlasType");
    const sideMode = document.getElementById("selectDktMode");
    if (sideMode && sideSel?.value === targetAid && Array.from(sideMode.options).some((o) => o.value === val && !o.disabled)) {
      sideMode.value = val;
      syncAtlasTypeSelectionUI(true, false);
    } else {
      syncAtlasStudioUI();
    }
  });

  bodyEl.querySelector("#atlasStudioBtnGenerate")?.addEventListener("click", async () => {
    const aid = bodyEl.querySelector("#atlasStudioGenSelect")?.value || "thomas";
    const t1Id = bodyEl.querySelector("#atlasStudioGenT1")?.value || state.refVolId;
    const mode = bodyEl.querySelector("#atlasStudioGenMode")?.value || null;
    await runAtlasGeneration(aid, mode, t1Id);
  });

  bodyEl.querySelector("#atlasStudioGenT1")?.addEventListener("change", () => {
    syncAtlasStudioUI();
  });
  bodyEl.querySelector("#atlasStudioHipsContrast")?.addEventListener("change", (e) => {
    state.atlasStudioState.hipsContrast = e.target.value === "wmn" ? "wmn" : "t1";
    syncHipsThomasOptionsUI();
  });
  bodyEl.querySelector("#atlasStudioHipsForce")?.addEventListener("change", (e) => {
    state.atlasStudioState.hipsForceRecompute = Boolean(e.target.checked);
    syncHipsThomasOptionsUI();
  });
  bodyEl.querySelector("#atlasStudioHipsCustomImage")?.addEventListener("change", (e) => {
    const val = String(e.target.value || "").trim();
    state.atlasStudioState.hipsCustomImage = val;
    persistHipsCustomImage(val);
    checkNeuroToolsStatus(true);
  });
  bodyEl.querySelector("#atlasStudioBtnCheckHips")?.addEventListener("click", () => {
    const inp = bodyEl.querySelector("#atlasStudioHipsCustomImage");
    const val = String(inp?.value || "").trim();
    state.atlasStudioState.hipsCustomImage = val;
    persistHipsCustomImage(val);
    checkNeuroToolsStatus(true);
  });
  bodyEl.querySelector("#atlasStudioBtnCancelJob")?.addEventListener("click", () => {
    cancelActiveAtlasJob();
  });

  bodyEl.querySelector("#atlasStudioHipsInput")?.addEventListener("change", async (e) => {
    const files = e.target.files;
    await importHipsThomasFiles(files);
    e.target.value = "";
  });
  bodyEl.querySelector("#atlasStudioHipsFolderInput")?.addEventListener("change", async (e) => {
    const files = e.target.files;
    await importHipsThomasFiles(files);
    e.target.value = "";
  });

  bodyEl.querySelector("#atlasStudioCustomNifti")?.addEventListener("change", (e) => {
    const f = e.target.files?.[0] || null;
    state.atlasStudioState.customNiftiFile = f;
    const lbl = bodyEl.querySelector("#atlasStudioCustomNiftiName");
    if (lbl) lbl.textContent = f ? f.name : "None";
  });
  bodyEl.querySelector("#atlasStudioCustomLut")?.addEventListener("change", (e) => {
    const f = e.target.files?.[0] || null;
    state.atlasStudioState.customLutFile = f;
    const lbl = bodyEl.querySelector("#atlasStudioCustomLutName");
    if (lbl) lbl.textContent = f ? f.name : "Auto";
  });
  bodyEl.querySelector("#atlasStudioBtnImportCustom")?.addEventListener("click", async () => {
    await importCustomAtlasFromFiles({
      niftiFile: state.atlasStudioState.customNiftiFile,
      lutFile: state.atlasStudioState.customLutFile,
      space: bodyEl.querySelector("#atlasStudioCustomSpace")?.value || "subject",
      refVolId: bodyEl.querySelector("#atlasStudioGenT1")?.value || state.refVolId,
    });
  });

  bodyEl.querySelector("#atlasStudioSearch")?.addEventListener("input", (e) => {
    state.atlasStudioState.search = String(e.target.value || "");
    syncAtlasStudioUI();
  });

  bodyEl.querySelectorAll(".atlas-hemi-btn[data-hemi]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.atlasStudioState.hemi = btn.getAttribute("data-hemi") || "all";
      syncAtlasStudioUI();
    });
  });

  bodyEl.querySelector("#atlasStudioBtnAllRois")?.addEventListener("click", async () => {
    const selVol = getSelectedStudioAtlasVolume();
    if (selVol) await setAtlasVisibleRegions(selVol.id, null);
  });
  bodyEl.querySelector("#atlasStudioBtnClearRois")?.addEventListener("click", async () => {
    const selVol = getSelectedStudioAtlasVolume();
    if (selVol) await setAtlasVisibleRegions(selVol.id, new Set());
  });
  bodyEl.querySelector("#atlasStudioBtnCopyToDraw")?.addEventListener("click", async () => {
    const selVol = getSelectedStudioAtlasVolume();
    if (selVol) await copySelectedAtlasRoisToDrawingLayer(selVol.id);
  });
}

function getSelectedStudioAtlasVolume() {
  const vols = getLoadedAtlasVolumes();
  if (!vols.length) return null;
  const wantId = state.atlasStudioState.selectedAtlasVolId || state.activeAtlasVolId || state.activeDktVolId;
  return vols.find((v) => v.id === wantId) || vols[0];
}

function syncAtlasStudioUI() {
  syncOverlayStyleButtonsUI();
  const pop = state.atlasStudioPopout;
  if (!pop || !pop.el) return;
  const bodyEl = pop.bodyEl;

  // 1. Loaded atlases list
  const loadedList = bodyEl.querySelector("#atlasStudioLoadedList");
  const loadedVols = getLoadedAtlasVolumes();
  const activeOverlayId = state.activeAtlasVolId || state.activeDktVolId;
  const selVol = getSelectedStudioAtlasVolume();
  if (selVol) state.atlasStudioState.selectedAtlasVolId = selVol.id;

  if (loadedList) {
    if (!loadedVols.length) {
      loadedList.innerHTML = `<div class="draw-empty-note">No atlases loaded. Generate THOMAS, Tian, FSL, or DKT below.</div>`;
    } else {
      loadedList.innerHTML = "";
      for (const vol of loadedVols) {
        const meta = state.layerMeta.get(vol.id);
        const badge = getAtlasBadgeInfo(meta);
        const isActive = vol.id === activeOverlayId && Boolean(meta?.visible);
        const isInspected = selVol && vol.id === selVol.id;
        const pct = Math.round((meta?.userAtlasOpacity ?? 0.48) * 100);
        const card = document.createElement("div");
        card.className = `atlas-loaded-card ${isInspected ? "is-active" : ""}`;
        card.innerHTML = `
          <div class="atlas-loaded-top">
            <input type="checkbox" role="switch" class="vis-switch" ${isActive ? "checked" : ""} title="Show this atlas overlay on slices (others stay loaded for stacked hover)" />
            <span class="role-badge dkt ${escapeDrawHtml(badge.cls)}">${escapeDrawHtml(badge.text)}</span>
            <span class="atlas-loaded-title" title="${escapeDrawHtml(meta?.atlasName || vol.name)}">${escapeDrawHtml(meta?.atlasShortName || vol.name)}</span>
            <button type="button" class="icon-btn" title="Remove atlas layer">✕</button>
          </div>
          <div class="atlas-loaded-controls">
            <span>α</span>
            <input type="range" min="0" max="100" value="${pct}" />
            <span class="opacity-pct">${pct}%</span>
            <span style="margin-left:auto;">${(meta?.regions || []).length} ROIs</span>
          </div>
        `;
        card.addEventListener("click", (ev) => {
          if (ev.target.closest("input,button")) return;
          state.atlasStudioState.selectedAtlasVolId = vol.id;
          syncAtlasStudioUI();
        });
        const chk = card.querySelector('input[type="checkbox"]');
        chk?.addEventListener("change", (ev) => {
          state.atlasStudioState.selectedAtlasVolId = vol.id;
          activateAtlasOverlay(vol.id, Boolean(ev.target.checked));
        });
        const rng = card.querySelector('input[type="range"]');
        rng?.addEventListener("input", async (ev) => {
          const val = Math.max(0, Math.min(1, parseFloat(ev.target.value) / 100));
          if (meta) {
            meta.userAtlasOpacity = val;
            meta.savedOpacity = val > 0 ? val : meta.savedOpacity;
            meta.visible = val > 0;
          }
          if (val > 0 && vol.id !== (state.activeAtlasVolId || state.activeDktVolId)) {
            await activateAtlasOverlay(vol.id, true);
          } else {
            await applyAtlasColormapToVolume(vol);
            state.nv?.drawScene?.();
            syncDktOverlayControlsUI();
          }
          const pctSpan = card.querySelector(".opacity-pct");
          if (pctSpan) pctSpan.textContent = `${Math.round(val * 100)}%`;
        });
        const remBtn = card.querySelector("button.icon-btn");
        remBtn?.addEventListener("click", () => {
          const vi = getVolumeIndexById(vol.id);
          if (vi >= 0) {
            state.nv.removeVolumeByIndex(vi);
            state.layerMeta.delete(vol.id);
            if (state.activeDktVolId === vol.id) state.activeDktVolId = null;
            if (state.activeAtlasVolId === vol.id) state.activeAtlasVolId = null;
            const remFirst = getLoadedAtlasVolumes()[0];
            if (remFirst) activateAtlasOverlay(remFirst.id, true);
            else {
              resetDktInspectorUI();
              renderLayerList();
              syncAtlasStudioUI();
            }
          }
        });
        loadedList.appendChild(card);
      }
    }
  }

  // 2. Sync Generate selectors & warp cache status
  const genSel = bodyEl.querySelector("#atlasStudioGenSelect");
  const sidebarSel = document.getElementById("selectAtlasType");
  if (genSel && sidebarSel && genSel.options.length !== sidebarSel.options.length) {
    const prev = genSel.value || sidebarSel.value || "thomas";
    genSel.innerHTML = sidebarSel.innerHTML;
    genSel.value = Array.from(genSel.options).some((o) => o.value === prev) ? prev : "thomas";
  }
  const genT1 = bodyEl.querySelector("#atlasStudioGenT1");
  const sideT1 = document.getElementById("selectDktT1Vol");
  if (genT1 && sideT1) {
    const prevT1 = genT1.value || sideT1.value;
    genT1.innerHTML = sideT1.innerHTML;
    if (prevT1 && Array.from(genT1.options).some((o) => o.value === prevT1)) genT1.value = prevT1;
  }
  const genMode = bodyEl.querySelector("#atlasStudioGenMode");
  const targetStudioAid = String(genSel?.value || state.selectedAtlasType || "thomas").toLowerCase();
  if (genMode) {
    const opts = getAtlasPipelineModeOptions(targetStudioAid);
    genMode.innerHTML = "";
    for (const o of opts) {
      const el = document.createElement("option");
      el.value = o.v;
      el.textContent = o.l;
      if (o.disabled) el.disabled = true;
      genMode.appendChild(el);
    }
    genMode.value = resolvePreferredAtlasPipelineMode(targetStudioAid, opts);
    if (genMode.dataset) genMode.dataset.atlasId = targetStudioAid;
  }
  const cachePill = bodyEl.querySelector("#atlasStudioCachePill");
  if (cachePill) {
    if (targetStudioAid === "thomas" && genMode?.value === "hips_thomas_docker") {
      if (state.atlasStudioState.hipsCacheHit === true) {
        cachePill.textContent = "HIPS Cached (~1s)";
        cachePill.className = "atlas-cache-pill";
      } else {
        cachePill.textContent = "HIPS-THOMAS Ready";
        cachePill.className = "atlas-cache-pill";
      }
    } else if (state.atlasStudioState.warpCacheHit === true) {
      cachePill.textContent = "Warp Cached (~1s)";
      cachePill.className = "atlas-cache-pill";
    } else if (state.atlasStudioState.warpCacheHit === false) {
      cachePill.textContent = "Warp Ready & Saved";
      cachePill.className = "atlas-cache-pill";
    } else {
      cachePill.textContent = state.neuroToolsStatus?.fnirt?.available ? "FNIRT + Cache Ready" : "FLIRT Ready";
      cachePill.className = "atlas-cache-pill is-idle";
    }
  }

  // Sync Studio HIPS-THOMAS Contrast / Force Recompute / Auto -dm / Custom Docker Image
  const studioHipsBox = bodyEl.querySelector("#atlasStudioHipsOpts");
  const studioHipsFlags = bodyEl.querySelector("#atlasStudioHipsFlagsRow");
  const studioContrast = bodyEl.querySelector("#atlasStudioHipsContrast");
  const studioForce = bodyEl.querySelector("#atlasStudioHipsForce");
  const studioDmBadge = bodyEl.querySelector("#atlasStudioAutoDmBadge");
  const studioCustomImg = bodyEl.querySelector("#atlasStudioHipsCustomImage");
  if (studioHipsBox) studioHipsBox.hidden = targetStudioAid !== "thomas";
  if (studioHipsFlags) studioHipsFlags.hidden = genMode?.value !== "hips_thomas_docker";
  if (studioContrast) studioContrast.value = state.atlasStudioState.hipsContrast || "t1";
  if (studioForce) studioForce.checked = Boolean(state.atlasStudioState.hipsForceRecompute);
  if (studioDmBadge) {
    const stT1Idx = getVolumeIndexById(genT1?.value || state.refVolId);
    const stT1Vol = stT1Idx >= 0 ? state.nv?.volumes?.[stT1Idx] : null;
    studioDmBadge.hidden = !(genMode?.value === "hips_thomas_docker" && isTargetVolumeSkullStripped(stT1Vol));
  }
  if (studioCustomImg && document.activeElement !== studioCustomImg) {
    studioCustomImg.value = state.atlasStudioState.hipsCustomImage || "";
  }
  renderActiveAtlasJobUI();

  // 3. Hemisphere filter buttons & Regions browser
  bodyEl.querySelectorAll(".atlas-hemi-btn[data-hemi]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-hemi") === (state.atlasStudioState.hemi || "all"));
  });

  const regTitle = bodyEl.querySelector("#atlasStudioRegionsTitle");
  const regList = bodyEl.querySelector("#atlasStudioRegionList");
  if (regList) {
    if (!selVol) {
      if (regTitle) regTitle.textContent = "Regions / Nuclei Browser";
      regList.innerHTML = `<div class="draw-empty-note">Select or generate an atlas to browse its ROIs.</div>`;
    } else {
      const meta = state.layerMeta.get(selVol.id);
      if (!meta.regions || meta.regions.length === 0) {
        meta.regions = computeClientAtlasRegions(selVol, meta.lut || state.dktLut);
      }
      const allRegs = meta.regions || [];
      if (regTitle) {
        const isoCount = meta.visibleRegionIds instanceof Set ? meta.visibleRegionIds.size : allRegs.length;
        regTitle.textContent = `${meta.atlasShortName || "Atlas"} ROIs (${isoCount}/${allRegs.length})`;
      }
      const q = (state.atlasStudioState.search || "").trim().toLowerCase();
      const hemiFilter = state.atlasStudioState.hemi || "all";
      const filtered = allRegs.filter((r) => {
        if (hemiFilter !== "all" && r.hemisphere !== hemiFilter) return false;
        if (q && !`${r.name} ${r.shortName} ${r.id}`.toLowerCase().includes(q)) return false;
        return true;
      });
      if (!filtered.length) {
        regList.innerHTML = `<div class="draw-empty-note">No matching regions (${escapeDrawHtml(q || hemiFilter)}).</div>`;
      } else {
        regList.innerHTML = "";
        for (const r of filtered.slice(0, 250)) {
          const isChecked = !(meta.visibleRegionIds instanceof Set) || meta.visibleRegionIds.has(Number(r.id));
          const row = document.createElement("div");
          row.className = "atlas-region-row";
          row.title = `Click to jump crosshair to ${r.name} (#${r.id}, ${r.volumeMm3} mm³)`;
          row.innerHTML = `
            <input type="checkbox" ${isChecked ? "checked" : ""} title="Show/isolate ${escapeDrawHtml(r.shortName || r.name)}" />
            <span class="atlas-region-swatch" style="background:${sanitizeCssHex(r.hex)};"></span>
            <span class="atlas-region-name">${escapeDrawHtml(r.name)}</span>
            <span class="atlas-region-meta">#${Number(r.id)} · ${Math.round(r.volumeMm3)}mm³</span>
          `;
          const cb = row.querySelector('input[type="checkbox"]');
          cb?.addEventListener("click", (ev) => ev.stopPropagation());
          cb?.addEventListener("change", (ev) => {
            toggleAtlasRegionVisibility(selVol.id, r.id, Boolean(ev.target.checked));
          });
          row.addEventListener("click", () => {
            jumpToAtlasRegion(selVol.id, r.id);
          });
          regList.appendChild(row);
        }
      }
    }
  }

  renderAtlasStudioCrosshairFooter();
}

function openAtlasStudioPopout() {
  if (!state.atlasStudioPopout) {
    const wrapRect = document.getElementById("canvasWrapper")?.getBoundingClientRect();
    const defX = wrapRect ? Math.max(16, Math.round(wrapRect.right - 432)) : undefined;
    const defY = wrapRect ? Math.max(66, Math.round(wrapRect.top + 16)) : 74;
    state.atlasStudioPopout = createPopoutWindow({
      id: "atlas-studio",
      title: "Brain Atlas Studio",
      width: 416,
      height: 640,
      minWidth: 360,
      minHeight: 460,
      x: defX,
      y: defY,
      autoOpen: true,
      onClose: () => syncAtlasHeaderButton(),
      onMinimize: () => syncAtlasHeaderButton(),
    });
    buildAtlasStudioDOM(state.atlasStudioPopout.bodyEl);
  } else {
    state.atlasStudioPopout.open();
  }
  syncAtlasStudioUI();
  syncAtlasHeaderButton();
  return state.atlasStudioPopout;
}

function toggleAtlasStudioPopout(forceOpen = null) {
  const existing = state.atlasStudioPopout;
  const currentlyOpen = Boolean(existing && existing.isOpen() && !existing.isMinimized());
  const nextOpen = forceOpen !== null ? Boolean(forceOpen) : !currentlyOpen;
  if (nextOpen) openAtlasStudioPopout();
  else if (existing) existing.close();
}

function syncStrokeOverlayCanvasSize() {
  const overlay = document.getElementById("drawStrokeOverlay");
  const nvCanvas = document.getElementById("niivueCanvas");
  if (!overlay || !nvCanvas) return null;
  const cssW = nvCanvas.clientWidth || overlay.clientWidth || 0;
  const cssH = nvCanvas.clientHeight || overlay.clientHeight || 0;
  if (cssW <= 0 || cssH <= 0) return null;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const targetW = Math.round(cssW * dpr);
  const targetH = Math.round(cssH * dpr);
  if (overlay.width !== targetW || overlay.height !== targetH) {
    overlay.width = targetW;
    overlay.height = targetH;
  }
  return { overlay, cssW, cssH, dpr };
}

function clearStrokeOverlayCanvas() {
  const info = syncStrokeOverlayCanvasSize();
  if (!info) return;
  const ctx = info.overlay.getContext("2d");
  if (ctx) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, info.overlay.width, info.overlay.height);
  }
}

function computeBrushRingCssRadius(tileInfo, layer) {
  const brushSize = state.drawing.brushRadius || 1;
  if (!tileInfo?.tile || !layer) return Math.max(4, brushSize * 3);
  const xPt = tileInfo.useDevice ? tileInfo.pos.devX : tileInfo.pos.cssX;
  const yPt = tileInfo.useDevice ? tileInfo.pos.devY : tileInfo.pos.cssY;
  const lin = measureLinearRASInTile(tileInfo.tile, xPt, yPt);
  if (!lin?.jacX) return Math.max(4, brushSize * 3);
  const mmPerCssPx = Math.hypot(lin.jacX[0], lin.jacX[1], lin.jacX[2]) * (tileInfo.useDevice ? tileInfo.pos.scaleX : 1);
  if (!(mmPerCssPx > 1e-4)) return Math.max(4, brushSize * 3);
  const aff = layer.affine || identityMat4();
  const voxSize = Math.min(
    Math.hypot(aff[0][0], aff[1][0], aff[2][0]) || 1,
    Math.hypot(aff[0][1], aff[1][1], aff[2][1]) || 1,
    Math.hypot(aff[0][2], aff[1][2], aff[2][2]) || 1
  );
  const voxRad = Math.max(0.5, brushSize - 0.5);
  return Math.max(3, Math.min(180, (voxRad * voxSize) / mmPerCssPx));
}

function renderStrokeOverlay(hoverState = null, activeStroke = null) {
  const info = syncStrokeOverlayCanvasSize();
  if (!info) return;
  const ctx = info.overlay.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(info.dpr, 0, 0, info.dpr, 0, 0);
  ctx.clearRect(0, 0, info.cssW, info.cssH);

  if (activeStroke && activeStroke.screenPts?.length) {
    const pts = activeStroke.screenPts;
    const isEraser = activeStroke.effectiveTool === "eraser";
    const isFilled = activeStroke.effectiveTool === "filled_pen";
    const strokeHex = isEraser ? "#f8fafc" : activeStroke.colorHex || "#f43f5e";

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = strokeHex;
    ctx.lineWidth = isFilled ? 2 : Math.max(2, activeStroke.pxRadius * 2);
    if (isEraser) ctx.setLineDash([4, 3]);

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    if (isFilled && pts.length > 2) {
      ctx.closePath();
      const [r, g, b] = hexToRgb(strokeHex);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.28)`;
      ctx.fill();
    }
    ctx.stroke();
    ctx.restore();
  }

  if (hoverState && state.drawing.paintEnabled && !state.zoom2d.isDraggingPan) {
    const { cssX, cssY, pxRadius, colorHex, isEraser } = hoverState;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cssX, cssY, Math.max(3, pxRadius), 0, Math.PI * 2);
    ctx.strokeStyle = isEraser ? "#f8fafc" : colorHex || "#f43f5e";
    ctx.lineWidth = 1.5;
    if (isEraser) ctx.setLineDash([3, 3]);
    ctx.stroke();

    // Crisp dark outer halo so cursor ring stays legible on bright T1/CT intensities
    ctx.beginPath();
    ctx.arc(cssX, cssY, Math.max(3, pxRadius) + 1.2, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(15, 23, 42, 0.72)";
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cssX, cssY, 1.3, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.restore();
  }
}

function wireDrawingCanvasListeners() {
  const nvCanvas = document.getElementById("niivueCanvas");
  if (!nvCanvas) return;

  nvCanvas.addEventListener("contextmenu", (e) => {
    if (state.drawing.paintEnabled) {
      e.preventDefault();
    }
  });

  let activeStroke = null;

  const clientPosToLayerVoxel = (clientX, clientY, tileInfo, layer, lockSlice = null) => {
    const rect = nvCanvas.getBoundingClientRect();
    let cssX = clientX - rect.left;
    let cssY = clientY - rect.top;

    // Clamp inside tile rectangle while dragging a stroke
    if (lockSlice && tileInfo?.tile?.leftTopWidthHeight) {
      const [l, t, w, h] = tileInfo.tile.leftTopWidthHeight;
      const x0 = w < 0 ? l + w : l;
      const y0 = h < 0 ? t + h : t;
      const aw = Math.abs(w);
      const ah = Math.abs(h);
      const sx = tileInfo.useDevice ? tileInfo.pos.scaleX : 1;
      const sy = tileInfo.useDevice ? tileInfo.pos.scaleY : 1;
      const minCssX = x0 / sx + 1;
      const maxCssX = (x0 + aw) / sx - 1;
      const minCssY = y0 / sy + 1;
      const maxCssY = (y0 + ah) / sy - 1;
      cssX = Math.max(minCssX, Math.min(maxCssX, cssX));
      cssY = Math.max(minCssY, Math.min(maxCssY, cssY));
    }

    const scaleX = (nvCanvas.width || rect.width) / rect.width;
    const scaleY = (nvCanvas.height || rect.height) / rect.height;
    const devX = cssX * scaleX;
    const devY = cssY * scaleY;
    const xPt = tileInfo.useDevice ? devX : cssX;
    const yPt = tileInfo.useDevice ? devY : cssY;

    const ras =
      sampleCanvasPtRAS(devX, devY, cssX, cssY) ||
      (tileInfo.tile ? measureLinearRASInTile(tileInfo.tile, xPt, yPt)?.mm : null);
    if (!ras) return null;

    const invAff = invertMat4(layer.affine);
    const vox = applyMat4Pt(invAff, ras);
    if (!vox.every(Number.isFinite)) return null;

    const { sliceAxis, uAxis, vAxis } = lockSlice || resolveSliceVoxelAxes(layer.affine, tileInfo.axCorSag);
    const sIdx = lockSlice ? lockSlice.sliceIdx : Math.max(0, Math.min(layer.dims[sliceAxis] - 1, Math.round(vox[sliceAxis])));
    const nu = layer.dims[uAxis];
    const nv = layer.dims[vAxis];
    const inPlane = vox[uAxis] >= -0.75 && vox[uAxis] <= nu - 0.25 && vox[vAxis] >= -0.75 && vox[vAxis] <= nv - 0.25;
    if (!inPlane) return null;

    const clampedVox = [
      Math.max(0, Math.min(layer.dims[0] - 1, Math.round(vox[0]))),
      Math.max(0, Math.min(layer.dims[1] - 1, Math.round(vox[1]))),
      Math.max(0, Math.min(layer.dims[2] - 1, Math.round(vox[2]))),
    ];
    clampedVox[sliceAxis] = sIdx;
    return { cssX, cssY, vox: clampedVox, sliceAxis, sliceIdx: sIdx, uAxis, vAxis };
  };

  nvCanvas.addEventListener(
    "mousedown",
    (e) => {
      if (!state.drawing.paintEnabled) return;
      // Respect Phase 1 2D Pan gestures & Alt+Click crosshair jump
      if (e.button === 1 || e.shiftKey || state.zoom2d.panModeActive) return;
      if (e.altKey && e.button === 0) return;
      if (e.button !== 0 && e.button !== 2) return;

      const tileInfo = get2DSliceTileAtClientPos(e.clientX, e.clientY);
      if (!tileInfo) return; // Allow normal 3D camera rotation on 3D render tile

      const layer = getActiveDrawingLayer();
      if (!layer) return;

      e.preventDefault();
      e.stopImmediatePropagation();

      const hit = clientPosToLayerVoxel(e.clientX, e.clientY, tileInfo, layer, null);
      if (!hit) return;

      state.drawing.lastSliceAxis = hit.sliceAxis;
      state.drawing.lastAxCorSag = tileInfo.axCorSag;

      const effectiveTool = e.button === 2 ? "eraser" : state.drawing.tool;
      const penVal = effectiveTool === "eraser" ? 0 : getActivePenValueForLayer(layer);
      const colorHex =
        layer.mode === "multilabel"
          ? layer.labelPalette?.[Math.max(0, (penVal || 1) - 1)]?.hex || layer.colorHex
          : layer.colorHex;

      if (effectiveTool === "flood_fill") {
        const diffMap = floodFill2DOnSlice(
          layer.voxels,
          layer.dims,
          hit.vox,
          hit.sliceAxis,
          hit.uAxis,
          hit.vAxis,
          penVal,
          state.drawing.overwriteLabels
        );
        const entry = applySparseDiffToVoxels(layer.voxels, diffMap, penVal, state.drawing.overwriteLabels);
        if (entry) {
          layer.undoStack.push(entry);
          if (layer.undoStack.length > 30) layer.undoStack.shift();
          layer.redoStack = [];
          commitDrawingLayerToGL(layer);
          showToast(`Flood filled ${entry.indices.length} pixel(s).`, 1800);
        } else {
          showToast("Flood fill made no change at this seed.", 1600);
        }
        return;
      }

      if (effectiveTool === "region_grow") {
        runBackendRegionGrow(layer, hit.vox, hit.sliceAxis);
        return;
      }

      const pxRadius = computeBrushRingCssRadius(tileInfo, layer);
      state.drawing.isStroking = true;
      const tooltip = document.getElementById("dktHoverTooltip");
      if (tooltip) tooltip.hidden = true;

      activeStroke = {
        layer,
        tileInfo,
        effectiveTool,
        penVal,
        colorHex,
        pxRadius,
        lockSlice: {
          sliceAxis: hit.sliceAxis,
          sliceIdx: hit.sliceIdx,
          uAxis: hit.uAxis,
          vAxis: hit.vAxis,
        },
        screenPts: [{ x: hit.cssX, y: hit.cssY }],
        voxPts: [hit.vox],
      };

      renderStrokeOverlay(
        {
          cssX: hit.cssX,
          cssY: hit.cssY,
          pxRadius,
          colorHex,
          isEraser: effectiveTool === "eraser",
        },
        activeStroke
      );
    },
    { capture: true }
  );

  window.addEventListener("mousemove", (e) => {
    if (activeStroke) {
      const hit = clientPosToLayerVoxel(
        e.clientX,
        e.clientY,
        activeStroke.tileInfo,
        activeStroke.layer,
        activeStroke.lockSlice
      );
      if (hit) {
        const lastScr = activeStroke.screenPts[activeStroke.screenPts.length - 1];
        if (!lastScr || Math.hypot(hit.cssX - lastScr.x, hit.cssY - lastScr.y) >= 1.5) {
          activeStroke.screenPts.push({ x: hit.cssX, y: hit.cssY });
          activeStroke.voxPts.push(hit.vox);
        }
        renderStrokeOverlay(
          {
            cssX: hit.cssX,
            cssY: hit.cssY,
            pxRadius: activeStroke.pxRadius,
            colorHex: activeStroke.colorHex,
            isEraser: activeStroke.effectiveTool === "eraser",
          },
          activeStroke
        );
      }
      return;
    }

    if (!state.drawing.paintEnabled || state.zoom2d.isDraggingPan) {
      nvCanvas.classList.remove("draw-brush-hover");
      clearStrokeOverlayCanvas();
      return;
    }

    const tileInfo = get2DSliceTileAtClientPos(e.clientX, e.clientY);
    const layer = getActiveDrawingLayer();
    if (!tileInfo || !layer || e.shiftKey || state.zoom2d.panModeActive) {
      nvCanvas.classList.remove("draw-brush-hover");
      clearStrokeOverlayCanvas();
      return;
    }

    const axInfo = resolveSliceVoxelAxes(layer.affine, tileInfo.axCorSag);
    state.drawing.lastSliceAxis = axInfo.sliceAxis;
    state.drawing.lastAxCorSag = tileInfo.axCorSag;

    const pxRadius = computeBrushRingCssRadius(tileInfo, layer);
    const penVal = getActivePenValueForLayer(layer);
    const colorHex =
      layer.mode === "multilabel"
        ? layer.labelPalette?.[Math.max(0, (penVal || 1) - 1)]?.hex || layer.colorHex
        : layer.colorHex;
    const isPointTool = state.drawing.tool === "flood_fill" || state.drawing.tool === "region_grow";
    nvCanvas.classList.toggle("draw-brush-hover", !isPointTool);
    if (isPointTool) {
      clearStrokeOverlayCanvas();
      return;
    }

    renderStrokeOverlay(
      {
        cssX: tileInfo.pos.cssX,
        cssY: tileInfo.pos.cssY,
        pxRadius,
        colorHex,
        isEraser: state.drawing.tool === "eraser",
      },
      null
    );
  });

  nvCanvas.addEventListener("mouseleave", () => {
    if (!activeStroke) {
      nvCanvas.classList.remove("draw-brush-hover");
      clearStrokeOverlayCanvas();
    }
  });

  const finishStroke = () => {
    if (!activeStroke) return;
    const finished = activeStroke;
    activeStroke = null;
    state.drawing.isStroking = false;
    clearStrokeOverlayCanvas();

    const { layer, effectiveTool, penVal, lockSlice, voxPts } = finished;
    if (!layer || !voxPts.length) return;

    const diffMap = new Map();
    if (effectiveTool === "filled_pen") {
      fillClosedPolygon2DOnSlice(
        layer.voxels,
        layer.dims,
        voxPts,
        lockSlice.sliceAxis,
        lockSlice.sliceIdx,
        lockSlice.uAxis,
        lockSlice.vAxis,
        penVal,
        state.drawing.overwriteLabels,
        diffMap
      );
    } else {
      const brushSz = state.drawing.brushRadius || 1;
      const is3D = Boolean(state.drawing.brush3D);
      const segCount = Math.max(1, voxPts.length - 1);
      for (let i = 0; i < segCount; i++) {
        const pA = voxPts[i];
        const pB = i + 1 < voxPts.length ? voxPts[i + 1] : pA;
        stampBrushSegmentVoxels(
          layer.voxels,
          layer.dims,
          pA,
          pB,
          penVal,
          brushSz,
          is3D,
          lockSlice.sliceAxis,
          state.drawing.overwriteLabels,
          diffMap
        );
      }
    }

    const entry = applySparseDiffToVoxels(layer.voxels, diffMap, penVal, state.drawing.overwriteLabels);
    if (entry) {
      layer.undoStack.push(entry);
      if (layer.undoStack.length > 30) layer.undoStack.shift();
      layer.redoStack = [];
      commitDrawingLayerToGL(layer);
    }
  };

  window.addEventListener("mouseup", finishStroke, { capture: true });
  window.addEventListener("blur", finishStroke);
}

// ============================================================================
// Wire Up All UI Event Listeners
// ============================================================================
function setupEventListeners() {
  // 1. Toggle Left Layers Panel (L) & Right Coregistration Panel (C) + Pin HUD
  document.getElementById("btnToggleLayersPanel")?.addEventListener("click", () => {
    toggleLayersPanel();
  });
  document.getElementById("btnCloseLayersPanel")?.addEventListener("click", () => {
    toggleLayersPanel(false);
  });
  document.getElementById("btnToggleDrawStudio")?.addEventListener("click", () => {
    toggleDrawingStudioPopout();
  });
  document.getElementById("btnToggleAtlasStudio")?.addEventListener("click", () => {
    toggleAtlasStudioPopout();
  });
  document.getElementById("btnOpenAtlasStudioSidebar")?.addEventListener("click", () => {
    openAtlasStudioPopout();
  });
  document.getElementById("dktOverlayStyleGroup")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".atlas-style-btn[data-atlas-style]");
    if (!btn) return;
    setAtlasOverlayStyle(btn.getAttribute("data-atlas-style"));
  });
  document.getElementById("btnPausePaintFromBanner")?.addEventListener("click", () => {
    setDrawingPaintEnabled(false);
  });
  document.getElementById("voiImportFileInput")?.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    for (const f of files) {
      try {
        await importFileAsDrawingLayer(f, { openStudio: true });
      } catch (err) {
        console.error("Failed to import VOI:", err);
        showToast(`Import failed for ${f.name}: ${err.message}`, 3600);
      }
    }
  });
  document.getElementById("btnToggleCoregPanel")?.addEventListener("click", () => {
    toggleCoregPanel();
  });
  document.getElementById("btnCloseCoregPanel")?.addEventListener("click", () => {
    toggleCoregPanel(false);
  });
  document.getElementById("btnPinHud")?.addEventListener("click", (e) => {
    state.hudPinned = !state.hudPinned;
    e.currentTarget.classList.toggle("active", state.hudPinned);
    showToast(
      state.hudPinned
        ? "Pinned Verification & Metrics HUD (stays visible when Coregistration Panel closes)."
        : "Unpinned Verification & Metrics HUD."
    );
  });

  // 1b. FSL BET Skull Stripping & FastSurfer DKT Atlas Controls (Separately Minimizable Modules)
  document.getElementById("betModuleHeader")?.addEventListener("click", () => {
    toggleNeuroModuleSection("betModuleSection", "betModuleHeader", "betModuleBody", "btnToggleBetModule");
  });
  document.getElementById("betModuleHeader")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleNeuroModuleSection("betModuleSection", "betModuleHeader", "betModuleBody", "btnToggleBetModule");
    }
  });
  document.getElementById("btnToggleBetModule")?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleNeuroModuleSection("betModuleSection", "betModuleHeader", "betModuleBody", "btnToggleBetModule");
  });

  document.getElementById("dktModuleHeader")?.addEventListener("click", (e) => {
    if (e.target.closest("#btnCheckFastSurfer")) return;
    toggleNeuroModuleSection("dktModuleSection", "dktModuleHeader", "dktModuleBody", "btnToggleDktModule");
  });
  document.getElementById("dktModuleHeader")?.addEventListener("keydown", (e) => {
    if (e.target.closest("#btnCheckFastSurfer")) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleNeuroModuleSection("dktModuleSection", "dktModuleHeader", "dktModuleBody", "btnToggleDktModule");
    }
  });
  document.getElementById("btnToggleDktModule")?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleNeuroModuleSection("dktModuleSection", "dktModuleHeader", "dktModuleBody", "btnToggleDktModule");
  });

  document.getElementById("inspectorModuleHeader")?.addEventListener("click", () => {
    toggleNeuroModuleSection("inspectorModuleSection", "inspectorModuleHeader", "inspectorModuleBody", "btnToggleInspectorModule");
  });
  document.getElementById("inspectorModuleHeader")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleNeuroModuleSection("inspectorModuleSection", "inspectorModuleHeader", "inspectorModuleBody", "btnToggleInspectorModule");
    }
  });
  document.getElementById("btnToggleInspectorModule")?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleNeuroModuleSection("inspectorModuleSection", "inspectorModuleHeader", "inspectorModuleBody", "btnToggleInspectorModule");
  });

  document.getElementById("btnCheckFastSurfer")?.addEventListener("click", (e) => {
    e.stopPropagation();
    checkNeuroToolsStatus(true);
  });
  document.getElementById("selectAtlasType")?.addEventListener("change", () => {
    syncAtlasTypeSelectionUI(true, true);
  });
  document.getElementById("selectDktMode")?.addEventListener("change", (e) => {
    const aid = String(document.getElementById("selectAtlasType")?.value || state.selectedAtlasType || "dkt").toLowerCase();
    const val = e.target?.value || document.getElementById("selectDktMode")?.value;
    if (val) {
      if (!state.atlasPreferredModeById) state.atlasPreferredModeById = new Map();
      state.atlasPreferredModeById.set(aid, val);
    }
    syncAtlasTypeSelectionUI(true, false);
    syncAtlasStudioUI();
  });
  document.getElementById("selectDktT1Vol")?.addEventListener("change", () => {
    syncHipsThomasOptionsUI();
  });
  document.getElementById("selectHipsContrast")?.addEventListener("change", (e) => {
    state.atlasStudioState.hipsContrast = e.target?.value === "wmn" ? "wmn" : "t1";
    syncAtlasStudioUI();
  });
  document.getElementById("chkHipsForceRecompute")?.addEventListener("change", (e) => {
    state.atlasStudioState.hipsForceRecompute = Boolean(e.target?.checked);
    syncAtlasStudioUI();
  });
  document.getElementById("btnCancelHipsJob")?.addEventListener("click", () => {
    cancelActiveAtlasJob();
  });
  document.getElementById("btnRunBet")?.addEventListener("click", () => {
    runFslBetSkullStrip();
  });
  document.getElementById("btnRunDktAtlas")?.addEventListener("click", () => {
    runDktSegmentation();
  });
  document.getElementById("dktFileInput")?.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      loadDktAtlasFile(e.target.files[0]);
      e.target.value = "";
    }
  });
  document.getElementById("fsLicenseFileInput")?.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      uploadAndSetFsLicenseFile(e.target.files[0]);
      e.target.value = "";
    }
  });
  document.getElementById("chkDktColorOverlay")?.addEventListener("change", (e) => {
    const dktVol = getDktVolume();
    if (!dktVol) return;
    const isOn = Boolean(e.target.checked);
    const atlasTitle = getAtlasShortTitle(dktVol);
    toggleVolumeVisibility(dktVol.id, isOn);
    showToast(
      isOn
        ? `${atlasTitle} segmentation color overlay enabled.`
        : `${atlasTitle} segmentation color overlay hidden — interactive cursor hover region lookup remains active!`
    );
  });
  document.getElementById("sliderDktOverlayOpacity")?.addEventListener("input", async (e) => {
    const dktVol = getDktVolume();
    if (!dktVol) return;
    const val = Math.max(0, Math.min(1, parseFloat(e.target.value) / 100));
    const meta = state.layerMeta.get(dktVol.id);
    if (meta) {
      meta.visible = val > 0;
      meta.userAtlasOpacity = val;
      if (val > 0) meta.savedOpacity = val;
    }
    await applyAtlasColormapToVolume(dktVol);
    state.nv.drawScene();
    renderLayerList();
    syncDktOverlayControlsUI();
    syncAtlasStudioUI();
  });

  const handleDktWindowInput = async () => {
    const dktVol = getDktVolume();
    if (!dktVol) return;
    const dm = state.layerMeta.get(dktVol.id);
    if (dm) dm.userCustomWindow = true;
    const minEl = document.getElementById("inputDktCalMin");
    const maxEl = document.getElementById("inputDktCalMax");
    const parsedMin = parseFloat(minEl?.value);
    const parsedMax = parseFloat(maxEl?.value);
    if (Number.isFinite(parsedMin)) {
      dktVol.cal_min = Math.max(0.5, parsedMin);
    }
    if (Number.isFinite(parsedMax)) {
      dktVol.cal_max = Math.max(dktVol.cal_min || 0.5, parsedMax);
    }
    if (dktVol.colormap === "FreeSurferColorLUT" || state.layerMeta.get(dktVol.id)?.isDktAtlas) {
      await applyFreeSurferColorLutToVolume(dktVol);
    }
    state.nv.updateGLVolume();
    state.nv.drawScene();
    renderLayerList();
  };

  document.getElementById("inputDktCalMin")?.addEventListener("change", handleDktWindowInput);
  document.getElementById("inputDktCalMin")?.addEventListener("input", handleDktWindowInput);
  document.getElementById("inputDktCalMax")?.addEventListener("change", handleDktWindowInput);
  document.getElementById("inputDktCalMax")?.addEventListener("input", handleDktWindowInput);

  document.getElementById("btnDktAutoWindow")?.addEventListener("click", async () => {
    const dktVol = getDktVolume();
    if (!dktVol) return;
    const defaultMax = getAtlasDefaultMaxLabel(dktVol);
    const atlasTitle = getAtlasShortTitle(dktVol);
    const dm = state.layerMeta.get(dktVol.id);
    if (dm) dm.userCustomWindow = false;
    dktVol.cal_min = 0.5;
    dktVol.cal_max = defaultMax;
    await applyFreeSurferColorLutToVolume(dktVol);
    state.nv.updateGLVolume();
    state.nv.drawScene();
    renderLayerList();
    showToast(`Reset ${atlasTitle} window to full brain label range (0.5 – ${defaultMax}).`);
  });

  // 1c. Responsive Browser Window Scaling (ResizeObserver + Window Resize) & Interactive DKT Cursor Hover
  const canvasWrapper = document.getElementById("canvasWrapper");
  if (canvasWrapper && typeof ResizeObserver === "function") {
    const ro = new ResizeObserver(() => {
      resizeWorkspaceCanvas();
    });
    ro.observe(canvasWrapper);
  }
  window.addEventListener("resize", () => {
    resizeWorkspaceCanvas();
    clampAllPopoutsToViewport();
  });

  wire2DZoomAndPanListeners();
  syncZoom2dUI();
  wireDrawingCanvasListeners();
  syncDrawHeaderButton();
  syncStrokeOverlayCanvasSize();

  const nvCanvas = document.getElementById("niivueCanvas");
  nvCanvas?.addEventListener("mousemove", (e) => {
    handleCanvasMouseHover(e);
  });
  nvCanvas?.addEventListener("mouseleave", () => {
    const tooltip = document.getElementById("dktHoverTooltip");
    if (tooltip) tooltip.hidden = true;
  });

  // 2. Patient Workspace (New Patient, Save Project, Open Project) & Data Loading
  document.getElementById("btnNewPatient")?.addEventListener("click", () => {
    createNewPatientWorkspace(true);
  });

  document.getElementById("btnSaveProject")?.addEventListener("click", () => {
    saveProjectToUserFolder();
  });

  document.getElementById("btnOpenProject")?.addEventListener("click", () => {
    openSavedProject();
  });

  // Close Application Button & 3-Choice Modal (Save & Close, Close Without Saving, Cancel)
  document.getElementById("btnCloseApp")?.addEventListener("click", () => {
    openCloseAppModal();
  });

  document.getElementById("btnModalCancelClose")?.addEventListener("click", () => {
    closeCloseAppModal();
  });

  document.getElementById("btnModalCloseWithoutSaving")?.addEventListener("click", async () => {
    closeCloseAppModal();
    await shutdownServerAndCloseTab();
  });

  document.getElementById("btnModalSaveAndClose")?.addEventListener("click", async () => {
    closeCloseAppModal();
    const saved = await saveProjectToUserFolder();
    if (saved) {
      await shutdownServerAndCloseTab();
    }
  });

  const closeAppModal = document.getElementById("closeAppModal");
  closeAppModal?.addEventListener("click", (e) => {
    if (e.target === closeAppModal) {
      closeCloseAppModal();
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (closeAppModal && !closeAppModal.hidden && !closeAppModal.classList.contains("hidden")) {
      closeCloseAppModal();
      return;
    }
    const topPopId = state.focusedPopoutId || getTopVisiblePopoutId();
    if (topPopId) {
      const pop = state.popouts.get(topPopId);
      if (pop && pop.isOpen()) {
        pop.close();
        return;
      }
    }
    if (state.zoom2d.panModeActive) {
      togglePan2dMode(false);
    }
  });

  document.getElementById("openProjectFolderInput")?.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      loadProjectFromFileList(Array.from(e.target.files));
      e.target.value = "";
    }
  });

  document.getElementById("btnLoadSampleSubject")?.addEventListener("click", () => {
    loadSampleMultimodalSubject();
  });

  document.getElementById("niftiFileInput")?.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      loadLocalNiftiFiles(e.target.files);
      e.target.value = "";
    }
  });

  document.getElementById("dicomFolderInput")?.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      loadDicomFolderFiles(e.target.files, false);
      e.target.value = "";
    }
  });

  document.getElementById("btnLoadSampleDicom")?.addEventListener("click", () => {
    loadDicomFolderFiles(null, true);
  });

  // 3. Drag & Drop Files or Folders onto Viewport
  const dropZone = document.getElementById("viewportDropZone");
  window.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone?.classList.add("drag-over");
  });
  window.addEventListener("dragleave", (e) => {
    if (e.target === document.body || e.target === dropZone) {
      dropZone?.classList.remove("drag-over");
    }
  });
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropZone?.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) {
      await loadLocalNiftiFiles(e.dataTransfer.files);
    }
  });

  // 4. View Mode Switcher (MPR, Axial, Coronal, Sagittal, 3D Render)
  document.querySelectorAll(".view-btn[data-slicetype]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-btn[data-slicetype]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.getAttribute("data-slicetype");
      const map = {
        axial: SLICE_TYPE.AXIAL,
        coronal: SLICE_TYPE.CORONAL,
        sagittal: SLICE_TYPE.SAGITTAL,
        multiplanar: SLICE_TYPE.MULTIPLANAR,
        render: SLICE_TYPE.RENDER,
      };
      state.nv.setSliceType(map[mode] ?? SLICE_TYPE.MULTIPLANAR);
      syncZoom2dUI();
    });
  });

  document.getElementById("btnToggleCrosshair")?.addEventListener("click", (e) => {
    const btn = e.currentTarget;
    const isOn = !btn.classList.contains("active");
    btn.classList.toggle("active", isOn);
    state.nv.opts.crosshairWidth = isOn ? 1 : 0;
    state.nv.drawScene();
  });

  document.getElementById("btnToggleInterp")?.addEventListener("click", (e) => {
    const btn = e.currentTarget;
    const isSmooth = !btn.classList.contains("active");
    btn.classList.toggle("active", isSmooth);
    state.nv.setInterpolation(!isSmooth);
  });

  // 5. Layer Stack Event Delegation (Toggle On/Off, REF/MOV/CHILD roles, Solo, Quick Export, Opacity, Colormap, Windowing)
  const layerListEl = document.getElementById("layerList");
  layerListEl?.addEventListener("change", (e) => {
    const target = e.target;
    const action = target.getAttribute("data-action");
    const volId = target.getAttribute("data-id");
    if (!action || !volId) return;
    const idx = getVolumeIndexById(volId);
    if (idx < 0) return;
    const vol = state.nv.volumes[idx];

    if (action === "toggle-vis") {
      toggleVolumeVisibility(volId, target.checked);
    } else if (action === "colormap") {
      if (target.value === "FreeSurferColorLUT") {
        applyFreeSurferColorLutToVolume(vol).then(() => {
          state.nv.updateGLVolume();
          state.nv.drawScene();
          renderLayerList();
        });
      } else {
        const meta = state.layerMeta.get(vol.id);
        if (meta && meta.unmaskedVolumeData && !meta.isDktAtlas && vol.img && vol.img.length === meta.unmaskedVolumeData.length) {
          vol.img.set(meta.unmaskedVolumeData);
          if (meta.savedCalMinBeforeFsLut !== undefined) {
            vol.cal_min = meta.savedCalMinBeforeFsLut;
          }
        }
        vol.colormapLabel = null;
        vol.colormap = target.value;
        state.nv.updateGLVolume();
        state.nv.drawScene();
        renderLayerList();
      }
    } else if (action === "cal-min") {
      const parsedMin = parseFloat(target.value);
      const meta = state.layerMeta.get(vol.id);
      if (meta?.isDktAtlas || vol.colormap === "FreeSurferColorLUT") {
        vol.cal_min = Number.isFinite(parsedMin) ? Math.max(0.5, parsedMin) : 0.5;
        applyFreeSurferColorLutToVolume(vol).then(() => {
          state.nv.updateGLVolume();
          state.nv.drawScene();
          syncDktOverlayControlsUI();
        });
      } else {
        vol.cal_min = Number.isFinite(parsedMin) ? parsedMin : 0;
        state.nv.updateGLVolume();
        state.nv.drawScene();
      }
    } else if (action === "cal-max") {
      const parsedMax = parseFloat(target.value);
      const meta = state.layerMeta.get(vol.id);
      if (meta?.isDktAtlas || vol.colormap === "FreeSurferColorLUT") {
        const defMax = getAtlasDefaultMaxLabel(vol);
        vol.cal_max = Number.isFinite(parsedMax) ? Math.max(vol.cal_min || 0.5, parsedMax) : defMax;
        applyFreeSurferColorLutToVolume(vol).then(() => {
          state.nv.updateGLVolume();
          state.nv.drawScene();
          syncDktOverlayControlsUI();
        });
      } else {
        vol.cal_max = Number.isFinite(parsedMax) ? parsedMax : 100;
        state.nv.updateGLVolume();
        state.nv.drawScene();
      }
    }
  });

  layerListEl?.addEventListener("input", (e) => {
    const target = e.target;
    if (target.getAttribute("data-action") === "opacity") {
      const volId = target.getAttribute("data-id");
      const idx = getVolumeIndexById(volId);
      if (idx < 0) return;
      const meta = state.layerMeta.get(volId);
      const val = parseFloat(target.value) / 100;
      if (meta) {
        meta.visible = val > 0;
        if (val > 0) meta.savedOpacity = val;
      }
      state.nv.setOpacity(idx, val);
      state.nv.drawScene();
      const card = target.closest(".layer-card");
      if (card) {
        card.classList.toggle("is-hidden", val === 0);
        const sw = card.querySelector(".vis-switch");
        if (sw) sw.checked = val > 0;
        const pct = card.querySelector(".opacity-pct");
        if (pct) pct.textContent = `${Math.round(val * 100)}%`;
      }
      if (meta?.isDktAtlas) {
        syncDktOverlayControlsUI();
      }
      if (meta?.isDrawingLayer && meta.drawingLayerId) {
        const dLayer = getDrawingLayerById(meta.drawingLayerId);
        if (dLayer) {
          dLayer.opacity = val;
          dLayer.visible = val > 0;
          if (state.drawing.activeLayerId === dLayer.id && state.drawing.popout?.bodyEl) {
            const pctVal = Math.round(val * 100);
            const sEl = state.drawing.popout.bodyEl.querySelector("#sliderDrawLayerOpacity");
            const vEl = state.drawing.popout.bodyEl.querySelector("#valDrawLayerOpacity");
            if (sEl) sEl.value = String(pctVal);
            if (vEl) vEl.textContent = `${pctVal}%`;
          }
        }
      }
    }
  });

  layerListEl?.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.getAttribute("data-action");
    const volId = btn.getAttribute("data-id");
    const idx = getVolumeIndexById(volId);
    if (idx < 0) return;
    const vol = state.nv.volumes[idx];
    const meta = state.layerMeta.get(volId);

    if (action === "set-ref") {
      state.refVolId = volId;
      if (meta) meta.isChildOfMov = false;
      if (state.movVolId === volId) {
        const alt = state.nv.volumes.find((v) => v.id !== volId);
        if (alt) state.movVolId = alt.id;
      }
      renderLayerList();
      computeAndDisplayAlignmentMetrics(true);
    } else if (action === "set-mov") {
      state.movVolId = volId;
      if (meta) meta.isChildOfMov = false;
      if (state.refVolId === volId) {
        const alt = state.nv.volumes.find((v) => v.id !== volId);
        if (alt) state.refVolId = alt.id;
      }
      renderLayerList();
      computeAndDisplayAlignmentMetrics(true);
    } else if (action === "toggle-child") {
      if (volId === state.refVolId || volId === state.movVolId) {
        showToast("A Child volume must be distinct from the Reference and Moving parent volumes.");
        return;
      }
      if (meta) {
        meta.isChildOfMov = !meta.isChildOfMov;
        meta.isCheckedForChildApply = meta.isChildOfMov;
        if (meta.isChildOfMov) {
          // Immediately sync this child volume with the Moving volume's current W_ras transform
          const W_ras = getCurrentMovingWorldMatrix();
          setVolumeAffineMatrix(idx, multiplyMat4(W_ras, meta.origAffine));
          showToast(`Linked "${meta.label}" as a live Child of Moving volume.`);
        } else {
          showToast(`Unlinked "${meta.label}" from live Moving parent updates.`);
        }
        renderLayerList();
      }
    } else if (action === "edit-drawing") {
      if (meta?.drawingLayerId) {
        selectActiveDrawingLayer(meta.drawingLayerId);
        openDrawingStudioPopout();
      }
    } else if (action === "export-drawing") {
      if (meta?.drawingLayerId) {
        exportDrawingLayer(meta.drawingLayerId, "save_as");
      }
    } else if (action === "convert-to-drawing") {
      convertVolumeToDrawingLayer(volId);
    } else if (action === "quick-export") {
      exportSingleVolumeAsNifti(volId).then((fname) => {
        if (fname) showToast(`Exported ${fname}`);
      });
    } else if (action === "solo") {
      if (state.soloVolId === volId) {
        state.soloVolId = null;
        state.nv.volumes.forEach((v, i) => {
          const m = state.layerMeta.get(v.id);
          if (m) {
            m.visible = true;
            state.nv.setOpacity(i, m.savedOpacity || 0.75);
          }
        });
      } else {
        state.soloVolId = volId;
        state.nv.volumes.forEach((v, i) => {
          const m = state.layerMeta.get(v.id);
          if (m) {
            const isTarget = v.id === volId;
            m.visible = isTarget;
            state.nv.setOpacity(i, isTarget ? 1.0 : 0.0);
          }
        });
      }
      state.nv.drawScene();
      renderLayerList();
    } else if (action === "auto-window") {
      if (meta?.isDktAtlas) {
        vol.cal_min = 0.5;
        vol.cal_max = getAtlasDefaultMaxLabel(vol);
        applyFreeSurferColorLutToVolume(vol).then(() => {
          state.nv.updateGLVolume();
          state.nv.drawScene();
          renderLayerList();
        });
      } else {
        if (typeof vol.calMinMax === "function") vol.calMinMax();
        state.nv.updateGLVolume();
        state.nv.drawScene();
        renderLayerList();
      }
    } else if (action === "remove") {
      if (meta?.isDrawingLayer && meta.drawingLayerId) {
        deleteDrawingLayer(meta.drawingLayerId);
        return;
      }
      state.nv.removeVolumeByIndex(idx);
      state.layerMeta.delete(volId);
      if (state.refVolId === volId) state.refVolId = null;
      if (state.movVolId === volId) state.movVolId = null;
      if (state.activeDktVolId === volId || state.activeAtlasVolId === volId) {
        state.activeDktVolId = null;
        state.activeAtlasVolId = null;
        const remaining = getLoadedAtlasVolumes()[0] || null;
        if (remaining) {
          activateAtlasOverlay(remaining.id, true);
          return;
        }
        resetDktInspectorUI();
      }
      renderLayerList();
      syncAtlasStudioUI();
    }
  });

  // Bulk layer visibility buttons
  document.getElementById("btnShowAllLayers")?.addEventListener("click", () => {
    state.soloVolId = null;
    state.nv.volumes.forEach((v, i) => {
      const m = state.layerMeta.get(v.id);
      if (m) {
        m.visible = true;
        state.nv.setOpacity(i, m.savedOpacity || 0.75);
      }
    });
    state.nv.drawScene();
    renderLayerList();
  });

  document.getElementById("btnHideOverlays")?.addEventListener("click", () => {
    state.nv.volumes.forEach((v, i) => {
      const m = state.layerMeta.get(v.id);
      if (m) {
        const isRef = v.id === state.refVolId;
        m.visible = isRef;
        state.nv.setOpacity(i, isRef ? 1.0 : 0.0);
      }
    });
    state.nv.drawScene();
    renderLayerList();
  });

  // 6. Verification Bar & Keyboard Shortcuts
  document.getElementById("btnBlinkToggle")?.addEventListener("click", toggleBlinkMode);
  window.addEventListener("keydown", (e) => {
    const activeEl = document.activeElement;
    const tag = activeEl?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || activeEl?.isContentEditable) return;

    const lowerKey = e.key.toLowerCase();
    const drawStudioOpen = Boolean(state.drawing.popout && state.drawing.popout.isOpen() && !state.drawing.popout.isMinimized());
    const drawCtxActive = state.drawing.paintEnabled || drawStudioOpen;

    if ((e.metaKey || e.ctrlKey) && lowerKey === "z" && !e.altKey) {
      if (drawCtxActive || state.drawing.layers.length > 0) {
        e.preventDefault();
        if (e.shiftKey) redoDrawingLayer();
        else undoDrawingLayer();
      }
      return;
    }
    if ((e.metaKey || e.ctrlKey) && lowerKey === "y" && !e.altKey) {
      if (drawCtxActive || state.drawing.layers.length > 0) {
        e.preventDefault();
        redoDrawingLayer();
      }
      return;
    }

    if (e.code === "Space") {
      e.preventDefault();
      stepBlinkComparator();
    } else if (lowerKey === "c" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      toggleCoregPanel();
    } else if (lowerKey === "l" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      toggleLayersPanel();
    } else if (lowerKey === "d" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      toggleDrawingStudioPopout();
    } else if (lowerKey === "a" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      toggleAtlasStudioPopout();
    } else if (lowerKey === "e" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      if (!state.drawing.paintEnabled) {
        openDrawingStudioPopout();
      }
      setDrawingPaintEnabled();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && lowerKey === "b") {
      e.preventDefault();
      state.drawing.tool = "pen";
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && lowerKey === "f") {
      e.preventDefault();
      state.drawing.tool = "filled_pen";
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && lowerKey === "x") {
      e.preventDefault();
      state.drawing.tool = "eraser";
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && lowerKey === "g") {
      e.preventDefault();
      state.drawing.tool = "flood_fill";
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && lowerKey === "w") {
      e.preventDefault();
      state.drawing.tool = "region_grow";
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && e.key === "[") {
      e.preventDefault();
      state.drawing.brushRadius = Math.max(1, (state.drawing.brushRadius || 1) - 1);
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if (drawCtxActive && !e.metaKey && !e.ctrlKey && !e.altKey && e.key === "]") {
      e.preventDefault();
      state.drawing.brushRadius = Math.min(5, (state.drawing.brushRadius || 1) + 1);
      syncDrawActiveBanner();
      renderDrawingStudioUI();
    } else if ((e.key === "+" || e.key === "=") && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      step2DZoom(1.25);
    } else if ((e.key === "-" || e.key === "_") && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      step2DZoom(0.8);
    } else if (e.key === "0" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      reset2DZoom();
    } else if (lowerKey === "h" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      togglePan2dMode();
    }
  });

  document.getElementById("crossfadeSlider")?.addEventListener("input", (e) => {
    const pct = parseInt(e.target.value, 10);
    document.getElementById("crossfadeValue").textContent = `${pct}%`;
    const refIdx = getVolumeIndexById(state.refVolId);
    const movIdx = getVolumeIndexById(state.movVolId);
    if (refIdx >= 0 && movIdx >= 0) {
      const alpha = pct / 100;
      state.nv.setOpacity(refIdx, 1.0);
      state.nv.setOpacity(movIdx, alpha);
      const movMeta = state.layerMeta.get(state.movVolId);
      if (movMeta) {
        movMeta.visible = alpha > 0;
        movMeta.savedOpacity = alpha > 0 ? alpha : movMeta.savedOpacity;
      }
      state.nv.drawScene();
      renderLayerList();
    }
  });

  document.getElementById("btnEdgeOverlay")?.addEventListener("click", toggle3DEdgeContourOverlay);
  document.getElementById("btnCheckerboard")?.addEventListener("click", toggle3DCheckerboardOverlay);
  document.getElementById("btnAnaglyphMode")?.addEventListener("click", toggleAnaglyphFusion);

  // 7. Right Panel Coregistration Controls
  document.getElementById("selectRefVol")?.addEventListener("change", (e) => {
    state.refVolId = e.target.value;
    renderLayerList();
    computeAndDisplayAlignmentMetrics(true);
  });

  document.getElementById("selectMovVol")?.addEventListener("change", (e) => {
    state.movVolId = e.target.value;
    renderLayerList();
    computeAndDisplayAlignmentMetrics(true);
  });

  ["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz"].forEach((param) => {
    const slider = document.getElementById(`slider_${param}`);
    slider?.addEventListener("input", () => {
      const movMeta = state.layerMeta.get(state.movVolId);
      if (!movMeta) return;
      movMeta.params[param] = parseFloat(slider.value);
      applyMovingTransformFromParams(true);
    });
  });

  document.querySelectorAll(".nudge-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const param = btn.getAttribute("data-param");
      const step = parseFloat(btn.getAttribute("data-step") || "0");
      const movMeta = state.layerMeta.get(state.movVolId);
      if (!movMeta || !param) return;
      movMeta.params[param] = Number((movMeta.params[param] + step).toFixed(2));
      applyMovingTransformFromParams(true);
    });
  });

  document.getElementById("btnResetTransform")?.addEventListener("click", undoAndClearCoregTransform);
  document.getElementById("btnUndoClearTransform")?.addEventListener("click", undoAndClearCoregTransform);
  document.getElementById("btnUndoAutoCoreg")?.addEventListener("click", undoAndClearCoregTransform);

  document.getElementById("btnAlignCenters")?.addEventListener("click", alignCentersOfMass);

  document.getElementById("btnApplyKnownDemoMisalign")?.addEventListener("click", () => {
    const movMeta = state.layerMeta.get(state.movVolId);
    if (!movMeta) return;
    movMeta.params = { tx: 10.0, ty: -8.0, tz: 5.0, rx: -5.0, ry: 4.0, rz: 10.0, sx: 1, sy: 1, sz: 1 };
    applyMovingTransformFromParams(true);
    state.baselineMetrics = computeAndDisplayAlignmentMetrics(false);
    updateMetricsFooterUI(state.baselineMetrics);
    showToast("Applied synthetic misalignment to Moving + linked Child volumes.");
  });

  document.getElementById("btnRunAutoCoreg")?.addEventListener("click", async () => {
    const method = document.getElementById("selectAutoMethod").value;
    const dof = parseInt(document.getElementById("selectAutoDOF").value || "6", 10);
    if (method === "browser_nmi") {
      await runInBrowserAutoCoregistration(dof);
    } else {
      await runBackendCoregistration(method, dof);
    }
  });

  // 8. Child Images & Transform Propagation Section
  const childListEl = document.getElementById("childVolumesList");
  childListEl?.addEventListener("change", (e) => {
    const volId = e.target.getAttribute("data-child-check");
    if (volId) {
      const m = state.layerMeta.get(volId);
      if (m) m.isCheckedForChildApply = e.target.checked;
    }
  });

  childListEl?.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-child-link]");
    if (!btn) return;
    const volId = btn.getAttribute("data-child-link");
    const idx = getVolumeIndexById(volId);
    const m = state.layerMeta.get(volId);
    if (idx >= 0 && m) {
      m.isChildOfMov = !m.isChildOfMov;
      m.isCheckedForChildApply = m.isChildOfMov;
      if (m.isChildOfMov) {
        const W_ras = getCurrentMovingWorldMatrix();
        setVolumeAffineMatrix(idx, multiplyMat4(W_ras, m.origAffine));
      }
      renderLayerList();
    }
  });

  document.getElementById("btnApplyMovTransformToChildren")?.addEventListener("click", () => {
    const W_ras = getCurrentMovingWorldMatrix();
    let appliedCount = 0;
    state.nv.volumes.forEach((v, idx) => {
      if (v.id === state.refVolId || v.id === state.movVolId) return;
      const m = state.layerMeta.get(v.id);
      if (m && (m.isCheckedForChildApply || m.isChildOfMov)) {
        setVolumeAffineMatrix(idx, multiplyMat4(W_ras, m.origAffine), true);
        appliedCount++;
      }
    });
    state.nv.updateGLVolume();
    state.nv.drawScene();
    showToast(`Applied Moving volume's 4×4 World RAS transform to ${appliedCount} child volume(s)!`);
  });

  document.getElementById("btnResetCheckedChildren")?.addEventListener("click", () => {
    let resetCount = 0;
    state.nv.volumes.forEach((v, idx) => {
      if (v.id === state.refVolId || v.id === state.movVolId) return;
      const m = state.layerMeta.get(v.id);
      if (m && (m.isCheckedForChildApply || m.isChildOfMov)) {
        setVolumeAffineMatrix(idx, m.origAffine, true);
        resetCount++;
      }
    });
    state.nv.updateGLVolume();
    state.nv.drawScene();
    showToast(`Reset ${resetCount} child volume(s) to their original header affine.`);
  });

  document.getElementById("btnSaveTransformSnapshot")?.addEventListener("click", () => {
    saveTransformSnapshot();
    showToast("Saved current Moving volume transformation to session library.");
  });

  document.getElementById("btnApplySelectedSavedTransform")?.addEventListener("click", () => {
    const selVal = document.getElementById("selectSavedTransform")?.value;
    const targetMode = document.getElementById("selectApplyTarget")?.value || "mov_and_children";
    if (selVal === "__current_mov__") {
      const W_ras = getCurrentMovingWorldMatrix();
      applyWorldMatrixToTargets(W_ras, targetMode, null);
      showToast(`Applied current Moving transform to ${targetMode.replace(/_/g, " ")}.`);
      return;
    }
    const found = state.savedTransforms.find((t) => t.id === selVal);
    if (!found) return;
    applyWorldMatrixToTargets(found.worldMat, targetMode, found.params);
    showToast(`Applied "${found.name}" to ${targetMode.replace(/_/g, " ")}.`);
  });

  document.getElementById("matFileInput")?.addEventListener("change", (e) => {
    if (e.target.files?.length) {
      handleImportMatFile(e.target.files[0]);
      e.target.value = "";
    }
  });

  // 9. Export Coregistered NIfTI & Matrix Actions
  const exportListEl = document.getElementById("exportVolumesChecklist");
  exportListEl?.addEventListener("change", (e) => {
    const volId = e.target.getAttribute("data-export-check");
    if (volId) {
      const m = state.layerMeta.get(volId);
      if (m) m.isCheckedForExport = e.target.checked;
    }
  });

  exportListEl?.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-export-single]");
    if (!btn) return;
    const volId = btn.getAttribute("data-export-single");
    if (volId) {
      exportSingleVolumeAsNifti(volId).then((fname) => {
        if (fname) showToast(`Exported ${fname}`);
      });
    }
  });

  document.getElementById("btnExportSelectedNifti")?.addEventListener("click", exportSelectedVolumesAsNifti);

  document.getElementById("btnCopyMatrix")?.addEventListener("click", () => {
    const W_ras = getCurrentMovingWorldMatrix();
    const text = W_ras.map((row) => row.map((v) => Number(v).toFixed(6)).join("  ")).join("\n");
    navigator.clipboard?.writeText(text);
    showToast("Copied 4×4 World RAS transform matrix to clipboard.");
  });

  document.getElementById("btnDownloadWorldMat")?.addEventListener("click", () => {
    const W_ras = getCurrentMovingWorldMatrix();
    const text = W_ras.map((row) => row.map((v) => Number(v).toFixed(6)).join("  ")).join("\n") + "\n";
    downloadTextFile("coreg_world_ras.mat", text);
    showToast("Downloaded coreg_world_ras.mat (4×4 World RAS transform).");
  });

  document.getElementById("btnDownloadFlirtMat")?.addEventListener("click", () => {
    const refIdx = getVolumeIndexById(state.refVolId);
    const movIdx = getVolumeIndexById(state.movVolId);
    if (refIdx < 0 || movIdx < 0) return;
    const W_ras = getCurrentMovingWorldMatrix();
    const flirtMat = worldRASToFlirtMat(W_ras, state.nv.volumes[refIdx], state.nv.volumes[movIdx]);
    const text = flirtMat.map((row) => row.map((v) => Number(v).toFixed(6)).join("  ")).join("\n") + "\n";
    downloadTextFile("coreg_fsl_flirt.mat", text);
    showToast("Downloaded coreg_fsl_flirt.mat (FSL FLIRT scaled-voxel convention).");
  });

  document.getElementById("btnCaptureBaseline")?.addEventListener("click", () => {
    state.baselineMetrics = computeAndDisplayAlignmentMetrics(false);
    updateMetricsFooterUI(state.baselineMetrics);
    showToast("Captured current alignment metrics as Baseline.");
  });

  document.getElementById("btnRecomputeMetrics")?.addEventListener("click", () => {
    computeAndDisplayAlignmentMetrics(false);
    showToast("Recomputed 3D alignment similarity metrics.");
  });
}

// Initialize NiiVue Instance & Load Default Multimodal Subject
async function initApp() {
  const nv = new NiivueClass({
    dragAndDropEnabled: false,
    backColor: [0.03, 0.05, 0.08, 1],
    show3Dcrosshair: true,
    crosshairColor: [0.22, 0.74, 0.97, 0.85],
    isOrientCube: true,
    isRadiologicalConvention: false,
    isCornerOrientationText: true,
    multiplanarForceRender: true,
    yoke3Dto2DZoom: false,
  });

  state.nv = nv;
  await nv.attachTo("niivueCanvas");
  nv.setSliceType(SLICE_TYPE.MULTIPLANAR);

  nv.onLocationChange = (data) => {
    updateCrosshairReadout(data);
  };

  setupEventListeners();
  toggleCoregPanel(false);
  await checkNeuroToolsStatus(false);
  await loadSampleMultimodalSubject();
}

window.NeuroCoregPopouts = {
  createPopoutWindow,
  getPopoutWindow,
  closeAllPopouts,
};

window.NeuroCoregZoom = {
  set2DZoom,
  step2DZoom,
  reset2DZoom,
  togglePan2dMode,
  getCurrentPan2D,
};

window.NeuroCoregDrawing = {
  createDrawingLayer,
  openDrawingStudioPopout,
  toggleDrawingStudioPopout,
  setDrawingPaintEnabled,
  exportDrawingLayer,
  mergeAndExportVisibleDrawingLayers,
  importFileAsDrawingLayer,
  undoDrawingLayer,
  redoDrawingLayer,
};

window.NeuroCoregAtlases = {
  getLoadedAtlasVolumes,
  lookupAllAtlasesAtRAS,
  lookupAtlasRegionAtRAS,
  activateAtlasOverlay,
  setAtlasOverlayStyle,
  copySelectedAtlasRoisToDrawingLayer,
  openAtlasStudioPopout,
  toggleAtlasStudioPopout,
};

initApp().catch((err) => {
  console.error("Failed to initialize NeuroCoreg Studio:", err);
  showToast(`Initialization Error: ${err.message}`, 6000);
});
