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
    return { modality: "Destrieux Atlas (FreeSurferColorLUT)", colormap: "actc", opacity: 0.35, isMask: true, isDktAtlas: true, atlasType: "destrieux" };
  }
  if (lower.includes("dkt") || lower.includes("aparc") || lower.includes("aseg") || lower.includes("atlas")) {
    return { modality: "DKT Atlas (104 ROIs)", colormap: "actc", opacity: 0.38, isMask: true, isDktAtlas: true, atlasType: "dkt" };
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
  const isMask = overrides.isMask !== undefined ? Boolean(overrides.isMask) : inferred.isMask;
  const isDktAtlas = overrides.isDktAtlas !== undefined ? Boolean(overrides.isDktAtlas) : Boolean(inferred.isDktAtlas);
  const atlasType = overrides.atlasType || inferred.atlasType || (isDktAtlas ? resolveAtlasTypeForVolume(vol) : null);

  state.layerMeta.set(vol.id, {
    label: overrides.label || vol.name || "Volume",
    modality: overrides.modality || inferred.modality,
    visible,
    savedOpacity,
    origAffine,
    params: { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0, sx: 1, sy: 1, sz: 1 },
    isChildOfMov: isChild,
    isCheckedForChildApply: isChild || isMask,
    isCheckedForExport: isChild || overrides.role === "moving" || isDktAtlas,
    isMask,
    isDktAtlas,
    atlasType,
    rawBytes: overrides.rawBytes || null,
    isSyntheticVerification: Boolean(overrides.isSyntheticVerification),
    isEdgeOfMov: Boolean(overrides.isEdgeOfMov),
    isCheckerboard: Boolean(overrides.isCheckerboard),
  });

  if (isDktAtlas) {
    state.activeDktVolId = vol.id;
  }

  if (!state.refVolId && !overrides.isSyntheticVerification && !isDktAtlas) {
    state.refVolId = vol.id;
  } else if (!state.movVolId && vol.id !== state.refVolId && !isChild && !overrides.isSyntheticVerification && !isDktAtlas) {
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
  meta.visible = nextVis;

  const targetOpacity = nextVis ? (meta.savedOpacity > 0 ? meta.savedOpacity : 0.75) : 0.0;
  state.nv.setOpacity(idx, targetOpacity);

  // If the user explicitly turns ON a DKT or Destrieux atlas layer in the Image Stack, make it the active atlas
  if (meta.isDktAtlas && nextVis) {
    state.activeDktVolId = volId;
    const vol = state.nv.volumes[idx];
    const detectedType = resolveAtlasTypeForVolume(vol);
    const atlasSelect = document.getElementById("selectAtlasType");
    if (atlasSelect && atlasSelect.value !== detectedType) {
      atlasSelect.value = detectedType;
      syncAtlasTypeSelectionUI(false, false);
    }
    state.nv.volumes.forEach((otherVol, otherIdx) => {
      if (otherVol.id !== volId) {
        const otherMeta = state.layerMeta.get(otherVol.id);
        if (otherMeta?.isDktAtlas && otherMeta.visible) {
          otherMeta.visible = false;
          state.nv.setOpacity(otherIdx, 0.0);
        }
      }
    });
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

    const isRef = vol.id === state.refVolId;
    const isMov = vol.id === state.movVolId;
    const isChild = Boolean(meta.isChildOfMov) && !isRef && !isMov;
    const isDkt = Boolean(meta.isDktAtlas);
    const isBet = Boolean(meta.isBetBrain);
    const isSolo = vol.id === state.soloVolId;

    const card = document.createElement("div");
    card.className = `layer-card ${!meta.visible ? "is-hidden" : ""} ${isRef ? "role-ref" : ""} ${isMov ? "role-mov" : ""} ${isChild ? "role-child" : ""}`;
    card.setAttribute("role", "listitem");

    const currentOpacityPct = Math.round((meta.visible ? vol.opacity : 0) * 100);
    const calMin = vol.cal_min !== undefined ? Number(vol.cal_min).toFixed(1) : "0";
    const calMax = vol.cal_max !== undefined ? Number(vol.cal_max).toFixed(1) : "100";

    card.innerHTML = `
      <div class="layer-top-row">
        <label class="vis-switch-label" title="Toggle volume visibility ON / OFF">
          <input type="checkbox" role="switch" class="vis-switch" data-action="toggle-vis" data-id="${vol.id}" ${meta.visible ? "checked" : ""} />
          <span class="layer-title" title="${meta.label} (${vol.name})">${meta.label}</span>
        </label>
        <div class="layer-badges">
          ${isBet ? `<span class="role-badge bet" title="Skull-Stripped Brain Extracted Volume (FSL BET)">BET</span>` : ""}
          ${isDkt ? `<span class="role-badge dkt" title="Active DKT Anatomical Parcellation Atlas for Cursor Hover">DKT</span>` : ""}
          <button class="role-badge ${isRef ? "ref" : "inactive"}" data-action="set-ref" data-id="${vol.id}" title="Set as Fixed Reference Volume">REF</button>
          <button class="role-badge ${isMov ? "mov" : "inactive"}" data-action="set-mov" data-id="${vol.id}" title="Set as Moving Parent Volume">MOV</button>
          <button class="role-badge ${isChild ? "child" : "inactive"}" data-action="toggle-child" data-id="${vol.id}" title="Link/Unlink as Child of Moving Volume (inherits Moving transforms automatically)">CHILD</button>
          <button class="icon-btn ${isSolo ? "solo-active" : ""}" data-action="solo" data-id="${vol.id}" title="Solo this volume">S</button>
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
          ${(() => { const d = getVolDims(vol); return `${d[1]}×${d[2]}×${d[3]}`; })()}
        </span>
      </div>
    `;

    listEl.appendChild(card);
  });

  syncCoregVolumeSelectors();
  renderChildAndExportLists();
  syncDktOverlayControlsUI();
}

// Populate the Reference & Moving dropdowns in the right-hand Coregistration Studio
function syncCoregVolumeSelectors() {
  const refSelect = document.getElementById("selectRefVol");
  const movSelect = document.getElementById("selectMovVol");
  if (!refSelect || !movSelect || !state.nv) return;

  const realVolumes = state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isSyntheticVerification;
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
    return m && !m.isSyntheticVerification && !m.isDktAtlas && !m.isMask;
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
}

// Render Section 4 (Child Images list) and Section 5 (NIfTI Export Checklist)
function renderChildAndExportLists() {
  const childListEl = document.getElementById("childVolumesList");
  const exportListEl = document.getElementById("exportVolumesChecklist");
  if (!state.nv) return;

  const candidateChildren = state.nv.volumes.filter((v) => {
    const m = state.layerMeta.get(v.id);
    return m && !m.isSyntheticVerification && v.id !== state.refVolId && v.id !== state.movVolId;
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
      return m && !m.isSyntheticVerification && v.id !== state.refVolId;
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

  for (const file of files) {
    try {
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

      if (inferred.isDktAtlas) {
        await addDktAtlasVolumeToWorkspace(
          rawBytes,
          fileName,
          `DKT Atlas (${file.name.replace(/\.(nii\.gz|nii|mgz|mgh)$/i, "")})`
        );
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
  showToast(`Loaded ${files.length} volume(s).`);
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
  state.baselineMetrics = null;
  state.savedTransforms = [];

  state.nv.updateGLVolume();
  state.nv.drawScene();

  renderSavedTransformsDropdown();
  renderLayerList();
  resetDktInspectorUI();

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
    return m && !m.isSyntheticVerification;
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
        isDktAtlas: Boolean(m?.isDktAtlas),
        atlasType: m?.atlasType || null,
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

  const manifest = {
    schema: "neurocoreg-project-v1",
    savedAt: new Date().toISOString(),
    patientId,
    viewState: {
      coregPanelOpen: state.coregPanelOpen,
      hudPinned: state.hudPinned,
      crosshairWidth: state.nv?.opts?.crosshairWidth ?? 1,
    },
    baselineMetrics: state.baselineMetrics,
    currentMetrics,
    activeMovingWorldMatrix: W_ras,
    activeMovingFlirtMatrix: flirtMat,
    savedTransforms: state.savedTransforms,
    volumes: volumeEntries.map((ve) => ve.entry),
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

  return { patientId, safePatient, manifest, volumeEntries, transformFiles };
}

// Save Project to a Folder of the User's Choice (returns true if saved, false if cancelled/failed)
async function saveProjectToUserFolder() {
  if (!state.nv) return false;
  const { patientId, safePatient, manifest, volumeEntries, transformFiles } = buildProjectBundleData();
  if (volumeEntries.length === 0) {
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
          showToast(`Saving workspace bundle (${volumeEntries.length} NIfTI volumes + transforms + project.json)...`, 0);

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

          setCoregStatus("Ready", false);
          showToast(
            `Saved project "${projDirHandle.name}" (${volumeEntries.length} NIfTIs, ${transformFiles.length} .mat files, project.json)!`,
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

    let resp = await fetch("/api/save-project", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patientId,
        manifest,
        volumes: volumesPayload,
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
async function restoreProjectWorkspace(manifest, niftiBytesByName) {
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
    registerVolumeMeta(vol, {
      label: vEntry.label || vEntry.name,
      modality: vEntry.modality || "Neuro Volume",
      opacity: vEntry.savedOpacity ?? vEntry.opacity ?? 0.85,
      visible: vEntry.visible !== false,
      role: vEntry.role,
      isChildOfMov: Boolean(vEntry.isChildOfMov),
      isMask: Boolean(vEntry.isMask),
      isDktAtlas: Boolean(vEntry.isDktAtlas),
      atlasType: vEntry.atlasType || null,
      rawBytes,
    });

    const meta = state.layerMeta.get(vol.id);
    if (meta) {
      if (vEntry.origAffine) meta.origAffine = cloneMat4(vEntry.origAffine);
      if (vEntry.params) meta.params = { ...vEntry.params };
      if (vEntry.isCheckedForChildApply !== undefined) meta.isCheckedForChildApply = Boolean(vEntry.isCheckedForChildApply);
      if (vEntry.isCheckedForExport !== undefined) meta.isCheckedForExport = Boolean(vEntry.isCheckedForExport);
    }

    const volIdx = getVolumeIndexById(vol.id);
    if (volIdx >= 0 && vEntry.currentAffine) {
      setVolumeAffineMatrix(volIdx, vEntry.currentAffine, true);
    }

    if (vEntry.isDktAtlas || vEntry.colormap === "FreeSurferColorLUT") {
      await applyFreeSurferColorLutToVolume(vol);
      if (vEntry.isDktAtlas) {
        state.activeDktVolId = vol.id;
      }
    }

    if (vEntry.role === "reference") state.refVolId = vol.id;
    if (vEntry.role === "moving") state.movVolId = vol.id;
  }

  // If both a brain mask and DKT Atlas volume were restored, ensure the DKT overlay is brain-masked
  if (getDktVolume() && findActiveBrainMaskVolume()) {
    await applyBrainMaskToDktVolume(getDktVolume(), findActiveBrainMaskVolume());
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
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  renderSavedTransformsDropdown();
  renderLayerList();
  computeAndDisplayAlignmentMetrics(true);
  setCoregStatus("Ready", false);
  showToast(
    `Opened project for Patient "${manifest.patientId || "sub-01"}" (${state.nv.volumes.length} volumes restored)!`,
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

          await restoreProjectWorkspace(manifest, niftiBytesByName);
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
      await restoreProjectWorkspace(res.manifest, niftiBytesByName);
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

    for (const f of files) {
      if (f.name.endsWith(".nii.gz") || f.name.endsWith(".nii")) {
        niftiBytesByName.set(f.name, new Uint8Array(await f.arrayBuffer()));
      }
    }

    await restoreProjectWorkspace(manifest, niftiBytesByName);
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
    const resp = await fetch("/api/neuro-tools-status");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Status check failed");

    state.neuroToolsStatus = data;
    if (data.dktLut && Object.keys(data.dktLut).length > 0) {
      state.dktLut = data.dktLut;
    }

    if (betBadge) {
      const betOk = Boolean(data.fslBet?.available);
      betBadge.textContent = betOk ? "FSL BET Ready" : "BET Fallback (SciPy)";
      betBadge.className = `tool-mini-badge ${betOk ? "ready" : "fallback"}`;
    }

    if (fsBadge) {
      const fs = data.fastsurfer || {};
      if (fs.available) {
        fsBadge.textContent = fs.mode === "docker" ? "FastSurfer (Docker)" : "FastSurfer Ready";
        fsBadge.title = fs.statusLabel || fs.fastsurferHome || "FastSurfer Ready";
        fsBadge.className = "tool-mini-badge ready";
        if (fsNote) {
          if (fs.mode === "macos_native" && fs.fastsurferHome) {
            const folderName = fs.fastsurferHome.split(/[\\/]/).pop();
            fsNote.textContent = `FastSurfer Ready (${folderName}/macos_setup_fastsurfer.sh configured per MACOS.md).`;
          } else {
            fsNote.textContent = `FastSurfer detected (${fs.cliPath || "Docker"}). Click "Generate DKT Atlas" to segment T1w into 104 DKT regions.`;
          }
        }
      } else if (fs.fallbackAvailable) {
        fsBadge.textContent = "FastSurfer N/A · DKT Ready";
        fsBadge.className = "tool-mini-badge fallback";
        if (fsNote) {
          fsNote.textContent =
            "FastSurfer CLI not installed locally — using Template Registration + FreeSurfer 104-region DKT (aparc.DKTatlas+aseg) parcellation.";
        }
      } else {
        fsBadge.textContent = "FastSurfer Not Found";
        fsBadge.className = "tool-mini-badge checking";
      }
    }

    syncAtlasTypeSelectionUI(false, false);

    if (showToastMsg) {
      const fs = data.fastsurfer || {};
      const msg = fs.available
        ? `FastSurfer configured from ${fs.setupScript || fs.fastsurferHome || fs.statusLabel} & FSL BET ready!`
        : `FastSurfer CLI not installed locally — FSL BET & FreeSurfer 104-region DKT Atlas parcellation ready!`;
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

// Helper: Determine whether an atlas volume or type string is Destrieux ('destrieux', max ID 12175) vs DKT ('dkt', max ID 2035)
function resolveAtlasTypeForVolume(vol = null, explicitType = null) {
  if (explicitType) {
    const t = String(explicitType).toLowerCase();
    if (t.includes("destrieux") || t.includes("a2009s")) return "destrieux";
    if (t.includes("dkt")) return "dkt";
  }
  if (vol) {
    const meta = state.layerMeta.get(vol.id);
    if (meta?.atlasType) return meta.atlasType;
    if (/destrieux|a2009s/i.test(vol.name || "") || /destrieux|a2009s/i.test(meta?.label || "")) {
      return "destrieux";
    }
    const srcArr = meta?.unmaskedDktData || vol.img;
    if (srcArr && srcArr.length) {
      for (let i = 0; i < srcArr.length; i += 8) {
        if (srcArr[i] >= 11100) return "destrieux";
      }
    }
  }
  const sel = document.getElementById("selectAtlasType")?.value;
  return sel === "destrieux" ? "destrieux" : "dkt";
}

function getAtlasDefaultMaxLabel(vol = null, explicitType = null) {
  return resolveAtlasTypeForVolume(vol, explicitType) === "destrieux" ? 12175 : 2035;
}

function getAtlasShortTitle(vol = null, explicitType = null) {
  return resolveAtlasTypeForVolume(vol, explicitType) === "destrieux" ? "Destrieux Atlas" : "DKT Atlas";
}

// Synchronize the Atlas Tab UI (Pipeline options, Slow-Processing Warning Banner, Generate/Load buttons, and Active Atlas Volume)
function syncAtlasTypeSelectionUI(showSlowNoticeToast = false, switchActiveVolume = false) {
  const atlasSelect = document.getElementById("selectAtlasType");
  const modeSelect = document.getElementById("selectDktMode");
  const warnBanner = document.getElementById("destrieuxSlowWarning");
  const runBtn = document.getElementById("btnRunDktAtlas");
  const loadLbl = document.getElementById("lblLoadDktBtn");
  const detailNote = document.getElementById("fastsurferDetailNote");

  const atlasType = atlasSelect?.value === "destrieux" ? "destrieux" : "dkt";
  const isDestrieux = atlasType === "destrieux";
  state.selectedAtlasType = atlasType;

  if (modeSelect) {
    const optVinn = modeSelect.querySelector('option[value="fastsurfer_vinn"]');
    const optFull = modeSelect.querySelector('option[value="fastsurfer_full"]');
    const optTmpl = modeSelect.querySelector('option[value="template_flirt"]');
    if (isDestrieux) {
      if (optVinn) optVinn.textContent = "FastSurfer Surface + --fsaparc (aparc.a2009s, Slower)";
      if (optFull) optFull.textContent = "Full FastSurfer + --fsaparc (aparc.a2009s + Cereb/HypVINN, Slower)";
      if (optTmpl) optTmpl.textContent = "Quick Template Destrieux (FSL FLIRT ~2s)";
    } else {
      if (optVinn) optVinn.textContent = "FastSurferVINN DKT-Only (Fast, Recommended)";
      if (optFull) optFull.textContent = "Full FastSurfer (--seg_only + Cereb/HypVINN)";
      if (optTmpl) optTmpl.textContent = "Quick Template DKT (FSL FLIRT ~2s)";
    }
  }

  const modeVal = modeSelect?.value || "fastsurfer_vinn";
  const isSlowFsAparc = isDestrieux && modeVal !== "template_flirt";

  if (warnBanner) {
    warnBanner.hidden = !isSlowFsAparc;
  }

  // Update FreeSurfer License status indicator in the Destrieux notice banner
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
      licStatusEl.title = "No valid FreeSurfer license file automatically found in $FS_LICENSE, ~/license.txt, or $FREESURFER_HOME";
      licStatusEl.className = "fs-license-status is-missing";
      if (licBtnLbl) licBtnLbl.textContent = "Select License...";
    }
  }

  if (detailNote) {
    detailNote.textContent = isDestrieux
      ? "Destrieux (aparc.a2009s+aseg, 148 cortical gyral/sulcal + subcortical regions): runs FastSurfer with --fsaparc and without --seg_only, or reverts to MNI152 template registration when FastSurfer is unavailable."
      : "DKT (aparc.DKTatlas+aseg, 104 cortical + subcortical regions): runs FastSurferVINN (--seg_only) or MNI152 template registration for interactive cursor hover.";
  }

  if (runBtn && !runBtn.disabled) {
    runBtn.textContent = isDestrieux ? "Generate Destrieux Atlas" : "Generate DKT Atlas";
  }
  if (loadLbl) {
    loadLbl.textContent = isDestrieux ? "Load Destrieux" : "Load DKT";
  }

  if (switchActiveVolume && state.nv && Array.isArray(state.nv.volumes)) {
    const matchingAtlasVol = state.nv.volumes.find((v) => {
      const m = state.layerMeta.get(v.id);
      return m?.isDktAtlas && resolveAtlasTypeForVolume(v) === atlasType;
    });
    if (matchingAtlasVol) {
      state.activeDktVolId = matchingAtlasVol.id;
      // Show the selected atlas overlay and hide other loaded parcellation overlays so only one atlas overlay is visible at a time
      state.nv.volumes.forEach((v, idx) => {
        const m = state.layerMeta.get(v.id);
        if (m?.isDktAtlas) {
          const isSelected = v.id === matchingAtlasVol.id;
          m.visible = isSelected;
          const targetAlpha = isSelected ? (m.savedOpacity > 0 ? m.savedOpacity : 0.35) : 0;
          state.nv.setOpacity(idx, targetAlpha);
        }
      });
      state.nv.drawScene();
      renderLayerList();
      const dktInfo = lookupDktRegionAtRAS(state.lastCrosshairRAS);
      updateDktReadoutUI(dktInfo, null);
    } else {
      syncDktOverlayControlsUI();
    }
  } else {
    syncDktOverlayControlsUI();
  }

  if (showSlowNoticeToast && isSlowFsAparc) {
    showToast(
      "Destrieux (aparc.a2009s) selected: FastSurfer will run with --fsaparc and without --seg_only (full surface reconstruction — processing is significantly slower). Reverts to template registration only if FastSurfer is unavailable.",
      5800
    );
  }
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

// Generate DKT (aparc.DKTatlas+aseg) or Destrieux (aparc.a2009s+aseg) anatomical segmentation & add interactive atlas to workspace
async function runDktSegmentation() {
  if (!state.nv) return;
  const t1Select = document.getElementById("selectDktT1Vol") || document.getElementById("selectNeuroT1Vol");
  const targetVolId = t1Select?.value || state.refVolId;
  const volIdx = getVolumeIndexById(targetVolId);
  if (volIdx < 0) {
    showToast("Select a T1w structural volume first.");
    return;
  }

  const srcVol = state.nv.volumes[volIdx];
  const srcMeta = state.layerMeta.get(srcVol.id);
  const atlasType = document.getElementById("selectAtlasType")?.value === "destrieux" ? "destrieux" : "dkt";
  const isDestrieux = atlasType === "destrieux";
  const atlasShort = isDestrieux ? "Destrieux" : "DKT";
  const aparcName = isDestrieux ? "aparc.a2009s+aseg" : "aparc.DKTatlas+aseg";

  if (!srcMeta?.rawBytes) {
    showToast(`Selected volume does not have raw NIfTI bytes available for ${atlasShort} segmentation.`);
    return;
  }

  const btn = document.getElementById("btnRunDktAtlas");
  let dktMode = document.getElementById("selectDktMode")?.value || "fastsurfer_vinn";

  // If Destrieux + FastSurfer (--fsaparc) is selected, ensure tools status has been checked and verify FreeSurfer license
  if (isDestrieux && dktMode !== "template_flirt") {
    if (!state.neuroToolsStatus) {
      await checkNeuroToolsStatus(false);
    }
    const fsAvail = Boolean(state.neuroToolsStatus?.fastsurfer?.available);
    const licAvail = Boolean(
      state.neuroToolsStatus?.fsLicense?.available ||
      state.neuroToolsStatus?.fastsurfer?.fsLicense?.available
    );
    if (fsAvail && !licAvail) {
      const wantSelect = window.confirm(
        "FastSurfer --fsaparc (Destrieux aparc.a2009s surface pipeline) requires a valid FreeSurfer license file (license.txt / .license), and none was automatically found in standard paths ($FS_LICENSE, ~/license.txt, $FREESURFER_HOME/license.txt).\n\nClick OK to select your local FreeSurfer license file now, or click Cancel to run Quick Template Destrieux registration instead."
      );
      if (wantSelect) {
        state._pendingDestrieuxRunAfterLicense = true;
        document.getElementById("fsLicenseFileInput")?.click();
        return;
      } else {
        dktMode = "template_flirt";
      }
    }
  }
  const fracInput = document.getElementById("betFracInput");
  const betFrac = Math.max(0.05, Math.min(0.95, parseFloat(fracInput?.value || "0.5") || 0.5));
  const activeBrainVol = findActiveBrainMaskVolume();
  const activeBrainMeta = activeBrainVol ? state.layerMeta.get(activeBrainVol.id) : null;

  try {
    if (btn) {
      btn.disabled = true;
      btn.textContent = isDestrieux ? "Segmenting Destrieux..." : "Segmenting DKT...";
    }
    setCoregStatus(`Segmenting ${atlasShort}...`, true);
    if (isDestrieux && dktMode !== "template_flirt") {
      showToast(
        `Running FastSurfer with --fsaparc (without --seg_only) to generate Destrieux (${aparcName}) for ${srcMeta.label}. Please note: full surface reconstruction is slower (reverts to template registration only if FastSurfer is unavailable)...`,
        0
      );
    } else {
      showToast(
        `Running ${dktMode === "template_flirt" ? `Template ${atlasShort}` : `FastSurfer ${atlasShort}`} parcellation (${aparcName}) for ${srcMeta.label}...`,
        0
      );
    }

    const reqPayload = {
      sourceName: srcVol.name,
      sourceBase64: uint8ArrayToBase64(srcMeta.rawBytes),
      currentAffine: srcVol.hdr.affine,
      dktMode,
      atlasType,
      betFrac,
    };
    if (activeBrainMeta?.rawBytes) {
      reqPayload.brainMaskBase64 = uint8ArrayToBase64(activeBrainMeta.rawBytes);
      reqPayload.brainMaskAffine = activeBrainVol.hdr?.affine || null;
    }

    const resp = await fetch("/api/dkt-segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqPayload),
    });
    const data = await resp.json();
    if (!resp.ok || !data.niftiBase64) {
      throw new Error(data.error || `${atlasShort} segmentation failed`);
    }

    if (data.dktLut && Object.keys(data.dktLut).length > 0) {
      state.dktLut = data.dktLut;
    }

    const dktBytes = base64ToUint8Array(data.niftiBase64);
    const unmaskedBytes = data.unmaskedNiftiBase64 ? base64ToUint8Array(data.unmaskedNiftiBase64) : null;
    const resolvedType = data.atlasType || atlasType;
    const resolvedTitle = resolvedType === "destrieux" ? "Destrieux Atlas" : "DKT Atlas";
    await addDktAtlasVolumeToWorkspace(
      dktBytes,
      data.fileName,
      `${resolvedTitle} (${data.regionCount || (resolvedType === "destrieux" ? 185 : 104)} Regions)`,
      unmaskedBytes,
      resolvedType
    );

    setCoregStatus("Ready", false);
    showToast(
      `${resolvedTitle} loaded & brain-masked (${data.regionCount} anatomical regions via ${data.engine})! Hover cursor over brain to inspect ${resolvedTitle} areas.`,
      5200
    );
  } catch (err) {
    console.error(`${atlasShort} segmentation error:`, err);
    setCoregStatus("Ready", false);
    showToast(`${atlasShort} Segmentation Error: ${err.message}`, 4500);
  } finally {
    if (btn) {
      btn.disabled = false;
      syncAtlasTypeSelectionUI(false, false);
    }
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
  // Support both DKT (max 2035) and Destrieux aparc.a2009s+aseg (max 12175)
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
      // Translucent label alpha (72/255) so 3D volume raycasting renders brain anatomy without an opaque shell
      A[i] = 72;
      labels[i] = entry.name || `ROI_${i}`;
    } else {
      // Background (0), out-of-window labels (< cal_min or > cal_max), unmapped integers, and non-brain IDs are 100% transparent
      R[i] = 0;
      G[i] = 0;
      B[i] = 0;
      A[i] = 0;
      labels[i] = "";
    }
  }
  return { R, G, B, A, I, labels };
}

async function applyFreeSurferColorLutToVolume(vol) {
  if (!vol) return;
  const dktLutObj = await ensureDktLutLoaded();
  const meta = state.layerMeta.get(vol.id);

  // Register "FreeSurferColorLUT" as a named 256-entry NiiVue colormap where low/outside-brain bins are 100% transparent (alpha = 0)
  if (state.nv && typeof state.nv.addColormap === "function" && !state._fsColormapRegistered) {
    const cR = new Array(256).fill(0);
    const cG = new Array(256).fill(0);
    const cB = new Array(256).fill(0);
    const cA = new Array(256).fill(0);
    const cI = new Array(256).fill(0);
    const validEntries = Object.values(dktLutObj).filter((e) => e && isBrainFreeSurferLutEntry(e.id, e));
    // Bins 0..48 correspond to background air and outside-brain tissue intensities -> keep 100% transparent (alpha = 0)
    const transparentCutoffBin = 48;
    for (let k = 0; k < 256; k++) {
      cI[k] = k;
      if (k <= transparentCutoffBin) {
        cR[k] = 0;
        cG[k] = 0;
        cB[k] = 0;
        cA[k] = 0;
      } else {
        const pick = validEntries.length > 0 ? validEntries[k % validEntries.length] : null;
        cR[k] = pick ? pick.rgb[0] : (k * 73) % 200 + 55;
        cG[k] = pick ? pick.rgb[1] : (k * 151) % 200 + 55;
        cB[k] = pick ? pick.rgb[2] : (k * 199) % 200 + 55;
        // Smoothly ramp 3D raycast alpha from 12 to 88 so it never obscures the 3D brain volume render
        const t = (k - transparentCutoffBin) / (255 - transparentCutoffBin);
        cA[k] = Math.round(12 + t * 76);
      }
    }
    try {
      state.nv.addColormap("FreeSurferColorLUT", { R: cR, G: cG, B: cB, A: cA, I: cI });
      state._fsColormapRegistered = true;
    } catch (_) {
      // ignore
    }
  }

  // Determine whether `vol` is a discrete DKT/Destrieux/label volume vs a continuous structural MRI/CT volume
  const isDiscreteLabelVol = Boolean(
    meta?.isDktAtlas ||
      meta?.isMask ||
      /dkt|destrieux|a2009s|aparc|aseg|atlas|seg|mask/i.test(vol.name || "")
  );

  if (isDiscreteLabelVol) {
    let maxVox = 2100;
    if (vol.img && vol.img.length) {
      const srcArr = meta?.unmaskedDktData || vol.img;
      for (let i = 0; i < vol.img.length; i++) {
        const lbl = Math.round(srcArr[i]);
        // Zero out any non-brain / exterior / skull / CSF / unmapped label so it is 0 and transparent
        if (lbl <= 0 || !isBrainFreeSurferLutEntry(lbl, dktLutObj[String(lbl)])) {
          vol.img[i] = 0;
        } else if (lbl > maxVox) {
          maxVox = lbl;
        }
      }
    }

    if (meta && !meta.atlasType) {
      meta.atlasType = maxVox >= 11100 ? "destrieux" : resolveAtlasTypeForVolume(vol);
    }
    const defaultMax = maxVox >= 11100 ? 12175 : getAtlasDefaultMaxLabel(vol);

    if (!Number.isFinite(vol.cal_min) || vol.cal_min < 0.5) {
      vol.cal_min = 0.5;
    }
    if (!Number.isFinite(vol.cal_max) || vol.cal_max <= 0) {
      vol.cal_max = defaultMax;
    }

    const labelCmap = buildFreeSurferNiiVueLabelColormap(dktLutObj, maxVox, vol.cal_min, vol.cal_max);
    labelCmap.A[0] = 0;

    vol.colormap = "FreeSurferColorLUT";
    if (typeof vol.setColormapLabel === "function") {
      vol.setColormapLabel(labelCmap);
    }
  } else {
    // Continuous structural volume (e.g. T1w / FLAIR / CT):
    // Do NOT attach a 0..2100 integer label LUT to raw continuous MRI intensities (where outside-brain air/scalp noise is 1..2100!).
    // Instead, mask out non-brain voxels if a brain mask or DKT atlas is available, and set cal_min to the brain parenchyma threshold.
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
      // Compute brain intensity threshold so any residual outside-brain / background voxels fall into the transparent 0..48 LUT bins
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

// Find the active BET / skull-stripped brain mask volume in the workspace if present
function findActiveBrainMaskVolume() {
  if (!state.nv || !Array.isArray(state.nv.volumes)) return null;
  // 1. Prefer explicitly generated BET volume (meta.isBetBrain === true)
  const betVol = state.nv.volumes.find((v) => state.layerMeta.get(v.id)?.isBetBrain);
  if (betVol) return betVol;
  // 2. Fallback to any non-DKT volume whose filename contains "_brain"
  const brainNamedVol = state.nv.volumes.find((v) => {
    const m = state.layerMeta.get(v.id);
    if (m?.isDktAtlas) return false;
    return /_brain(\.nii(\.gz)?$|\b)/i.test(v.name || "");
  });
  return brainNamedVol || null;
}

// Mask the DKT / Destrieux Atlas volume in-place using the active brain mask volume so any voxel outside the brain is set to 0 and transparent
async function applyBrainMaskToDktVolume(dktVol = null, brainVolOverride = null) {
  const targetDktVol = dktVol || getDktVolume();
  if (!targetDktVol || !targetDktVol.img) return false;

  const dktLutObj = await ensureDktLutLoaded();
  const brainVol = brainVolOverride || findActiveBrainMaskVolume();
  if (!brainVol || !brainVol.img) return false;

  const dktMeta = state.layerMeta.get(targetDktVol.id);
  if (dktMeta && !dktMeta.unmaskedDktData) {
    dktMeta.unmaskedDktData = new Int16Array(targetDktVol.img);
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
    const len = dImg.length;
    for (let i = 0; i < len; i++) {
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

  targetDktVol.cal_min = 0.5;
  await applyFreeSurferColorLutToVolume(targetDktVol);
  return true;
}

async function addDktAtlasVolumeToWorkspace(dktBytes, fileName, displayLabel, unmaskedBytes = null, explicitAtlasType = null) {
  // Remove any previously loaded atlas with the same filename
  const existingIdx = state.nv.volumes.findIndex((v) => {
    const m = state.layerMeta.get(v.id);
    return m?.isDktAtlas && v.name === fileName;
  });
  if (existingIdx >= 0) {
    const oldId = state.nv.volumes[existingIdx].id;
    state.nv.removeVolumeByIndex(existingIdx);
    state.layerMeta.delete(oldId);
  }

  await ensureDktLutLoaded();

  const initialType = resolveAtlasTypeForVolume(null, explicitAtlasType || fileName);
  const initialMax = initialType === "destrieux" ? 12175 : 2035;

  const dktVol = await loadNVImageFromBytes(dktBytes, {
    name: fileName,
    colormap: "gray",
    opacity: 0.35,
    cal_min: 0.5,
    cal_max: initialMax,
  });

  let unmaskedDktData = null;
  if (unmaskedBytes) {
    try {
      const unmaskedVol = await loadNVImageFromBytes(unmaskedBytes, {
        name: `unmasked_${fileName}`,
        colormap: "gray",
        opacity: 0.0,
      });
      if (unmaskedVol?.img?.length === dktVol.img?.length) {
        unmaskedDktData = new Int16Array(unmaskedVol.img);
      }
    } catch (_) {
      // Fallback to dktVol.img below
    }
  }
  if (!unmaskedDktData && dktVol.img) {
    unmaskedDktData = new Int16Array(dktVol.img);
  }

  const detectedAtlasType = resolveAtlasTypeForVolume(dktVol, explicitAtlasType || fileName);
  const isDestrieux = detectedAtlasType === "destrieux";
  const atlasTitle = isDestrieux ? "Destrieux Atlas" : "DKT Atlas";
  dktVol.cal_max = isDestrieux ? 12175 : 2035;

  // Hide any other loaded parcellation overlays so only the newly activated atlas color overlay is displayed
  state.nv.volumes.forEach((v, idx) => {
    const m = state.layerMeta.get(v.id);
    if (m?.isDktAtlas) {
      m.visible = false;
      state.nv.setOpacity(idx, 0);
    }
  });

  await applyFreeSurferColorLutToVolume(dktVol);

  state.nv.addVolume(dktVol);
  registerVolumeMeta(dktVol, {
    label: displayLabel || `${atlasTitle} (${fileName})`,
    modality: `${atlasTitle} (FreeSurferColorLUT)`,
    opacity: 0.35,
    visible: true,
    isMask: true,
    isDktAtlas: true,
    atlasType: detectedAtlasType,
    rawBytes: dktBytes,
  });
  const dktMeta = state.layerMeta.get(dktVol.id);
  if (dktMeta && unmaskedDktData) {
    dktMeta.unmaskedDktData = unmaskedDktData;
  }
  state.activeDktVolId = dktVol.id;

  // Sync the Atlas Type dropdown to match the newly loaded/generated atlas
  const atlasSelect = document.getElementById("selectAtlasType");
  if (atlasSelect && atlasSelect.value !== detectedAtlasType) {
    atlasSelect.value = detectedAtlasType;
  }
  syncAtlasTypeSelectionUI(false, false);

  // If a brain mask volume is present in the stack, ensure the Atlas overlay is masked to it
  if (findActiveBrainMaskVolume()) {
    await applyBrainMaskToDktVolume(dktVol);
  }

  state.nv.updateGLVolume();
  state.nv.drawScene();
  renderLayerList();

  // Immediately evaluate Atlas region at current crosshair position
  const dktInfo = lookupDktRegionAtRAS(state.lastCrosshairRAS);
  updateDktReadoutUI(dktInfo, null);
}

async function loadDktAtlasFile(file) {
  if (!file) return;
  const selectedType = document.getElementById("selectAtlasType")?.value === "destrieux" ? "destrieux" : "dkt";
  try {
    setCoregStatus("Loading Atlas...", true);
    showToast(`Loading Atlas file ${file.name}...`, 0);

    // Ensure FreeSurfer/DKT/Destrieux LUT is loaded
    if (!state.dktLut || Object.keys(state.dktLut).length === 0) {
      await checkNeuroToolsStatus(false);
    }

    const ab = await file.arrayBuffer();
    let dktBytes = new Uint8Array(ab);
    let fileName = file.name;
    let detectedType = /destrieux|a2009s/i.test(file.name) ? "destrieux" : (/dkt/i.test(file.name) ? "dkt" : selectedType);

    if (file.name.toLowerCase().endsWith(".mgz") || file.name.toLowerCase().endsWith(".mgh")) {
      const resp = await fetch("/api/convert-volume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sourceName: file.name,
          sourceBase64: uint8ArrayToBase64(dktBytes),
          isDktAtlas: true,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.niftiBase64) {
        throw new Error(data.error || "Failed to convert FreeSurfer .mgz volume");
      }
      dktBytes = base64ToUint8Array(data.niftiBase64);
      fileName = data.fileName;
      if (data.dktLut) state.dktLut = data.dktLut;
      if (data.atlasType) detectedType = data.atlasType;
    }

    const atlasTitle = detectedType === "destrieux" ? "Destrieux Atlas" : "DKT Atlas";
    await addDktAtlasVolumeToWorkspace(dktBytes, fileName, `${atlasTitle} (${file.name})`, null, detectedType);
    setCoregStatus("Ready", false);
    showToast(`Loaded ${atlasTitle} "${file.name}"! Hover cursor over brain slices to view ${atlasTitle} anatomical areas.`);
  } catch (err) {
    console.error("Failed to load Atlas file:", err);
    setCoregStatus("Ready", false);
    showToast(`Load Atlas Error: ${err.message}`, 4500);
  }
}

// Find the active DKT / Destrieux Atlas NVImage volume in workspace
function getDktVolume() {
  if (!state.nv) return null;
  if (state.activeDktVolId) {
    const idx = getVolumeIndexById(state.activeDktVolId);
    if (idx >= 0) return state.nv.volumes[idx];
  }
  const preferredType = document.getElementById("selectAtlasType")?.value || state.selectedAtlasType || "dkt";
  const preferred = state.nv.volumes.find((v) => {
    const m = state.layerMeta.get(v.id);
    return m?.isDktAtlas && resolveAtlasTypeForVolume(v) === preferredType;
  });
  if (preferred) {
    state.activeDktVolId = preferred.id;
    return preferred;
  }
  const found = state.nv.volumes.find((v) => state.layerMeta.get(v.id)?.isDktAtlas);
  if (found) {
    state.activeDktVolId = found.id;
    return found;
  }
  return null;
}

// Sample the integer DKT / Destrieux label at a 3D RAS coordinate [x, y, z] and resolve via FreeSurferColorLUT
function lookupDktRegionAtRAS(ras) {
  const dktVol = getDktVolume();
  if (!dktVol || !dktVol.hdr?.affine || !dktVol.img) {
    return null;
  }
  const atlasTitle = getAtlasShortTitle(dktVol);
  const invAff = invertMat4(dktVol.hdr.affine);
  const rawVal = sampleVolumeAtRAS(dktVol, invAff, ras, true);
  const labelId = Math.round(rawVal);
  const entry = state.dktLut?.[String(labelId)] || null;
  if (!labelId || labelId <= 0 || !isBrainFreeSurferLutEntry(labelId, entry)) {
    return { labelId: 0, entry: null, ras, atlasTitle };
  }
  return { labelId, entry, ras, atlasTitle };
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

  if (!dktVol) {
    if (chk) {
      chk.checked = false;
      chk.disabled = true;
    }
    if (slider) {
      slider.value = "35";
      slider.disabled = true;
    }
    if (pctEl) pctEl.textContent = "35%";
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
  const isOverlayOn = Boolean(meta ? meta.visible && dktVol.opacity > 0 : dktVol.opacity > 0);
  const savedAlpha = meta?.savedOpacity > 0 ? meta.savedOpacity : 0.35;
  const activeAlpha = isOverlayOn ? dktVol.opacity : 0;

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
  if (subEl) subEl.textContent = `Click "Generate ${atlasTitle}" above, then hover cursor over brain`;
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

  const { labelId, entry, ras, atlasTitle: infoAtlasTitle } = dktInfo;
  const activeAtlasTitle = infoAtlasTitle || getAtlasShortTitle(getDktVolume());
  const rasStr = ras ? `RAS [${ras[0].toFixed(1)}, ${ras[1].toFixed(1)}, ${ras[2].toFixed(1)}] mm` : "";

  if (labelId > 0 && entry) {
    if (swatch) swatch.style.background = entry.hex || "#38bdf8";
    if (titleEl) titleEl.textContent = entry.title;
    if (subEl) subEl.textContent = `${entry.name} (${activeAtlasTitle} #${labelId}) · ${rasStr}`;

    if (hudPill) {
      hudPill.hidden = false;
      hudPill.innerHTML = `
        <span class="dkt-swatch" style="background:${entry.hex || "#38bdf8"};"></span>
        <span>${entry.title}</span>
        <span style="color:var(--text-secondary);font-family:var(--font-mono);font-size:10px;">#${labelId}</span>
      `;
    }

    if (tooltip && cursorEvent && wrapper) {
      const rect = wrapper.getBoundingClientRect();
      const relX = cursorEvent.clientX - rect.left;
      const relY = cursorEvent.clientY - rect.top;
      const maxX = Math.max(10, rect.width - 295);
      const maxY = Math.max(10, rect.height - 58);
      const tx = Math.min(maxX, Math.max(10, relX + 16));
      const ty = Math.min(maxY, Math.max(10, relY + 16));

      tooltip.innerHTML = `
        <span class="dkt-swatch" style="background:${entry.hex || "#38bdf8"};"></span>
        <div class="dkt-inspector-text">
          <div class="dkt-region-title">${entry.title}</div>
          <div class="dkt-region-sub">${entry.name} (#${labelId}) · ${rasStr}</div>
        </div>
      `;
      tooltip.style.transform = `translate3d(${Math.round(tx)}px, ${Math.round(ty)}px, 0)`;
      tooltip.hidden = false;
    }
  } else {
    // Cursor is over background (label 0) while an Atlas is loaded
    if (swatch) swatch.style.background = "#334155";
    if (titleEl) titleEl.textContent = "Outside Parcellated Brain (Background)";
    if (subEl) subEl.textContent = `${activeAtlasTitle} Label #0 · ${rasStr} — Hover over cortical/subcortical tissue`;
    if (tooltip) tooltip.hidden = true;
    if (hudPill) {
      hudPill.hidden = false;
      hudPill.innerHTML = `
        <span class="dkt-swatch" style="background:#475569;"></span>
        <span style="color:var(--text-secondary);">${activeAtlasTitle} Active — Hover over brain tissue</span>
      `;
    }
  }
}

// Convert mouse hover position on NiiVue canvas -> 3D RAS world coordinate -> DKT anatomical lookup
function handleCanvasMouseHover(e) {
  if (!state.nv) return;
  const dktVol = getDktVolume();
  if (!dktVol) return;

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
    // Try device-scaled canvas coordinates first (standard in NiiVue), then CSS coordinates if needed
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

  // Also update DKT anatomical area readout whenever the 3D crosshair moves
  if (getDktVolume()) {
    const dktInfo = lookupDktRegionAtRAS(ras);
    updateDktReadoutUI(dktInfo, null);
  }

  const tableEl = document.getElementById("voxelReadoutTable");
  if (!tableEl || !state.nv) return;

  tableEl.innerHTML = state.nv.volumes
    .map((vol) => {
      const meta = state.layerMeta.get(vol.id);
      const invAff = invertMat4(vol.hdr.affine);
      const val = sampleVolumeAtRAS(vol, invAff, ras, Boolean(meta?.isMask));
      const vox = applyMat4Pt(invAff, ras).map((v) => Math.round(v));
      if (meta?.isDktAtlas) {
        const lid = Math.round(val);
        const lutEntry = state.dktLut?.[String(lid)];
        const dktText = lid > 0 && lutEntry ? `${lutEntry.title} (#${lid})` : `0 (BG)`;
        return `
          <div class="voxel-row">
            <span class="voxel-name" title="${meta?.label || vol.name}">${meta?.label || vol.name}</span>
            <span class="voxel-val" style="color:${lutEntry?.hex || "#c4b5fd"};">${dktText}</span>
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
  document.getElementById("selectDktMode")?.addEventListener("change", () => {
    syncAtlasTypeSelectionUI(true, false);
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
  document.getElementById("sliderDktOverlayOpacity")?.addEventListener("input", (e) => {
    const dktVol = getDktVolume();
    if (!dktVol) return;
    const idx = getVolumeIndexById(dktVol.id);
    if (idx < 0) return;
    const val = Math.max(0, Math.min(1, parseFloat(e.target.value) / 100));
    const meta = state.layerMeta.get(dktVol.id);
    if (meta) {
      meta.visible = val > 0;
      if (val > 0) meta.savedOpacity = val;
    }
    state.nv.setOpacity(idx, val);
    state.nv.drawScene();
    renderLayerList();
  });

  const handleDktWindowInput = async () => {
    const dktVol = getDktVolume();
    if (!dktVol) return;
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
  window.addEventListener("resize", resizeWorkspaceCanvas);

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
    if (e.key === "Escape" && closeAppModal && !closeAppModal.hidden && !closeAppModal.classList.contains("hidden")) {
      closeCloseAppModal();
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
      state.nv.removeVolumeByIndex(idx);
      state.layerMeta.delete(volId);
      if (state.refVolId === volId) state.refVolId = null;
      if (state.movVolId === volId) state.movVolId = null;
      if (state.activeDktVolId === volId) {
        state.activeDktVolId = null;
        if (!getDktVolume()) {
          resetDktInspectorUI();
        }
      }
      renderLayerList();
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

  // 6. Verification Bar & Keyboard Shortcuts (Space = Blink, C = Toggle Coreg Panel, L = Toggle Layers Panel)
  document.getElementById("btnBlinkToggle")?.addEventListener("click", toggleBlinkMode);
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.code === "Space") {
      e.preventDefault();
      stepBlinkComparator();
    } else if (e.key === "c" || e.key === "C") {
      if (!e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        toggleCoregPanel();
      }
    } else if (e.key === "l" || e.key === "L") {
      if (!e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        toggleLayersPanel();
      }
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

initApp().catch((err) => {
  console.error("Failed to initialize NeuroCoreg Studio:", err);
  showToast(`Initialization Error: ${err.message}`, 6000);
});
