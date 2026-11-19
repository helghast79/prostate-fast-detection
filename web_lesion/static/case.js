const params = new URLSearchParams(location.search);
const pid = params.get("patient_id") || "";
const sid = params.get("study_instance_uid") || "";

const title = document.getElementById("title");
const metaEl = document.getElementById("meta");
const stats = document.getElementById("stats");
const slicesEl = document.getElementById("slices");
const onlyMask = document.getElementById("only-mask");
const discardBtn = document.getElementById("discard-btn");

let caseMeta = null;
let nSlices = 0;
let hasMask = [];

function sliceUrl(z) {
  return `/api/case/${encodeURIComponent(pid)}/${encodeURIComponent(sid)}/slice/${z}.png`;
}

function renderSlices() {
  slicesEl.innerHTML = "";
  let shown = 0;
  for (let z = 0; z < nSlices; z++) {
    if (onlyMask.checked && !hasMask[z]) continue;
    shown += 1;
    const fig = document.createElement("figure");
    fig.className = "slice" + (hasMask[z] ? " has-mask" : "");
    fig.innerHTML = `
      <img src="${sliceUrl(z)}" alt="slice ${z}" loading="lazy" />
      <figcaption>${z}${hasMask[z] ? " · mask" : ""}</figcaption>`;
    slicesEl.appendChild(fig);
  }
  stats.textContent = `${shown} / ${nSlices} slices`;
}

async function setDiscarded(discarded) {
  const res = await fetch("/api/case", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: pid,
      study_instance_uid: sid,
      discarded,
    }),
  });
  if (!res.ok) return;
  caseMeta.discarded = discarded;
  discardBtn.textContent = discarded ? "keep" : "discard";
  discardBtn.className = discarded ? "keep" : "danger";
}

async function boot() {
  if (!pid || !sid) {
    metaEl.textContent = "Missing patient_id / study_instance_uid query params.";
    return;
  }
  title.textContent = pid;
  document.title = `Slices · ${pid.slice(0, 24)}`;
  metaEl.textContent = "Loading T2 + lesion SEG overlays…";
  const res = await fetch(`/api/case/${encodeURIComponent(pid)}/${encodeURIComponent(sid)}`);
  const data = await res.json();
  if (!res.ok || !data.ok) {
    metaEl.textContent = data.error || "Failed to load case";
    return;
  }
  caseMeta = data.case;
  nSlices = data.n_slices;
  hasMask = data.has_mask || [];
  const t2 = caseMeta.t2?.series_description || "—";
  const seg = caseMeta.lesion_seg?.series_description || "—";
  metaEl.textContent =
    `${sid} · T2: ${t2} · SEG: ${seg} · lesion ${caseMeta.lesion_voxels ?? "?"}` +
    ` · matched ${data.matched_planes ?? "?"}/${(data.matched_planes ?? 0) + (data.unmatched_planes ?? 0)}`;
  discardBtn.textContent = caseMeta.discarded ? "keep" : "discard";
  discardBtn.className = caseMeta.discarded ? "keep" : "danger";
  renderSlices();
}

onlyMask.onchange = renderSlices;
discardBtn.onclick = () => setDiscarded(!caseMeta?.discarded);
boot();
