const grid = document.getElementById("grid");
const stats = document.getElementById("stats");
const readyOnly = document.getElementById("ready-only");
const hideDiscarded = document.getElementById("hide-discarded");
const search = document.getElementById("search");

let cases = [];

function previewUrl(c) {
  if (!c.preview) return null;
  return `/previews/${encodeURIComponent(c.preview.split("/").pop())}`;
}

function render() {
  const q = search.value.trim().toLowerCase();
  const shown = cases.filter((c) => {
    if (readyOnly.checked && !c.ready) return false;
    if (hideDiscarded.checked && c.discarded) return false;
    if (!q) return true;
    const blob = `${c.patient_id} ${c.study_instance_uid} ${c.t2?.series_description || ""} ${c.zone_seg?.series_description || ""}`.toLowerCase();
    return blob.includes(q);
  });
  const kept = cases.filter((c) => c.ready && !c.discarded).length;
  stats.textContent = `${shown.length} shown · ${kept} ready kept · ${cases.length} total`;
  grid.innerHTML = "";
  for (const c of shown) {
    const card = document.createElement("article");
    card.className = "card" + (c.discarded ? " discarded" : "");
    const img = previewUrl(c);
    card.innerHTML = `
      ${img ? `<img src="${img}" alt="preview" loading="lazy" />` : `<div class="meta" style="padding:2rem;text-align:center">${c.preview_error || "no preview"}</div>`}
      <div class="body">
        <div class="pid">${c.patient_id}</div>
        <div class="row">
          <span class="badge ${c.ready ? "ok" : "no"}">${c.ready ? "ready" : "incomplete"}</span>
          <span class="meta">gland ${c.gland_voxels ?? "?"} · pz ${c.pz_voxels ?? "?"}</span>
        </div>
        <div class="legend"><span class="g">● gland</span><span class="p">● PZ</span></div>
        <div class="desc">T2: ${c.t2?.series_description || "—"}</div>
        <div class="desc">SEG: ${c.zone_seg?.series_description || "—"}</div>
        <div class="row">
          <button class="${c.discarded ? "keep" : "danger"}" data-act="toggle">
            ${c.discarded ? "keep" : "discard"}
          </button>
          <a class="button-link" data-act="slices" target="_blank" rel="noopener"
             href="/case?patient_id=${encodeURIComponent(c.patient_id)}&study_instance_uid=${encodeURIComponent(c.study_instance_uid)}">
            review slices
          </a>
        </div>
      </div>`;
    card.querySelector("[data-act=toggle]").onclick = async () => {
      const discarded = !c.discarded;
      const res = await fetch("/api/case", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: c.patient_id,
          study_instance_uid: c.study_instance_uid,
          discarded,
        }),
      });
      if (!res.ok) return;
      c.discarded = discarded;
      render();
    };
    grid.appendChild(card);
  }
}

async function boot() {
  const res = await fetch("/api/catalog");
  const data = await res.json();
  cases = data.cases || [];
  render();
}

readyOnly.onchange = hideDiscarded.onchange = search.oninput = render;
boot();
