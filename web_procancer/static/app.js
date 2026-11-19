const grid = document.getElementById("grid");
const stats = document.getElementById("stats");
const completeOnly = document.getElementById("complete-only");
const hideDiscarded = document.getElementById("hide-discarded");
const search = document.getElementById("search");

let cases = [];

function previewUrl(c) {
  if (!c.preview) return null;
  const name = c.preview.split("/").pop();
  return `/previews/${encodeURIComponent(name)}`;
}

function render() {
  const q = search.value.trim().toLowerCase();
  const shown = cases.filter((c) => {
    if (completeOnly.checked && !c.complete) return false;
    if (hideDiscarded.checked && c.discarded) return false;
    if (!q) return true;
    return `${c.patient_id} ${c.study_instance_uid}`.toLowerCase().includes(q);
  });
  const kept = cases.filter((c) => c.complete && !c.discarded).length;
  stats.textContent = `${shown.length} shown · ${kept} complete kept · ${cases.length} total`;
  grid.innerHTML = "";
  for (const c of shown) {
    const card = document.createElement("article");
    card.className = "card" + (c.discarded ? " discarded" : "");
    const img = previewUrl(c);
    card.innerHTML = `
      ${img ? `<img src="${img}" alt="preview" loading="lazy" />` : `<div class="meta" style="padding:2rem;text-align:center">no preview</div>`}
      <div class="body">
        <div class="pid">${c.patient_id}</div>
        <div class="row">
          <span class="badge ${c.complete ? "ok" : "no"}">${c.complete ? "complete" : "incomplete"}</span>
          <span class="meta">${c.lesion_ml ?? "?"} mL · ${c.lesion_slices ?? "?"} slices</span>
        </div>
        ${c.missing?.length ? `<div class="missing">missing: ${c.missing.join(", ")}</div>` : ""}
        <div class="row">
          <button class="${c.discarded ? "keep" : "danger"}" data-act="toggle">
            ${c.discarded ? "keep" : "discard"}
          </button>
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

completeOnly.onchange = hideDiscarded.onchange = search.oninput = render;
boot();
