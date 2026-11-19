(() => {
  const state = {
    patients: [],
    annotations: {},
    dirty: false,
    page: 0,
  };

  const el = {
    status: document.getElementById("status"),
    patients: document.getElementById("patients"),
    filter: document.getElementById("filter"),
    withSegOnly: document.getElementById("with-seg-only"),
    untaggedOnly: document.getElementById("untagged-only"),
    showDiscarded: document.getElementById("show-discarded"),
    lesionOnly: document.getElementById("lesion-only"),
    pageSize: document.getElementById("page-size"),
    pageLabel: document.getElementById("page-label"),
    prev: document.getElementById("prev"),
    next: document.getElementById("next"),
    save: document.getElementById("save"),
  };

  function setStatus(text) {
    el.status.textContent = text;
  }

  function thumbUrl(series) {
    if (!series.orthanc_series_id) return null;
    const params = new URLSearchParams({ size: "144" });
    if (series.modality === "SEG") {
      params.set("modality", "SEG");
    } else if (series.preview_instance_id) {
      params.set("instance_id", series.preview_instance_id);
    }
    return `/api/thumbnail/${encodeURIComponent(series.orthanc_series_id)}?${params}`;
  }

  function isTagged(series) {
    if (series.discarded) return true;
    if (series.modality === "MR") return Boolean(series.mr_tag);
    return (
      series.is_prostate_regions === true ||
      series.is_prostate_regions === false ||
      Boolean(series.lesion_seg)
    );
  }

  const LESION_TOKENS = ["lesion", "neoplasm", "heatmap", "cspca", "cancer", "tumor", "tumour"];

  function isLesionCandidate(series) {
    if (series.modality !== "SEG") return false;
    if (series.lesion_seg) return true;
    const blob = [series.series_description || "", ...(series.segment_labels || [])]
      .join(" ")
      .toLowerCase();
    return LESION_TOKENS.some((t) => blob.includes(t));
  }

  // T2 + ADC + DWI tagged and at least one SEG used as lesion label.
  function lesionReadiness(patient) {
    const live = (patient.series || []).filter((s) => !s.discarded);
    const has = (tag) => live.some((s) => s.modality === "MR" && s.mr_tag === tag);
    const lesion = live.filter((s) => s.lesion_seg === "binary" || s.lesion_seg === "prob60");
    const missing = ["t2", "adc", "dwi"].filter((t) => !has(t));
    if (!lesion.length) missing.push("lesion SEG");
    return { ready: missing.length === 0, missing, nLesion: lesion.length };
  }

  function visibleSeries(patient) {
    const series = patient.series || [];
    if (el.showDiscarded.checked) return series;
    return series.filter((s) => !s.discarded);
  }

  function patientMatches(patient, query, withSegOnly, untaggedOnly) {
    const series = visibleSeries(patient);
    if (!series.length) return false;
    if (withSegOnly && !series.some((s) => s.modality === "SEG")) return false;
    if (el.lesionOnly.checked && !(patient.series || []).some(isLesionCandidate)) return false;
    if (untaggedOnly && series.every(isTagged)) return false;
    if (!query) return true;
    const blob = [
      patient.patient_id,
      ...series.map((s) =>
        [s.modality, s.series_description, s.series_instance_uid, ...(s.segment_labels || [])].join(
          " "
        )
      ),
    ]
      .join("\n")
      .toLowerCase();
    return blob.includes(query);
  }

  function groupSeries(seriesList) {
    const byUid = new Map(seriesList.map((s) => [s.series_instance_uid, s]));
    const segs = seriesList.filter((s) => s.modality === "SEG");
    const bases = seriesList.filter((s) => s.modality !== "SEG");

    const usedSeg = new Set();
    const rows = [];

    for (const base of bases) {
      const linked = segs.filter((seg) =>
        (seg.referenced_series_uids || []).includes(base.series_instance_uid)
      );
      linked.forEach((seg) => usedSeg.add(seg.series_instance_uid));
      rows.push({ kind: linked.length ? "linked" : "single", items: [base, ...linked] });
    }

    for (const seg of segs) {
      if (usedSeg.has(seg.series_instance_uid)) continue;
      const missingRefs = (seg.referenced_series_uids || []).filter((uid) => !byUid.has(uid));
      rows.push({
        kind: missingRefs.length ? "orphan-seg" : "single",
        items: [seg],
      });
    }

    return rows;
  }

  function ensureAnnotation(uid) {
    if (!state.annotations[uid]) state.annotations[uid] = {};
    return state.annotations[uid];
  }

  function markDirty() {
    state.dirty = true;
    el.save.classList.remove("saved");
    el.save.textContent = "Save tags";
  }

  function onMrTagChange(uid, value) {
    const ann = ensureAnnotation(uid);
    ann.mr_tag = value || null;
    ann.discarded = false;
    delete ann.is_prostate_regions;
    markDirty();
  }

  function onProstateFlagChange(uid, value) {
    const ann = ensureAnnotation(uid);
    if (value === "true") ann.is_prostate_regions = true;
    else if (value === "false") ann.is_prostate_regions = false;
    else ann.is_prostate_regions = null;
    ann.discarded = false;
    delete ann.mr_tag;
    markDirty();
  }

  function setDiscarded(series, discarded) {
    series.discarded = discarded;
    const ann = ensureAnnotation(series.series_instance_uid);
    ann.discarded = discarded;
    if (discarded) {
      // Keep prior tags in annotations only if already saved; clear active pick.
      series.mr_tag = null;
      if (series.modality === "MR") {
        ann.mr_tag = null;
      }
    }
    markDirty();
    render();
  }

  function renderCard(series) {
    const card = document.createElement("article");
    card.className = `card ${series.modality === "SEG" ? "seg" : "mr"}${
      series.discarded ? " discarded" : ""
    }`;

    const thumb = document.createElement("div");
    thumb.className = "thumb-wrap";
    const url = thumbUrl(series);
    if (url) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = series.series_description || series.modality;
      img.src = url;
      img.onerror = () => {
        thumb.innerHTML = `<div class="placeholder">No preview</div>`;
      };
      thumb.appendChild(img);
    } else {
      thumb.innerHTML = `<div class="placeholder">No Orthanc id</div>`;
    }

    const body = document.createElement("div");
    body.className = "card-body";
    body.innerHTML = `
      <span class="badge ${series.modality === "SEG" ? "seg" : "mr"}">${series.modality}</span>
      <div class="desc" title="${escapeAttr(series.series_description || "")}">${escapeHtml(
        series.series_description || "(no description)"
      )}</div>
      <div class="uid" title="${escapeAttr(series.series_instance_uid)}">${escapeHtml(
        shortUid(series.series_instance_uid)
      )}</div>
    `;

    const actions = document.createElement("div");
    actions.className = "card-actions";

    if (series.modality === "MR") {
      const select = document.createElement("select");
      select.disabled = Boolean(series.discarded);
      select.innerHTML = `
        <option value="">tag…</option>
        <option value="t2">t2</option>
        <option value="adc">adc</option>
        <option value="dwi">dwi</option>
      `;
      select.value = series.mr_tag || "";
      select.addEventListener("change", () => {
        series.mr_tag = select.value || null;
        series.discarded = false;
        onMrTagChange(series.series_instance_uid, select.value);
        render();
      });
      actions.appendChild(select);
    } else {
      const select = document.createElement("select");
      select.disabled = Boolean(series.discarded);
      select.title = "Is prostate regions SEG?";
      select.innerHTML = `
        <option value="">prostate regions…</option>
        <option value="true">true</option>
        <option value="false">false</option>
      `;
      if (series.is_prostate_regions === true) select.value = "true";
      else if (series.is_prostate_regions === false) select.value = "false";
      else select.value = "";
      select.addEventListener("change", () => {
        if (select.value === "true") series.is_prostate_regions = true;
        else if (select.value === "false") series.is_prostate_regions = false;
        else series.is_prostate_regions = null;
        series.discarded = false;
        onProstateFlagChange(series.series_instance_uid, select.value);
        render();
      });
      actions.appendChild(select);

      if (isLesionCandidate(series)) {
        const lesion = document.createElement("select");
        lesion.disabled = Boolean(series.discarded);
        lesion.className = "lesion-select";
        lesion.title = "Use as lesion ground truth?";
        lesion.innerHTML = `
          <option value="">lesion label…</option>
          <option value="binary">lesion: binary mask</option>
          <option value="prob60">lesion: probability ≥ 60%</option>
          <option value="no">not a lesion label</option>
        `;
        lesion.value = series.lesion_seg || "";
        lesion.addEventListener("change", () => {
          series.lesion_seg = lesion.value || null;
          const ann = ensureAnnotation(series.series_instance_uid);
          ann.lesion_seg = series.lesion_seg;
          markDirty();
          render();
        });
        actions.appendChild(lesion);
      }
    }

    const discardBtn = document.createElement("button");
    discardBtn.type = "button";
    if (series.discarded) {
      discardBtn.className = "restore";
      discardBtn.textContent = "Restore";
      discardBtn.addEventListener("click", () => setDiscarded(series, false));
    } else {
      discardBtn.className = "discard";
      discardBtn.textContent = "Discard";
      discardBtn.title = "Remove from series of interest";
      discardBtn.addEventListener("click", () => setDiscarded(series, true));
    }
    actions.appendChild(discardBtn);
    body.appendChild(actions);

    if (series.modality === "SEG" && series.segment_labels && series.segment_labels.length) {
      const labs = document.createElement("div");
      labs.className = "uid";
      const extra = series.segment_labels.length > 4 ? ` · +${series.segment_labels.length - 4}` : "";
      labs.textContent = series.segment_labels.slice(0, 4).join(" · ") + extra;
      labs.title = series.segment_labels.join("\n");
      body.appendChild(labs);
    }
    if (series.modality === "SEG" && series.segmentation_type) {
      const t = document.createElement("div");
      t.className = `uid segtype ${series.segmentation_type.toLowerCase()}`;
      t.textContent = series.segmentation_type;
      body.appendChild(t);
    }

    card.appendChild(thumb);
    card.appendChild(body);
    return card;
  }

  function filteredPatients() {
    const query = el.filter.value.trim().toLowerCase();
    const withSegOnly = el.withSegOnly.checked;
    const untaggedOnly = el.untaggedOnly.checked;
    return state.patients.filter((p) =>
      patientMatches(p, query, withSegOnly, untaggedOnly)
    );
  }

  function render() {
    const matched = filteredPatients();
    const pageSize = Number(el.pageSize.value) || 50;
    const pageCount = Math.max(1, Math.ceil(matched.length / pageSize));
    if (state.page >= pageCount) state.page = pageCount - 1;
    if (state.page < 0) state.page = 0;
    const start = state.page * pageSize;
    const pageItems = matched.slice(start, start + pageSize);

    el.pageLabel.textContent = matched.length
      ? `${state.page + 1} / ${pageCount}`
      : "0 / 0";
    el.prev.disabled = state.page <= 0;
    el.next.disabled = state.page >= pageCount - 1;

    el.patients.innerHTML = "";
    if (!matched.length) {
      el.patients.innerHTML = `<div class="empty">No patients match the current filters.</div>`;
      setStatus(`Showing 0 / ${state.patients.length} patients`);
      return;
    }

    const frag = document.createDocumentFragment();
    for (const patient of pageItems) {
      const section = document.createElement("section");
      section.className = "patient";
      const shown = visibleSeries(patient);
      const nMr = shown.filter((s) => s.modality === "MR").length;
      const nSeg = shown.filter((s) => s.modality === "SEG").length;
      const nDiscarded = (patient.series || []).filter((s) => s.discarded).length;
      let lesionBadge = "";
      if ((patient.series || []).some(isLesionCandidate)) {
        const r = lesionReadiness(patient);
        lesionBadge = r.ready
          ? `<span class="ready yes">lesion dataset ✓ (${r.nLesion} lesion SEG)</span>`
          : `<span class="ready no">lesion dataset: missing ${escapeHtml(r.missing.join(", "))}</span>`;
      }
      section.innerHTML = `
        <div class="patient-head">
          <div class="patient-id">${escapeHtml(patient.patient_id)}</div>
          <div class="patient-meta">${nMr} MR · ${nSeg} SEG${
            nDiscarded ? ` · ${nDiscarded} discarded` : ""
          } ${lesionBadge}</div>
        </div>
      `;

      const rowsWrap = document.createElement("div");
      rowsWrap.className = "rows";
      for (const row of groupSeries(shown)) {
        const rowEl = document.createElement("div");
        rowEl.className = "row";
        for (const series of row.items) {
          rowEl.appendChild(renderCard(series));
        }
        rowsWrap.appendChild(rowEl);
      }
      section.appendChild(rowsWrap);
      frag.appendChild(section);
    }

    el.patients.appendChild(frag);
    setStatus(
      `Showing ${pageItems.length} on page ${state.page + 1}/${pageCount} · matched ${matched.length} / ${state.patients.length}`
    );
  }

  function collectAnnotations() {
    const series = {};
    for (const patient of state.patients) {
      for (const s of patient.series) {
        const uid = s.series_instance_uid;
        const fromState = state.annotations[uid] || {};
        const item = {};
        if (s.discarded || fromState.discarded) item.discarded = true;
        if (s.modality === "MR") {
          const tag = s.discarded ? fromState.mr_tag : s.mr_tag;
          if (tag) item.mr_tag = tag;
          else item.mr_tag = null;
        } else if (
          s.is_prostate_regions === true ||
          s.is_prostate_regions === false ||
          fromState.is_prostate_regions === true ||
          fromState.is_prostate_regions === false
        ) {
          item.is_prostate_regions =
            s.is_prostate_regions === true || s.is_prostate_regions === false
              ? s.is_prostate_regions
              : fromState.is_prostate_regions;
        }
        if (s.modality === "SEG" && (s.lesion_seg || fromState.lesion_seg)) {
          item.lesion_seg = s.lesion_seg || fromState.lesion_seg;
        }
        if (
          item.discarded ||
          item.mr_tag ||
          item.lesion_seg ||
          item.is_prostate_regions === true ||
          item.is_prostate_regions === false
        ) {
          series[uid] = item;
        }
      }
    }
    return series;
  }

  async function save() {
    el.save.disabled = true;
    try {
      const series = collectAnnotations();
      const res = await fetch("/api/annotations", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ series }),
      });
      if (!res.ok) throw new Error(await res.text());
      state.annotations = series;
      state.dirty = false;
      el.save.classList.add("saved");
      el.save.textContent = "Saved";
    } catch (err) {
      setStatus(`Save failed: ${err}`);
    } finally {
      el.save.disabled = false;
    }
  }

  function shortUid(uid) {
    if (!uid) return "";
    if (uid.length <= 28) return uid;
    return `${uid.slice(0, 12)}…${uid.slice(-10)}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replaceAll("`", "&#96;");
  }

  async function init() {
    const resetPageAndRender = () => {
      state.page = 0;
      render();
    };
    el.filter.addEventListener("input", resetPageAndRender);
    el.withSegOnly.addEventListener("change", resetPageAndRender);
    el.untaggedOnly.addEventListener("change", resetPageAndRender);
    el.showDiscarded.addEventListener("change", resetPageAndRender);
    el.lesionOnly.addEventListener("change", resetPageAndRender);
    el.pageSize.addEventListener("change", resetPageAndRender);
    el.prev.addEventListener("click", () => {
      state.page -= 1;
      render();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    el.next.addEventListener("click", () => {
      state.page += 1;
      render();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    el.save.addEventListener("click", () => save());
    window.addEventListener("beforeunload", (event) => {
      if (!state.dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });

    try {
      setStatus("Loading manifest…");
      const res = await fetch("/api/manifest");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      state.patients = data.patients || [];
      const annRes = await fetch("/api/annotations");
      const ann = annRes.ok ? await annRes.json() : { series: {} };
      state.annotations = ann.series || {};
      // Apply discarded flags from saved annotations onto in-memory series.
      for (const patient of state.patients) {
        for (const s of patient.series) {
          const saved = state.annotations[s.series_instance_uid];
          if (saved && saved.discarded) s.discarded = true;
        }
      }
      setStatus(
        `${data.patient_count} patients · ${data.series_count} series · generated ${data.generated_at || "?"}`
      );
      render();
    } catch (err) {
      setStatus(`Failed to load manifest: ${err}`);
      el.patients.innerHTML = `<div class="empty">Build the manifest first:<br><code>python scripts/build_series_manifest.py</code></div>`;
    }
  }

  init();
})();
