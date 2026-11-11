(() => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const preview = document.getElementById("preview");
  const statusEl = document.getElementById("status");
  const btnMatch = document.getElementById("btn-match");
  const btnGenerate = document.getElementById("btn-generate");
  const resultPanel = document.getElementById("result-panel");
  const outEmotion = document.getElementById("out-emotion");
  const outEra = document.getElementById("out-era");
  const outWhy = document.getElementById("out-why");
  const outMode = document.getElementById("out-mode");
  const outImage = document.getElementById("out-image");
  const outCaption = document.getElementById("out-caption");
  const downloadLink = document.getElementById("download-link");

  let objectUrl = null;

  function setStatus(msg) {
    statusEl.textContent = msg || "";
  }

  function setBusy(busy) {
    btnMatch.disabled = busy;
    btnGenerate.disabled = busy;
  }

  function showPreview(file) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    preview.hidden = false;
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) showPreview(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });
  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    showPreview(file);
  });

  function fillMatch(match) {
    const emotion = match?.emotion?.dominant_emotion || "—";
    const era = match?.era || "—";
    outEmotion.textContent = emotion;
    outEra.textContent = era;
    outWhy.textContent = match?.why_this_era || "";
    resultPanel.hidden = false;
  }

  async function runMatch() {
    if (!fileInput.files?.length) {
      setStatus("Choose a selfie first.");
      return null;
    }
    setBusy(true);
    setStatus("Matching emotion × era…");
    const body = new FormData();
    body.append("file", fileInput.files[0]);
    const res = await fetch("/match", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    fillMatch(data);
    outMode.textContent = `match mode: ${data.match_mode || "—"}`;
    outImage.removeAttribute("src");
    outCaption.textContent = data.style_prompt || "";
    downloadLink.hidden = true;
    setStatus("Match ready.");
    return data;
  }

  async function runGenerate() {
    if (!fileInput.files?.length) {
      setStatus("Choose a selfie first.");
      return;
    }
    setBusy(true);
    setStatus("Matching + generating (first run may download/load SD1.5)…");
    const body = new FormData();
    body.append("file", fileInput.files[0]);
    const res = await fetch("/generate/json?steps=20&strength=0.72", {
      method: "POST",
      body,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    fillMatch(data.match || {});
    const gen = data.generation || {};
    outMode.textContent = `generate: ${gen.mode || "?"} · device ${gen.device || "?"} · ${gen.elapsed_sec ?? "?"}s · FaceID ${gen.faceid_used ? "on" : "stubbed"}`;
    if (gen.notes?.length) {
      outMode.textContent += ` · ${gen.notes[0]}`;
    }
    const src = `data:image/png;base64,${data.image_base64_png}`;
    outImage.src = src;
    outCaption.textContent = (data.match && data.match.style_prompt) || "";
    downloadLink.href = src;
    downloadLink.download = data.image_filename || "portrait.png";
    downloadLink.hidden = false;
    setStatus(`Done · saved ${data.image_filename || ""}`);
  }

  btnMatch.addEventListener("click", async () => {
    try {
      await runMatch();
    } catch (err) {
      setStatus(`Match failed: ${err.message || err}`);
    } finally {
      setBusy(false);
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await runGenerate();
    } catch (err) {
      setStatus(`Generate failed: ${err.message || err}`);
    } finally {
      setBusy(false);
    }
  });
})();
