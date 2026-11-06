const $ = (selector) => document.querySelector(selector);
const binaryFields = ["headache", "myalgia", "abdominal_pain", "rash", "vomiting", "breathlessness", "bleeding", "low_blood_pressure", "organomegaly", "iv_fluids", "antibiotics", "antipyretics", "steroids"];
const numericFields = ["fever_days", "hb", "pcv", "lowest_wbc", "highest_wbc", "platelet_count", "alt", "creatinine", "ferritin"];
let latestBatchRows = [];

function setLoading(button, loading, label) {
  button.disabled = loading;
  button.dataset.original = button.dataset.original || button.innerHTML;
  button.innerHTML = loading ? `<span class="button-spinner"></span>${label}` : button.dataset.original;
}

document.querySelectorAll(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;
    document.querySelectorAll(".mode-tab").forEach((item) => item.classList.toggle("active", item === tab));
    $("#individual-view").hidden = mode !== "individual";
    $("#batch-view").hidden = mode !== "batch";
  });
});

function numericValue(id) {
  const value = $(`#${id}`).value.trim();
  return value === "" ? null : Number(value);
}

function collectFeatures() {
  const features = {
    age_over_6: Number($("#age_over_6").value),
    gender_female: Number($("#gender_female").value),
    fever_days: numericValue("fever_days"), hb: numericValue("hb"), pcv: numericValue("pcv"),
    lowest_wbc: numericValue("lowest_wbc"), highest_wbc: numericValue("highest_wbc"),
    platelet_count: numericValue("platelet_count"), alt: numericValue("alt"),
    creatinine: numericValue("creatinine"), ferritin: numericValue("ferritin")
  };
  binaryFields.forEach((field) => { features[field] = $(`[data-field="${field}"]`).checked ? 1 : 0; });
  return features;
}

function cleanFeatures(features) {
  const cleaned = { ...features };

  numericFields.forEach((field) => {
    const input = $(`#${field}`);
    const rawValue = input.value.trim().replace(",", ".");
    if (rawValue === "") {
      cleaned[field] = null;
      return;
    }

    const value = Number(rawValue);
    const min = input.min === "" ? null : Number(input.min);
    const max = input.max === "" ? null : Number(input.max);
    if (!Number.isFinite(value) || (min !== null && value < min) || (max !== null && value > max)) {
      throw new Error(`Revisa el valor de ${field.replaceAll("_", " ")}`);
    }
    cleaned[field] = value;
    input.value = String(value);
  });

  binaryFields.forEach((field) => { cleaned[field] = cleaned[field] === 1 ? 1 : 0; });
  cleaned.age_over_6 = cleaned.age_over_6 === 1 ? 1 : 0;
  cleaned.gender_female = cleaned.gender_female === 1 ? 1 : 0;

  if (cleaned.lowest_wbc !== null && cleaned.highest_wbc !== null && cleaned.lowest_wbc > cleaned.highest_wbc) {
    throw new Error("El WBC mínimo no puede ser mayor que el WBC máximo");
  }
  return cleaned;
}

function setCleaningMessage(message, error = false) {
  const element = $("#cleaning-message");
  element.textContent = message;
  element.classList.toggle("error", error);
}

function showModel(id, data) {
  $(`#${id}-percent`).textContent = data.percentage.toFixed(1);
  $(`#${id}-bar`).style.width = `${Math.max(2, data.percentage)}%`;
  $(`#${id}-label`).textContent = `${data.label} · nivel ${data.level}`;
}

function renderIndividual(data) {
  const consensus = data.consensus;
  $("#empty-state").hidden = true;
  $("#result-content").hidden = false;
  $("#consensus-percent").innerHTML = `${consensus.percentage.toFixed(1)}<small>%</small>`;
  $("#consensus-level").textContent = consensus.level;
  $("#consensus-message").textContent = consensus.message;
  $("#risk-badge").className = `risk-badge ${consensus.level}`;
  showModel("logistic", data.models.logistic);
  showModel("neural", data.models.neural_network);
  $("#result-status-text").textContent = "Completado";
  $(".result-status").classList.add("complete");
}

$("#risk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!event.target.checkValidity()) {
    event.target.reportValidity();
    return;
  }

  let features;
  try {
    features = cleanFeatures(collectFeatures());
  } catch (error) {
    setCleaningMessage(error.message, true);
    return;
  }

  const button = event.target.querySelector(".primary-button");
  const original = button.innerHTML;
  setLoading(button, true, "Procesando…");
  setCleaningMessage("Datos limpiados y validados. Ejecutando predicción…");
  try {
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "No fue posible ejecutar la predicción");
    renderIndividual(data);
    setCleaningMessage("Datos limpiados y predicción completada.");
  } catch (error) {
    setCleaningMessage(error.message, true);
    alert(`No fue posible completar la predicción: ${error.message}`);
  }
  finally { button.dataset.original = original; setLoading(button, false); }
});

const batchFile = $("#batch-file");
const dropZone = $("#drop-zone");
const batchButton = $("#batch-button");

function selectBatchFile(file) {
  if (!file) return;
  batchFile.files = (() => { const transfer = new DataTransfer(); transfer.items.add(file); return transfer.files; })();
  $("#file-name").textContent = file.name;
  $("#file-name").classList.add("visible");
  batchButton.disabled = false;
  $("#batch-message").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · listo para procesar`;
}

batchFile.addEventListener("change", () => selectBatchFile(batchFile.files[0]));
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => selectBatchFile(event.dataTransfer.files[0]));

function percent(value) { return value === null || value === undefined ? "—" : `${Number(value).toFixed(1)}%`; }
function classPill(label) { const high = label === "Complicado"; const neutral = label === "Sin dato"; return `<span class="result-pill ${high ? "high" : neutral ? "neutral" : ""}">${label}</span>`; }

function renderMatrix(target, matrix) {
  const rows = [["No complicado", matrix[0][0], matrix[0][1]], ["Complicado", matrix[1][0], matrix[1][1]]];
  $(target).innerHTML = rows.map((row) => `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td></tr>`).join("");
}

function renderBatch(data) {
  latestBatchRows = data.rows;
  $("#batch-results").hidden = false;
  $("#row-count").textContent = `· ${data.row_count} registros`;
  $("#no-target-note").hidden = data.has_target;
  $("#metrics-area").hidden = !data.has_target;
  $("#confusion-area").hidden = !data.has_target;
  if (data.has_target && data.metrics) {
    const logistic = data.metrics.logistic;
    const neural = data.metrics.neural_network;
    [["logistic", logistic], ["neural", neural]].forEach(([prefix, metrics]) => {
      $("#batch-" + prefix + "-accuracy").textContent = metrics.accuracy.toFixed(3);
      $("#batch-" + prefix + "-precision").textContent = metrics.precision.toFixed(3);
      $("#batch-" + prefix + "-recall").textContent = metrics.recall.toFixed(3);
      $("#batch-" + prefix + "-f1").textContent = metrics.f1.toFixed(3);
      $("#batch-" + prefix + "-auc").textContent = metrics.roc_auc === null ? "—" : metrics.roc_auc.toFixed(3);
    });
    renderMatrix("#logistic-matrix", logistic.confusion_matrix);
    renderMatrix("#neural-matrix", data.metrics.neural_network.confusion_matrix);
  }
  $("#actual-heading").hidden = !data.has_target;
  $("#results-body").innerHTML = data.rows.map((row) => `<tr><td>${row.registro}</td><td>${percent(row.logistic_percentage)} ${classPill(row.logistic_class)}</td><td>${percent(row.neural_percentage)} ${classPill(row.neural_class)}</td><td><strong>${percent(row.consensus_percentage)}</strong></td><td>${classPill(row.consensus_class)}</td>${data.has_target ? `<td>${classPill(row.actual_class)}</td>` : ""}</tr>`).join("");
}

batchButton.addEventListener("click", async () => {
  if (!batchFile.files[0]) return;
  const formData = new FormData();
  formData.append("file", batchFile.files[0]);
  setLoading(batchButton, true, "Procesando lote…");
  $("#batch-message").textContent = "Aplicando limpieza y modelos…";
  try {
    const response = await fetch("/api/predict-batch", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || "No fue posible procesar el archivo");
    renderBatch(data);
    $("#batch-message").textContent = "Procesamiento completado";
    $("#batch-results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) { $("#batch-message").textContent = error.message; }
  finally { setLoading(batchButton, false); }
});

$("#download-results").addEventListener("click", () => {
  if (!latestBatchRows.length) return;
  const columns = Object.keys(latestBatchRows[0]);
  const csv = [columns.join(","), ...latestBatchRows.map((row) => columns.map((column) => `"${String(row[column] ?? "").replaceAll('"', '""')}"`).join(","))].join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  link.download = "resultados_complicasense.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});
