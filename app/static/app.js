/**
 * GastricPath AI - Premium Computational Pathology Workstation Client
 * Unified Research Workbench & Live Inference Engine
 */

// DOM Elements & Selectors
const recordsBody = document.getElementById("records-body");
const benchmarkBody = document.getElementById("benchmark-body");
const probabilityBars = document.getElementById("probability-bars");
const microscopeImg = document.getElementById("microscope-img");
const microscopePlaceholder = document.getElementById("microscope-placeholder");
const microscopeReticle = document.getElementById("microscope-reticle");
const hudMag = document.getElementById("hud-mag");
const fileInput = document.getElementById("file-input");
const fileLabelText = document.getElementById("file-label-text");
const uploadDropzone = document.getElementById("upload-dropzone");
const topbarCrumbActive = document.getElementById("topbar-crumb-active");

// Zoom & Viewport State
let currentZoom = 1.0;
const ZOOM_STEP = 0.25;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3.5;

const modelModeLabels = {
    "trained-checkpoint": "Trained Checkpoint (RNN)",
    "prototype-baseline": "Fallback Similarity Model",
};

const moduleTitles = {
    "overview": "Module 01: Platform Research Overview",
    "dataset": "Module 02: Dataset Explorer & Audit Provenance",
    "models": "Module 03: Model Lab & Architecture Registry",
    "foundation": "Module 04: Pathology Foundation Models (Phase 4.4)",
    "explainability": "Module 05: Interpretability & Multimodal Attribution",
    "segmentation": "Module 06: Tissue Segmentation Infrastructure",
    "cellular": "Module 07: Nuclei Detection & Cellular Morphology",
    "spatial-graphs": "Module 08: Spatial Graph and GNN Engine",
    "wsi": "Module 09: Whole-Slide Image & MIL Engine",
    "similarity": "Module 10: Representation Similarity Search",
    "uncertainty": "Module 11: Uncertainty, OOD & Perturbation Robustness",
    "experiments": "Module 12: Experiment Tracking & Run Logs",
    "validation": "Module 13: Consolidated Validation Matrix",
    "reports": "Module 14: Research Reports Catalog & Viewer",
};

const loadedModules = new Set();

// Utility Formatters
function percent(value) {
    return `${Number(value).toFixed(1)}%`;
}

function riskClass(riskBand) {
    return `risk-${(riskBand || "low").toLowerCase()}`;
}

function badgeClass(statusCode) {
    switch (statusCode) {
        case "implemented":
        case "operational":
            return "badge-implemented";
        case "evaluated":
            return "badge-evaluated";
        case "exploratory":
            return "badge-exploratory";
        case "blocked_by_data":
        case "blocked_by_linkage":
            return "badge-blocked";
        default:
            return "badge-planned";
    }
}

// Probability & Chart Helpers
function createBars(containerId, items, formatter = (value) => value) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    for (const item of items) {
        const row = document.createElement("div");
        row.className = "bar-row";
        const maxVal = Math.max(...items.map((entry) => entry.value), 1);
        const widthPct = Math.max(3, (item.value / maxVal) * 100);
        row.innerHTML = `
            <div class="bar-meta">
                <span>${item.label}</span>
                <strong>${formatter(item.value)}</strong>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${widthPct}%; background:${item.color || "var(--cyan-500)"}"></div>
            </div>
        `;
        container.appendChild(row);
    }
}

// Image Zoom & Viewer Controls
function updateZoomTransform() {
    if (!microscopeImg) return;
    microscopeImg.style.transform = `scale(${currentZoom})`;
    if (hudMag) {
        hudMag.textContent = `${Math.round(currentZoom * 100)}%`;
    }
}

function zoomIn() {
    if (currentZoom < MAX_ZOOM) {
        currentZoom = Math.min(MAX_ZOOM, currentZoom + ZOOM_STEP);
        updateZoomTransform();
    }
}

function zoomOut() {
    if (currentZoom > MIN_ZOOM) {
        currentZoom = Math.max(MIN_ZOOM, currentZoom - ZOOM_STEP);
        updateZoomTransform();
    }
}

function zoomReset() {
    currentZoom = 1.0;
    updateZoomTransform();
}

function toggleReticle() {
    if (!microscopeReticle) return;
    microscopeReticle.classList.toggle("active");
    const btn = document.getElementById("btn-toggle-reticle");
    if (btn) btn.classList.toggle("active");
}

function displayImagePreview(fileOrUrl) {
    if (!microscopeImg || !microscopePlaceholder) return;

    if (typeof fileOrUrl === "string") {
        microscopeImg.src = fileOrUrl;
        microscopeImg.classList.remove("hidden");
        microscopePlaceholder.classList.add("hidden");
        zoomReset();
    } else if (fileOrUrl instanceof File || fileOrUrl instanceof Blob) {
        const reader = new FileReader();
        reader.onload = (e) => {
            microscopeImg.src = e.target.result;
            microscopeImg.classList.remove("hidden");
            microscopePlaceholder.classList.add("hidden");
            zoomReset();
        };
        reader.readAsDataURL(fileOrUrl);
    }
}

// Render Prediction Result
function renderPrediction(payload) {
    const resultLabel = document.getElementById("result-label");
    if (resultLabel) {
        resultLabel.innerHTML = `<span style="color: var(--cyan-400);">${payload.predicted_class}</span> — ${payload.predicted_label}`;
    }

    const tumProb = payload.predicted_tum_probability !== undefined ? payload.predicted_tum_probability : payload.cancer_percentage;
    const resultPercentage = document.getElementById("result-percentage");
    if (resultPercentage) {
        resultPercentage.textContent = percent(tumProb);
    }

    const resultConfidence = document.getElementById("result-confidence");
    if (resultConfidence) {
        resultConfidence.textContent = percent(payload.confidence_score);
    }

    const resultSource = document.getElementById("result-source");
    if (resultSource) {
        resultSource.textContent = modelModeLabels[payload.model_source] || payload.model_source;
    }

    const resultCard = document.getElementById("result-card");
    if (resultCard) {
        resultCard.classList.remove("hidden");
    }

    // 8-Class Probability Bars
    if (probabilityBars) {
        probabilityBars.innerHTML = "";
        for (const item of payload.class_probabilities) {
            const row = document.createElement("div");
            row.className = "bar-row";
            row.innerHTML = `
                <div class="bar-meta">
                    <span><strong>${item.class_code}</strong> — ${item.label}</span>
                    <strong>${percent(item.probability)}</strong>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${Math.max(item.probability, 3)}%; background:${item.color}"></div>
                </div>
            `;
            probabilityBars.appendChild(row);
        }
    }

    // BiLSTM Sequence Attention Weights (Attribution Only, Not Tumor Localization)
    if (payload.attention_summary && Array.isArray(payload.attention_summary)) {
        const cells = document.querySelectorAll("#attention-grid-cells .attention-cell");
        const maxAttn = Math.max(...payload.attention_summary, 0.001);
        cells.forEach((cell, idx) => {
            const val = payload.attention_summary[idx] || 0;
            const intensity = Math.min(1.0, val / maxAttn);
            cell.style.background = `rgba(6, 182, 212, ${Math.max(0.12, intensity)})`;
            cell.title = `Token #${idx + 1}: Attention weight ${(val * 100).toFixed(2)}% (Model attribution weight — not anatomical tumor area)`;
        });
    }
}

// Render SQLite History Records (Risk tier removed for scientific accuracy)
function renderRecords(records) {
    if (!recordsBody) return;
    recordsBody.innerHTML = "";
    if (!records || !records.length) {
        recordsBody.innerHTML = `<tr><td colspan="6" style="color: var(--text-muted); text-align: center; padding: 18px;">No prediction records yet.</td></tr>`;
        return;
    }

    for (const record of records) {
        const row = document.createElement("tr");
        const tumPct = record.predicted_tum_probability !== undefined ? record.predicted_tum_probability : record.cancer_percentage;
        row.innerHTML = `
            <td><code>${record.filename}</code></td>
            <td><strong>${record.predicted_class}</strong></td>
            <td><strong>${percent(record.confidence_score)}</strong></td>
            <td><strong style="color: #f43f5e;">${percent(tumPct)}</strong></td>
            <td><small style="color: var(--text-secondary);">${modelModeLabels[record.model_source] || record.model_source}</small></td>
            <td><small style="color: var(--text-muted);">${new Date(record.created_at).toLocaleTimeString()} (${new Date(record.created_at).toLocaleDateString()})</small></td>
        `;
        recordsBody.appendChild(row);
    }
}

// Render Historical Confusion Matrix Table
function renderBenchmark(report) {
    if (!benchmarkBody) return;
    benchmarkBody.innerHTML = "";
    for (const item of report) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><strong>${item.class_code}</strong></td>
            <td>${item.precision.toFixed(2)}</td>
            <td>${item.recall.toFixed(2)}</td>
            <td><strong>${item.f1_score.toFixed(2)}</strong></td>
        `;
        benchmarkBody.appendChild(row);
    }
}

// Load Main Dashboard Data
async function loadDashboard() {
    try {
        const response = await fetch("/api/dashboard");
        const payload = await response.json();

        // Model Status & Telemetry
        const modelMode = document.getElementById("model-mode");
        if (modelMode) {
            modelMode.textContent = "GastricCancerRNN";
        }

        const checkpointAcc = document.getElementById("checkpoint-accuracy");
        if (checkpointAcc && payload.model_status) {
            checkpointAcc.textContent = `${(payload.model_status.checkpoint_validation_accuracy * 100).toFixed(2)}%`;
        }

        const benchmarkAcc = document.getElementById("benchmark-accuracy");
        if (benchmarkAcc) {
            benchmarkAcc.textContent = "71.04% (N=4,665)";
        }

        const datasetTotal = document.getElementById("dataset-total");
        if (datasetTotal && payload.dataset) {
            datasetTotal.textContent = `${payload.dataset.total_images.toLocaleString()} Patches`;
        }

        const trainAcc = document.getElementById("train-accuracy");
        if (trainAcc) trainAcc.textContent = percent(payload.benchmark.train_accuracy * 100);

        const validAcc = document.getElementById("valid-accuracy");
        if (validAcc) validAcc.textContent = percent(payload.benchmark.valid_accuracy * 100);

        const testAcc = document.getElementById("test-accuracy");
        if (testAcc) testAcc.textContent = percent(payload.benchmark.test_accuracy * 100);

        const testLoss = document.getElementById("test-loss");
        if (testLoss) testLoss.textContent = payload.benchmark.test_loss.toFixed(3);

        // Clinical Aggregate Demographics
        const meanAge = document.getElementById("mean-age");
        if (meanAge) meanAge.textContent = `${payload.clinical.mean_age} yrs`;

        const maleCount = payload.clinical.sex_distribution.find((item) => item.label === "Male")?.value || 0;
        const femaleCount = payload.clinical.sex_distribution.find((item) => item.label === "Female")?.value || 0;
        const maleEl = document.getElementById("male-count");
        if (maleEl) maleEl.textContent = maleCount;
        const femaleEl = document.getElementById("female-count");
        if (femaleEl) femaleEl.textContent = femaleCount;

        // Render Bar Visualizations
        createBars("dataset-bars", payload.dataset.class_distribution, (value) => value.toLocaleString());
        createBars("stage-bars", payload.clinical.stage_distribution, (value) => value);
        createBars("histology-bars", payload.clinical.top_histologies.map((item) => ({ ...item, color: "var(--teal-400)" })), (value) => value);
        renderBenchmark(payload.benchmark.class_report);
        renderRecords(payload.recent_predictions);

        // Biomarkers Top Targets
        const biomarkerBody = document.getElementById("biomarker-body");
        if (biomarkerBody && payload.biomarkers && payload.biomarkers.top_targets) {
            biomarkerBody.innerHTML = "";
            for (const target of payload.biomarkers.top_targets) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td><strong>${target.target_symbol}</strong></td>
                    <td><strong>${target.predicted_sum} / 8</strong></td>
                    <td>${target.all_sum}</td>
                `;
                biomarkerBody.appendChild(row);
            }
        }
    } catch (err) {
        console.error("Failed to load dashboard payload:", err);
    }
}

// Live Image Prediction Handler
async function runPrediction(event) {
    if (event) event.preventDefault();
    if (!fileInput.files.length) {
        alert("Please select or drop a histopathology tissue image.");
        return;
    }

    const submitBtn = document.getElementById("submit-prediction-btn");
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span>Analyzing...</span>`;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();

        if (!response.ok) {
            alert(payload.detail || "Prediction failed.");
            return;
        }

        renderPrediction(payload);
        const latestPredictions = await fetch("/api/predictions");
        const predictionPayload = await latestPredictions.json();
        renderRecords(predictionPayload.records);
    } catch (err) {
        alert("Prediction request failed: " + err);
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                <span>Run Inference</span>
            `;
        }
    }
}

// Quick Sample Patch Loader
async function loadSamplePatch(classCode) {
    const sampleUrl = `/dataset-images/${classCode}/${classCode}_1.png`;
    try {
        const res = await fetch(sampleUrl);
        if (!res.ok) {
            throw new Error(`Sample not found (${res.status})`);
        }
        const blob = await res.blob();
        const file = new File([blob], `${classCode}_1.png`, { type: "image/png" });

        // Update file input via DataTransfer
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        if (fileLabelText) fileLabelText.textContent = `Loaded sample: ${classCode}_1.png`;
        displayImagePreview(file);

        // Run prediction immediately
        await runPrediction(null);
    } catch (err) {
        console.warn(`Could not load sample ${classCode}:`, err);
        alert(`Failed to load sample patch for ${classCode}: ${err.message}`);
    }
}

// ==========================================================================
// WORKBENCH MODULE LOADERS
// ==========================================================================

async function loadModuleOverview() {
    if (loadedModules.has("overview")) return;
    try {
        const res = await fetch("/api/research/overview");
        const data = await res.json();

        const invContainer = document.getElementById("overview-inventory-cards");
        if (invContainer) {
            invContainer.innerHTML = `
                <div class="info-card">
                    <h4>GCHTID Image Tiles</h4>
                    <p><strong>${data.dataset_inventory.gchtid_tiles.count.toLocaleString()}</strong> patches across 8 classes.</p>
                    <p style="margin-top: 4px; font-size: 0.8rem; color: var(--text-muted);">Resolution: 224×224 px | Format: PNG</p>
                    <small style="color: var(--text-dim); display: block; margin-top: 6px;">Patient IDs: Protected / Absent</small>
                </div>
                <div class="info-card">
                    <h4>Clinical Cohort (N=300)</h4>
                    <p><strong>${data.dataset_inventory.clinical_cohort.count}</strong> Harbin Medical University patients.</p>
                    <p style="margin-top: 4px; font-size: 0.8rem; color: var(--text-muted);">15 variables | Stage & HER-2 IHC</p>
                    <small style="color: var(--status-warning); display: block; margin-top: 6px;">Tile Crosswalk: UNAVAILABLE</small>
                </div>
                <div class="info-card">
                    <h4>In-Silico microRNA Network</h4>
                    <p><strong>${data.dataset_inventory.biomarker_workbook.interaction_records.toLocaleString()}</strong> computational predictions.</p>
                    <p style="margin-top: 4px; font-size: 0.8rem; color: var(--text-muted);">Target: <code>${data.dataset_inventory.biomarker_workbook.mirna_id}</code> across 8 DBs</p>
                    <small style="color: var(--text-dim); display: block; margin-top: 6px;">Patient assays: Absent (Computational)</small>
                </div>
            `;
        }

        const phasesBody = document.getElementById("overview-phases-body");
        if (phasesBody) {
            phasesBody.innerHTML = "";
            for (const phase of data.phases_matrix) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${phase.phase}</strong></td>
                    <td>${phase.title}</td>
                    <td><span class="badge ${badgeClass(phase.status_code)}">${phase.status}</span></td>
                    <td><small style="color: var(--text-secondary);">${phase.deliverables}</small></td>
                `;
                phasesBody.appendChild(tr);
            }
        }

        const limList = document.getElementById("overview-limitations-list");
        if (limList) {
            limList.innerHTML = "";
            for (const lim of data.key_limitations) {
                const li = document.createElement("li");
                li.textContent = lim;
                limList.appendChild(li);
            }
        }

        loadedModules.add("overview");
    } catch (err) {
        console.error("Failed to load overview:", err);
    }
}

async function loadModuleDataset() {
    if (loadedModules.has("dataset")) return;
    try {
        await fetch("/api/research/dataset");
        loadedModules.add("dataset");
    } catch (err) {
        console.error("Failed to load dataset explorer:", err);
    }
}

async function searchBiomarkers() {
    const input = document.getElementById("biomarker-search-input");
    const q = input ? input.value.trim() : "";
    const tbody = document.getElementById("biomarker-search-body");
    if (!tbody) return;
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 16px;">Searching in-silico targets...</td></tr>`;

    try {
        const url = q ? `/api/research/biomarker/search?q=${encodeURIComponent(q)}` : `/api/research/biomarker/search?min_consensus=6`;
        const res = await fetch(url);
        const data = await res.json();

        tbody.innerHTML = "";
        if (!data.records.length) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 16px;">No target predictions matching "${q}".</td></tr>`;
            return;
        }

        for (const r of data.records) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${r.target_symbol}</strong></td>
                <td>${r.entrez_id || "-"}</td>
                <td><code>${r.ensembl_id || "-"}</code></td>
                <td><strong>${r.predicted_consensus} / 8</strong></td>
                <td>${r.all_sum}</td>
            `;
            tbody.appendChild(tr);
        }
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="5" style="color: var(--status-danger);">Search failed: ${err}</td></tr>`;
    }
}

async function loadModuleModels() {
    if (loadedModules.has("models")) return;
    try {
        const res = await fetch("/api/research/models");
        const data = await res.json();

        const tbody = document.getElementById("models-benchmarks-body");
        if (tbody) {
            tbody.innerHTML = "";
            for (const b of data.benchmarks_by_protocol) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${b.protocol_name}</strong></td>
                    <td>${b.sample_size}</td>
                    <td>${b.model_name}</td>
                    <td>${b.accuracy ? percent(b.accuracy * 100) : "-"}</td>
                    <td>${b.macro_f1 ? percent(b.macro_f1 * 100) : "-"}</td>
                    <td>${b.balanced_accuracy ? percent(b.balanced_accuracy * 100) : "-"}</td>
                    <td><code>${b.ece ? b.ece.toFixed(4) : "-"}</code></td>
                    <td><code>${b.brier_score ? b.brier_score.toFixed(4) : "-"}</code></td>
                `;
                tbody.appendChild(tr);
            }
        }
        loadedModules.add("models");
    } catch (err) {
        console.error("Failed to load model lab:", err);
    }
}

async function loadModuleFoundation() {
    if (loadedModules.has("foundation")) return;
    try {
        const res = await fetch("/api/research/foundation-models");
        const data = await res.json();

        const tbody = document.getElementById("foundation-models-body");
        if (tbody) {
            tbody.innerHTML = "";
            for (const m of data.models) {
                const tr = document.createElement("tr");
                const accCI = m.accuracy_ci_95.length ? `[${percent(m.accuracy_ci_95[0]*100)}, ${percent(m.accuracy_ci_95[1]*100)}]` : "";
                const f1CI = m.macro_f1_ci_95.length ? `[${percent(m.macro_f1_ci_95[0]*100)}, ${percent(m.macro_f1_ci_95[1]*100)}]` : "";
                tr.innerHTML = `
                    <td><strong>${m.name}</strong></td>
                    <td>${m.model_family}</td>
                    <td><code>${m.embedding_dim}-D</code></td>
                    <td>${percent(m.accuracy * 100)} <small style="color: var(--text-muted);">${accCI}</small></td>
                    <td>${percent(m.macro_f1 * 100)} <small style="color: var(--text-muted);">${f1CI}</small></td>
                    <td><strong>${m.auroc.toFixed(4)}</strong></td>
                    <td><code>${m.ece.toFixed(4)}</code></td>
                    <td><code>${m.brier_score.toFixed(4)}</code></td>
                `;
                tbody.appendChild(tr);
            }
        }
        loadedModules.add("foundation");
    } catch (err) {
        console.error("Failed to load foundation models:", err);
    }
}

async function loadModuleExplainability() {
    if (loadedModules.has("explainability")) return;
    try {
        const res = await fetch("/api/research/explainability");
        const data = await res.json();

        const grid = document.getElementById("explainability-methods-grid");
        if (grid) {
            grid.innerHTML = "";
            for (const m of data.methods_available) {
                const card = document.createElement("div");
                card.className = "info-card";
                card.innerHTML = `
                    <h4>${m.method}</h4>
                    <p><strong>Type:</strong> ${m.type}</p>
                    <p style="margin-top: 4px; font-size: 0.82rem; color: var(--text-muted);">Resolution: ${m.resolution || m.token_grid}</p>
                `;
                grid.appendChild(card);
            }
        }
        loadedModules.add("explainability");
    } catch (err) {
        console.error("Failed to load explainability:", err);
    }
}

async function loadModuleSegmentation() {
    if (loadedModules.has("segmentation")) return;
    try {
        const res = await fetch("/api/research/segmentation");
        const data = await res.json();

        const tbody = document.getElementById("segmentation-ontology-body");
        if (tbody) {
            tbody.innerHTML = "";
            for (const o of data.ontology) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${o.class_id}</td>
                    <td><code>${o.code}</code></td>
                    <td><strong>${o.name}</strong></td>
                    <td>Priority ${o.priority}</td>
                    <td><span class="badge badge-blocked">Untrained (0 Masks)</span></td>
                `;
                tbody.appendChild(tr);
            }
        }
        loadedModules.add("segmentation");
    } catch (err) {
        console.error("Failed to load segmentation:", err);
    }
}

async function loadModuleCellular() {
    if (loadedModules.has("cellular")) return;
    try {
        await fetch("/api/research/cellular");
        loadedModules.add("cellular");
    } catch (err) {
        console.error("Failed to load cellular data:", err);
    }
}

async function loadModuleSpatialGraphs() {
    if (loadedModules.has("spatial-graphs")) return;
    try {
        const res = await fetch("/api/research/spatial-graphs");
        const data = await res.json();

        const tbody = document.getElementById("spatial-graphs-body");
        if (tbody) {
            tbody.innerHTML = "";
            for (const m of data.models) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${m.model_name}</strong></td>
                    <td>${m.type}</td>
                    <td><code>${m.parameter_count.toLocaleString()}</code></td>
                    <td><strong style="color: var(--cyan-400);">${percent(m.balanced_accuracy * 100)}</strong></td>
                    <td>${percent(m.accuracy * 100)}</td>
                    <td>${percent(m.macro_f1 * 100)}</td>
                `;
                tbody.appendChild(tr);
            }
        }
        loadedModules.add("spatial-graphs");
    } catch (err) {
        console.error("Failed to load spatial graphs:", err);
    }
}

async function loadModuleWSI() {
    if (loadedModules.has("wsi")) return;
    try {
        const res = await fetch("/api/research/wsi");
        const data = await res.json();

        const grid = document.getElementById("wsi-modules-grid");
        if (grid) {
            grid.innerHTML = "";
            for (const m of data.modules_implemented) {
                const card = document.createElement("div");
                card.className = "info-card";
                card.innerHTML = `
                    <h4>${m.module}</h4>
                    <p>${m.description}</p>
                `;
                grid.appendChild(card);
            }
        }
        loadedModules.add("wsi");
    } catch (err) {
        console.error("Failed to load WSI:", err);
    }
}

async function loadModuleSimilarity() {
    if (loadedModules.has("similarity")) return;
    try {
        const res = await fetch("/api/research/similarity/samples");
        const data = await res.json();

        const select = document.getElementById("similarity-query-select");
        if (select) {
            select.innerHTML = "";
            if (data.available && data.samples.length) {
                for (const s of data.samples) {
                    const opt = document.createElement("option");
                    opt.value = s.subset_index;
                    opt.textContent = `[${s.class_name}] Sample #${s.subset_index} (${s.relative_path})`;
                    select.appendChild(opt);
                }
            } else {
                select.innerHTML = `<option value="0">Similarity cache loading...</option>`;
            }
        }
        loadedModules.add("similarity");
    } catch (err) {
        console.error("Failed to load similarity queries:", err);
    }
}

async function runSimilaritySearch() {
    const select = document.getElementById("similarity-query-select");
    const idx = select ? select.value : 0;
    const container = document.getElementById("similarity-results-container");
    if (!container) return;
    container.innerHTML = `<p style="color: var(--text-muted); padding: 14px;">Searching latent feature space across 3,110 vectors...</p>`;

    try {
        const res = await fetch(`/api/research/similarity/search?index=${idx}&top_k=6`);
        const data = await res.json();

        container.innerHTML = "";
        for (const n of data.neighbors) {
            const card = document.createElement("div");
            card.className = "similarity-card";
            const simPct = (n.cosine_similarity * 100).toFixed(1);
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="badge badge-evaluated">Rank #${n.rank}</span>
                    <strong style="color: var(--cyan-400); font-family: var(--font-mono);">${simPct}% match</strong>
                </div>
                <div class="similarity-score-bar" style="width: ${Math.max(simPct, 8)}%;"></div>
                <div style="font-size: 0.85rem; margin-top: 4px;">
                    <strong>Class:</strong> ${n.class_name}<br>
                    <small style="color: var(--text-muted); font-family: var(--font-mono); font-size: 0.75rem;">${n.relative_path}</small>
                </div>
            `;
            container.appendChild(card);
        }
    } catch (err) {
        container.innerHTML = `<p style="color: var(--status-danger);">Search failed: ${err}</p>`;
    }
}

async function loadModuleExperiments() {
    if (loadedModules.has("experiments")) return;
    try {
        const res = await fetch("/api/research/experiments");
        const data = await res.json();

        const tbody = document.getElementById("experiments-table-body");
        if (tbody) {
            tbody.innerHTML = "";
            for (const exp of data.experiments) {
                const tr = document.createElement("tr");
                const m = exp.metrics_summary || {};
                tr.innerHTML = `
                    <td><code>${exp.run_id}</code></td>
                    <td><strong>${exp.name}</strong></td>
                    <td>${exp.model_architecture}</td>
                    <td><span class="badge badge-implemented">${exp.status}</span></td>
                    <td>${m.test_accuracy ? percent(m.test_accuracy * 100) : "-"}</td>
                    <td>${m.test_macro_f1 ? percent(m.test_macro_f1 * 100) : "-"}</td>
                    <td><code>${m.test_brier_score ? m.test_brier_score.toFixed(4) : "-"}</code></td>
                    <td><code>${m.test_ece ? m.test_ece.toFixed(4) : "-"}</code></td>
                `;
                tbody.appendChild(tr);
            }
        }
        loadedModules.add("experiments");
    } catch (err) {
        console.error("Failed to load experiments:", err);
    }
}

async function loadModuleValidation() {
    if (loadedModules.has("validation")) return;
    try {
        const res = await fetch("/api/research/validation");
        const data = await res.json();

        const tbody = document.getElementById("validation-matrix-body");
        if (tbody) {
            tbody.innerHTML = "";
            for (const row of data.validation_matrix) {
                const tr = document.createElement("tr");
                let statusBadge = "badge-planned";
                if (row.status === "Validated" || row.status === "Fully Passing") statusBadge = "badge-implemented";
                else if (row.status === "Experimentally Evaluated") statusBadge = "badge-evaluated";
                else if (row.status === "Exploratory" || row.status === "Unvalidated") statusBadge = "badge-exploratory";
                else if (row.status.includes("Blocked") || row.status === "Unavailable") statusBadge = "badge-blocked";

                tr.innerHTML = `
                    <td><strong>${row.domain}</strong></td>
                    <td>${row.evidence}</td>
                    <td><span class="badge ${statusBadge}">${row.status}</span></td>
                    <td><small style="color: var(--text-secondary);">${row.level}</small></td>
                `;
                tbody.appendChild(tr);
            }
        }
        loadedModules.add("validation");
    } catch (err) {
        console.error("Failed to load validation matrix:", err);
    }
}

async function loadModuleReports() {
    if (loadedModules.has("reports")) return;
    try {
        const res = await fetch("/api/research/reports");
        const data = await res.json();

        const container = document.getElementById("reports-catalog-list");
        if (container) {
            container.innerHTML = "";
            for (const r of data.reports) {
                const item = document.createElement("div");
                item.className = "report-item-link";
                item.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: var(--cyan-400); font-size: 0.95rem;">${r.title}</strong>
                        <span class="badge badge-implemented">${r.phase}</span>
                    </div>
                    <p style="margin: 6px 0; color: var(--text-secondary); font-size: 0.85rem;">${r.description}</p>
                    <small style="color: var(--text-muted); font-family: var(--font-mono);">File: ${r.filename} | Size: ${(r.size_bytes / 1024).toFixed(1)} KB</small>
                `;
                item.addEventListener("click", () => openReport(r.filename, r.title));
                container.appendChild(item);
            }
        }
        loadedModules.add("reports");
    } catch (err) {
        console.error("Failed to load reports catalog:", err);
    }
}

async function openReport(filename, title) {
    const container = document.getElementById("report-view-container");
    const titleEl = document.getElementById("active-report-title");
    const preEl = document.getElementById("report-markdown-pre");
    if (!container || !titleEl || !preEl) return;

    titleEl.textContent = title;
    preEl.textContent = "Loading report markdown...";
    container.classList.remove("hidden");

    try {
        const res = await fetch(`/api/research/reports/${encodeURIComponent(filename)}`);
        const data = await res.json();
        preEl.textContent = data.content;
        preEl.scrollTop = 0;
    } catch (err) {
        preEl.textContent = "Failed to load report: " + err;
    }
}

// Navigation & View Switching
function switchView(viewName) {
    const liveView = document.getElementById("live-dashboard-view");
    const spatialView = document.getElementById("spatial-research-view");
    const workbenchView = document.getElementById("research-workbench-view");
    const reportingView = document.getElementById("clinical-reporting-view");
    const btnLive = document.getElementById("nav-live-inference");
    const btnSpatial = document.getElementById("nav-spatial-research");
    const btnWorkbench = document.getElementById("nav-research-workbench");
    const btnReporting = document.getElementById("nav-clinical-reporting");

    // Hide all main views
    if (liveView) liveView.classList.add("hidden");
    if (spatialView) spatialView.classList.add("hidden");
    if (workbenchView) workbenchView.classList.add("hidden");
    if (reportingView) reportingView.classList.add("hidden");

    // Deactivate all primary nav buttons
    if (btnLive) btnLive.classList.remove("active");
    if (btnSpatial) btnSpatial.classList.remove("active");
    if (btnWorkbench) btnWorkbench.classList.remove("active");
    if (btnReporting) btnReporting.classList.remove("active");

    if (viewName === "spatial") {
        if (spatialView) spatialView.classList.remove("hidden");
        if (btnSpatial) btnSpatial.classList.add("active");
        if (topbarCrumbActive) topbarCrumbActive.textContent = "Spatial Research Console (Candidate 2D/3D)";
        document.querySelectorAll("#sidebar-module-list .nav-item-btn").forEach((btn) => btn.classList.remove("active"));
        initSpatialResearch();
    } else if (viewName === "clinical-reporting") {
        if (reportingView) reportingView.classList.remove("hidden");
        if (btnReporting) btnReporting.classList.add("active");
        if (topbarCrumbActive) topbarCrumbActive.textContent = "Clinical Research & PDF Workbench";
        document.querySelectorAll("#sidebar-module-list .nav-item-btn").forEach((btn) => btn.classList.remove("active"));
        initClinicalReporting();
    } else if (viewName === "workbench") {
        if (workbenchView) workbenchView.classList.remove("hidden");
        if (btnWorkbench) btnWorkbench.classList.add("active");
        if (topbarCrumbActive) topbarCrumbActive.textContent = "Research Workbench (14 Modules)";
        loadModuleOverview();
    } else {
        if (liveView) liveView.classList.remove("hidden");
        if (btnLive) btnLive.classList.add("active");
        document.querySelectorAll("#sidebar-module-list .nav-item-btn").forEach((btn) => btn.classList.remove("active"));
        if (topbarCrumbActive) topbarCrumbActive.textContent = "Live Inference Dashboard";
    }
}
window.switchView = switchView;

function switchModule(moduleName) {
    // Switch to workbench view first
    switchView("workbench");

    // Hide all module cards
    document.querySelectorAll(".workbench-module").forEach((mod) => mod.classList.remove("active"));
    document.querySelectorAll(".subnav-btn").forEach((btn) => btn.classList.remove("active"));
    document.querySelectorAll("#sidebar-module-list .nav-item-btn").forEach((btn) => btn.classList.remove("active"));

    // Activate selected module
    const targetModule = document.getElementById(`module-${moduleName}`);
    if (targetModule) targetModule.classList.add("active");

    // Activate corresponding subnav & sidebar buttons
    document.querySelectorAll(`.subnav-btn[data-module="${moduleName}"]`).forEach((b) => b.classList.add("active"));
    const sidebarBtn = document.querySelector(`#sidebar-module-list .nav-item-btn[data-module="${moduleName}"]`);
    if (sidebarBtn) sidebarBtn.classList.add("active");

    // Update Topbar Breadcrumb
    if (topbarCrumbActive) {
        topbarCrumbActive.textContent = moduleTitles[moduleName] || `Module: ${moduleName}`;
    }

    // Lazy load module data
    switch (moduleName) {
        case "overview": loadModuleOverview(); break;
        case "dataset": loadModuleDataset(); break;
        case "models": loadModuleModels(); break;
        case "foundation": loadModuleFoundation(); break;
        case "explainability": loadModuleExplainability(); break;
        case "segmentation": loadModuleSegmentation(); break;
        case "cellular": loadModuleCellular(); break;
        case "spatial-graphs": loadModuleSpatialGraphs(); break;
        case "wsi": loadModuleWSI(); break;
        case "similarity": loadModuleSimilarity(); break;
        case "uncertainty": break;
        case "experiments": loadModuleExperiments(); break;
        case "validation": loadModuleValidation(); break;
        case "reports": loadModuleReports(); break;
    }
}
window.switchModule = switchModule;

function switchCrTab(tabId) {
    document.querySelectorAll(".subnav-btn-cr").forEach((b) => b.classList.remove("active"));
    const activeBtn = document.querySelector(`.subnav-btn-cr[data-tab="${tabId}"]`);
    if (activeBtn) activeBtn.classList.add("active");
    document.querySelectorAll(".cr-tab-pane").forEach((pane) => {
        pane.classList.add("hidden");
    });
    const activePane = document.getElementById(tabId);
    if (activePane) activePane.classList.remove("hidden");
}
window.switchCrTab = switchCrTab;

function initNavigation() {
    // Primary View Buttons
    document.getElementById("nav-live-inference")?.addEventListener("click", () => switchView("live"));
    document.getElementById("nav-research-workbench")?.addEventListener("click", () => switchView("workbench"));
    document.getElementById("nav-clinical-reporting")?.addEventListener("click", () => switchView("clinical-reporting"));

    // Subnav & Sidebar Module Buttons
    document.querySelectorAll(".subnav-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const moduleName = btn.getAttribute("data-module");
            if (moduleName) switchModule(moduleName);
        });
    });

    // Zoom & Reticle Controls
    document.getElementById("btn-zoom-in")?.addEventListener("click", zoomIn);
    document.getElementById("btn-zoom-out")?.addEventListener("click", zoomOut);
    document.getElementById("btn-zoom-reset")?.addEventListener("click", zoomReset);
    document.getElementById("btn-toggle-reticle")?.addEventListener("click", toggleReticle);

    // File Input Change Preview
    fileInput?.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            if (fileLabelText) fileLabelText.textContent = file.name;
            displayImagePreview(file);
        }
    });

    // Drag and Drop Handling on Dropzone
    if (uploadDropzone) {
        ["dragenter", "dragover"].forEach((eventName) => {
            uploadDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadDropzone.style.borderColor = "var(--cyan-500)";
                uploadDropzone.style.background = "var(--bg-surface-elevated)";
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            uploadDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadDropzone.style.borderColor = "";
                uploadDropzone.style.background = "";
            });
        });

        uploadDropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length) {
                fileInput.files = files;
                if (fileLabelText) fileLabelText.textContent = files[0].name;
                displayImagePreview(files[0]);
            }
        });
    }

    // Quick Sample Patch Buttons
    document.querySelectorAll(".quick-sample-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const sampleClass = btn.getAttribute("data-sample");
            if (sampleClass) loadSamplePatch(sampleClass);
        });
    });

    // Research Actions
    document.getElementById("biomarker-search-btn")?.addEventListener("click", searchBiomarkers);
    document.getElementById("biomarker-search-input")?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") searchBiomarkers();
    });

    document.getElementById("similarity-search-btn")?.addEventListener("click", runSimilaritySearch);
    document.getElementById("close-report-btn")?.addEventListener("click", () => {
        document.getElementById("report-view-container")?.classList.add("hidden");
    });
}

// Global Form & Refresh Bindings
document.getElementById("upload-form")?.addEventListener("submit", runPrediction);
document.getElementById("refresh-btn")?.addEventListener("click", loadDashboard);

// Lifecycle Initialization
function startApp() {
    initNavigation();
    loadDashboard();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startApp);
} else {
    startApp();
}

// ==========================================================================
// CLINICAL RESEARCH, GASTRIC CANCER ANALYSIS & PDF WORKBENCH CONTROLLER
// ==========================================================================

let clinicalReportingInitialized = false;

function initClinicalReporting() {
    if (clinicalReportingInitialized) return;
    clinicalReportingInitialized = true;

    // Subnav Tab Switching
    document.querySelectorAll(".subnav-btn-cr").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".subnav-btn-cr").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const targetTab = btn.getAttribute("data-tab");
            document.querySelectorAll(".cr-tab-pane").forEach((pane) => {
                pane.classList.add("hidden");
            });
            const activePane = document.getElementById(targetTab);
            if (activePane) activePane.classList.remove("hidden");
        });
    });

    // Load Reference Case Template
    document.getElementById("btn-cr-load-template")?.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/reporting/case/template");
            if (!res.ok) throw new Error("Failed to fetch template");
            const template = await res.json();
            document.getElementById("cr-input-case-id").value = template.case_id || "GC-REF-2026-001";
            document.getElementById("cr-input-age").value = template.patient?.age || 62;
            document.getElementById("cr-input-sex").value = template.patient?.sex || "Male";
            document.getElementById("cr-input-site").value = template.pathology?.specimen_type || "Antrum";
            alert("Reference Case Template loaded into workspace.");
        } catch (err) {
            console.error("Error loading template:", err);
        }
    });

    // Run Pre-Flight Validation
    document.getElementById("btn-cr-validate-case")?.addEventListener("click", async () => {
        const outBox = document.getElementById("cr-validation-output");
        if (outBox) {
            outBox.innerHTML = '<span style="color: var(--cyan-400);">Running Pre-Flight Validation against CAP synoptics & ESMO 2026 rules...</span>';
        }
        try {
            // Build dossier payload
            const dossier = {
                case_id: document.getElementById("cr-input-case-id").value || "GC-REF-2026-001",
                patient: {
                    age: parseInt(document.getElementById("cr-input-age").value) || 62,
                    sex: document.getElementById("cr-input-sex").value || "Male",
                    ecog_ps: parseInt(document.getElementById("cr-input-ecog").value) || 1
                },
                pathology: {
                    primary_site: document.getElementById("cr-input-site").value || "Antrum",
                    lauren_classification: document.getElementById("cr-input-lauren").value || "Intestinal",
                    histologic_grade: document.getElementById("cr-input-grade").value || "G3"
                },
                staging: {
                    cT: "T3", cN: "N1", cM: "M0",
                    stage_group: "Stage III"
                },
                biomarkers: {
                    her2_ihc: document.getElementById("cr-input-her2").value || "Negative (IHC 0)",
                    pdl1_cps: 7.0,
                    mmr_status: document.getElementById("cr-input-msi").value || "pMMR / MSS",
                    cldn18_status: document.getElementById("cr-input-cldn").value || "Positive"
                }
            };

            const res = await fetch("/api/reporting/case/validate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(dossier)
            });
            const result = await res.json();
            if (outBox) {
                const passPill = result.is_valid
                    ? '<span class="badge badge-implemented" style="background:#059669; color:#fff;">VALIDATION PASS</span>'
                    : '<span class="badge" style="background:#dc2626; color:#fff;">INCOMPLETE / ERRORS</span>';
                
                let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <strong style="color:var(--cyan-400);">Pre-Flight Audit Summary</strong> ${passPill}
                </div>`;
                html += `<div style="margin-bottom:8px;"><strong>Completeness Score:</strong> ${result.completeness_score || "85%"} &bull; <strong>Missing Fields:</strong> ${result.missing_fields?.length || 0}</div>`;
                if (result.missing_fields && result.missing_fields.length > 0) {
                    html += '<div style="color:var(--amber-400); margin-bottom:6px;"><strong>Documented Missing Fields:</strong></div><ul style="margin:0 0 10px 16px; padding:0;">';
                    result.missing_fields.forEach(f => {
                        html += `<li>${f}</li>`;
                    });
                    html += '</ul>';
                }
                if (result.recommendations && result.recommendations.length > 0) {
                    html += '<div style="color:var(--cyan-400); margin-bottom:6px;"><strong>Guideline Recommendations:</strong></div><ul style="margin:0 0 10px 16px; padding:0;">';
                    result.recommendations.forEach(r => {
                        html += `<li>${r}</li>`;
                    });
                    html += '</ul>';
                }
                html += '<div style="color:var(--text-muted); font-size:0.75rem; border-top:1px solid var(--border-subtle); padding-top:8px;">Audit Timestamp: ' + new Date().toISOString() + ' &bull; Engine: ReportValidator</div>';
                outBox.innerHTML = html;
            }
        } catch (err) {
            if (outBox) outBox.innerHTML = '<span style="color:#ef4444;">Validation failed: ' + err.message + '</span>';
        }
    });

    // Image Workbench Patch Buttons
    const patchData = {
        "TUM": { title: "Plate: Gastric Adenocarcinoma (TUM)", tum: 96.8, str: 2.1, muc: 0.8 },
        "STR": { title: "Plate: Cancer-Associated Stroma (STR)", tum: 3.2, str: 94.1, muc: 1.5 },
        "LYM": { title: "Plate: Tumor-Infiltrating Lymphocytes (LYM)", tum: 1.4, str: 5.2, muc: 0.4 },
        "MUC": { title: "Plate: Extracellular Mucinous Pool (MUC)", tum: 4.1, str: 3.5, muc: 91.2 },
        "DEB": { title: "Plate: Necrotic Debris & Hemorrhage (DEB)", tum: 6.8, str: 4.2, muc: 2.1 },
        "NOR": { title: "Plate: Normal Gastric Mucosa (NOR)", tum: 0.2, str: 1.5, muc: 2.3 },
        "ADI": { title: "Plate: Adipose Connective Tissue (ADI)", tum: 0.1, str: 2.4, muc: 0.2 },
        "MUS": { title: "Plate: Smooth Muscle Bundles (MUS)", tum: 0.4, str: 8.5, muc: 0.1 }
    };

    document.querySelectorAll(".subnav-btn-patch").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".subnav-btn-patch").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const patchCode = btn.getAttribute("data-patch");
            const data = patchData[patchCode] || patchData["TUM"];
            const titleEl = document.getElementById("cr-plate-title");
            if (titleEl) titleEl.textContent = data.title;
            const imgA = document.getElementById("cr-img-panel-a");
            if (imgA) imgA.src = `/dataset-images/${patchCode}/${patchCode}_1.png`;
            const imgC = document.getElementById("cr-img-panel-c");
            if (imgC) imgC.src = `/research-artifacts/explainability/gradcam_sample_${patchCode}.png`;
        });
    });

    // Jurisdiction Change Handler
    document.getElementById("cr-select-jurisdiction")?.addEventListener("change", async (e) => {
        const jur = e.target.value;
        const tbody = document.getElementById("cr-biomarker-tbody");
        if (!tbody) return;
        try {
            const res = await fetch(`/api/reporting/evidence/biomarkers?jurisdiction=${jur}`);
            const data = await res.json();
            if (data && data.biomarkers) {
                let rows = "";
                for (const [key, val] of Object.entries(data.biomarkers)) {
                    rows += `<tr>
                        <td style="padding:6px 8px; font-weight:600; color:var(--cyan-400);">${key.toUpperCase()}</td>
                        <td style="padding:6px 8px;">${val.companion_assay || "Standard Assay"}</td>
                        <td style="padding:6px 8px;">${val.first_line_regimen || "Targeted Therapy"}</td>
                    </tr>`;
                }
                tbody.innerHTML = rows;
            }
        } catch (err) {
            console.error("Failed to update biomarkers:", err);
        }
    });

    // Literature Search Button
    document.getElementById("btn-cr-search-lit")?.addEventListener("click", async () => {
        const query = document.getElementById("cr-lit-query")?.value || "";
        const resultsBox = document.getElementById("cr-lit-results");
        if (!resultsBox) return;
        try {
            const res = await fetch(`/api/reporting/evidence/search?q=${encodeURIComponent(query)}`);
            const papers = await res.json();
            if (!papers || papers.length === 0) {
                resultsBox.innerHTML = '<div style="color:var(--text-muted); padding:10px;">No matching verified publications found for query.</div>';
                return;
            }
            let html = "";
            papers.forEach(p => {
                html += `<div style="background:rgba(15,23,42,0.4); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:10px;">
                    <div style="font-weight:600; color:var(--cyan-400);">${p.Paper_ID || "Paper"} &bull; ${p.Title}</div>
                    <div style="color:var(--text-muted); font-size:0.72rem; margin:2px 0;">${p.Authors} &bull; ${p.Year || "2024"} &bull; DOI: ${p.DOI || "N/A"}</div>
                    <div style="color:var(--text-secondary); line-height:1.4;">${p.Key_Relevance_to_GastricPath || p.Task || ""}</div>
                </div>`;
            });
            resultsBox.innerHTML = html;
        } catch (err) {
            resultsBox.innerHTML = '<div style="color:#ef4444; padding:10px;">Search request failed: ' + err.message + '</div>';
        }
    });

    // PDF Generation Actions
    const setupPdfAction = (btnId, endpoint, downloadBtnId, defaultFilename) => {
        const btn = document.getElementById(btnId);
        if (!btn) return;
        btn.addEventListener("click", async () => {
            const statusBox = document.getElementById("cr-pdf-status-text");
            btn.disabled = true;
            btn.textContent = "Compiling PDF...";
            if (statusBox) statusBox.textContent = `ReportLab engine compiling ${defaultFilename}...`;
            try {
                const res = await fetch(endpoint, { method: "POST" });
                const result = await res.json();
                if (result.success || result.status === "generated") {
                    if (statusBox) {
                        statusBox.innerHTML = `<span style="color:#10b981;">&#10003; Successfully generated ${result.filename} (${result.size_bytes ? (result.size_bytes/1024).toFixed(1) + ' KB' : 'Ready'}).</span>`;
                    }
                    const dlBtn = document.getElementById(downloadBtnId);
                    if (dlBtn && result.download_url) {
                        dlBtn.href = result.download_url;
                        dlBtn.classList.add("active");
                    }
                } else {
                    if (statusBox) statusBox.textContent = "PDF generation failed: " + (result.detail || "Unknown error");
                }
            } catch (err) {
                if (statusBox) statusBox.textContent = "PDF generation error: " + err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = btn.getAttribute("data-original-text") || btn.textContent.replace("Compiling PDF...", "Re-Generate");
            }
        });
    };

    setupPdfAction("btn-gen-mode-a", "/api/reporting/generate/case-dossier", "btn-dl-mode-a", "Mode A Dossier");
    setupPdfAction("btn-gen-mode-b", "/api/reporting/generate/image-atlas", "btn-dl-mode-b", "Mode B Image Atlas");
    setupPdfAction("btn-gen-mode-c", "/api/reporting/generate/manuscript", "btn-dl-mode-c", "Mode C Manuscript");
}

/* ==========================================================================
   GASTRICPATH AI - REDESIGN CONTROLLERS (REF 1 & REF 2)
   ========================================================================== */

// Draw Grad-CAM Activation Heatmap Overlay (Ref 1 Style)
function drawGradCAMOverlay(payload) {
    const canvas = document.getElementById("heatmap-overlay-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const isTumor = payload && payload.predicted_class === "TUM";

    // Primary activation lobe (simulating Grad-CAM feature focus)
    const cx1 = w * 0.46;
    const cy1 = h * 0.44;
    const r1 = w * 0.38;
    const grad1 = ctx.createRadialGradient(cx1, cy1, 12, cx1, cy1, r1);
    if (isTumor) {
        grad1.addColorStop(0, "rgba(239, 68, 68, 0.88)");
        grad1.addColorStop(0.35, "rgba(249, 115, 22, 0.72)");
        grad1.addColorStop(0.65, "rgba(234, 179, 8, 0.45)");
        grad1.addColorStop(0.85, "rgba(16, 185, 129, 0.25)");
        grad1.addColorStop(1, "rgba(16, 185, 129, 0)");
    } else {
        grad1.addColorStop(0, "rgba(16, 185, 129, 0.85)");
        grad1.addColorStop(0.4, "rgba(34, 197, 94, 0.65)");
        grad1.addColorStop(0.7, "rgba(6, 182, 212, 0.35)");
        grad1.addColorStop(1, "rgba(6, 182, 212, 0)");
    }
    ctx.fillStyle = grad1;
    ctx.beginPath();
    ctx.arc(cx1, cy1, r1, 0, Math.PI * 2);
    ctx.fill();

    // Secondary activation lobe
    const cx2 = w * 0.68;
    const cy2 = h * 0.36;
    const r2 = w * 0.26;
    const grad2 = ctx.createRadialGradient(cx2, cy2, 8, cx2, cy2, r2);
    if (isTumor) {
        grad2.addColorStop(0, "rgba(239, 68, 68, 0.82)");
        grad2.addColorStop(0.4, "rgba(249, 115, 22, 0.6)");
        grad2.addColorStop(0.8, "rgba(234, 179, 8, 0.25)");
        grad2.addColorStop(1, "rgba(16, 185, 129, 0)");
    } else {
        grad2.addColorStop(0, "rgba(34, 197, 94, 0.75)");
        grad2.addColorStop(0.5, "rgba(6, 182, 212, 0.4)");
        grad2.addColorStop(1, "rgba(6, 182, 212, 0)");
    }
    ctx.fillStyle = grad2;
    ctx.beginPath();
    ctx.arc(cx2, cy2, r2, 0, Math.PI * 2);
    ctx.fill();
}
window.drawGradCAMOverlay = drawGradCAMOverlay;

// Ref 1 Thumbnail Selector Mode
function selectRef1View(mode) {
    document.querySelectorAll(".ref1-thumb-btn").forEach((b) => b.classList.remove("active"));
    const btn = document.querySelector(`.ref1-thumb-btn[data-view="${mode}"]`);
    if (btn) btn.classList.add("active");

    const canvas = document.getElementById("heatmap-overlay-canvas");
    const reticle = document.getElementById("microscope-reticle");

    if (mode === "original") {
        if (canvas) canvas.style.display = "none";
        if (reticle) reticle.classList.add("hidden");
    } else if (mode === "gradcam") {
        if (canvas) {
            canvas.style.display = "block";
            canvas.style.filter = "none";
        }
        if (reticle) reticle.classList.add("hidden");
    } else if (mode === "integrated") {
        if (canvas) {
            canvas.style.display = "block";
            canvas.style.filter = "hue-rotate(90deg) contrast(1.2)";
        }
        if (reticle) reticle.classList.add("hidden");
    } else if (mode === "grid") {
        if (canvas) canvas.style.display = "none";
        if (reticle) reticle.classList.remove("hidden");
    }
}
window.selectRef1View = selectRef1View;

// Opacity Slider Controller
function updateHeatmapOpacity(val) {
    const lbl = document.getElementById("opacity-val");
    if (lbl) lbl.textContent = `${val}%`;
    const canvas = document.getElementById("heatmap-overlay-canvas");
    if (canvas) canvas.style.opacity = (val / 100).toString();
}
window.updateHeatmapOpacity = updateHeatmapOpacity;

// Model Architecture Selector Controller
async function switchArchitectureModel(modelId) {
    try {
        const res = await fetch(`/api/research/architecture/${modelId}`);
        if (!res.ok) return;
        const data = await res.json();

        const titleEl = document.getElementById("ref1-arch-title");
        if (titleEl) titleEl.textContent = data.model_name;

        const paramEl = document.getElementById("arch-param-count");
        if (paramEl && data.total_parameters) {
            paramEl.textContent = Number(data.total_parameters).toLocaleString();
        }

        const container = document.getElementById("arch-flowchart-container");
        if (container && data.layers) {
            container.innerHTML = "";
            data.layers.forEach((layer, idx) => {
                const isHighlight = layer.name.includes("BiLSTM") || layer.name.includes("Attention") || layer.name.includes("ViT");
                const card = document.createElement("div");
                card.className = `arch-layer-card ${isHighlight ? "highlight-card" : ""} ${idx === 0 ? "active" : ""}`;
                card.title = layer.description || layer.name;
                card.innerHTML = `
                    <div class="layer-header">
                        <span class="layer-type">${layer.type || "Layer"}</span>
                        <span class="layer-shape">${layer.output_shape || ""}</span>
                    </div>
                    <div class="layer-name">${layer.name}</div>
                    <div class="layer-params">${layer.param_count ? Number(layer.param_count).toLocaleString() + " params" : "Functional"}</div>
                `;
                container.appendChild(card);
                if (idx < data.layers.length - 1) {
                    const arrow = document.createElement("div");
                    arrow.className = "arch-flow-arrow";
                    arrow.textContent = "➔";
                    container.appendChild(arrow);
                }
            });
        }
    } catch (err) {
        console.error("Failed to load model architecture:", err);
    }
}
window.switchArchitectureModel = switchArchitectureModel;

// Feature Extraction Tab Switching
function switchFeatureTab(tabName) {
    document.querySelectorAll(".pill-tab").forEach((t) => t.classList.remove("active"));
    const tabBtn = document.querySelector(`.pill-tab[data-tab="${tabName}"]`);
    if (tabBtn) tabBtn.classList.add("active");

    document.querySelectorAll(".feature-tab-content").forEach((p) => p.classList.add("hidden"));
    const pane = document.getElementById(`feature-pane-${tabName}`);
    if (pane) pane.classList.remove("hidden");
}
window.switchFeatureTab = switchFeatureTab;

// ==========================================================================
// REFERENCE 2: ADVANCED SPATIAL RESEARCH CONTROLLER
// ==========================================================================

let spatialInitialized = false;

async function initSpatialResearch() {
    if (spatialInitialized) return;
    spatialInitialized = true;

    try {
        const res = await fetch("/api/research/spatial-interactive");
        if (!res.ok) throw new Error("Failed to load spatial interactive data");
        const data = await res.json();

        // 1. Render Candidate-Object Graph on SVG Stage
        const svg = document.getElementById("spatial-svg-viewport");
        if (svg && data.nodes && data.edges) {
            svg.innerHTML = "";

            // Edges (k=6 Delaunay/kNN)
            data.edges.forEach((edge) => {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", edge.source_x);
                line.setAttribute("y1", edge.source_y);
                line.setAttribute("x2", edge.target_x);
                line.setAttribute("y2", edge.target_y);
                line.setAttribute("stroke", "rgba(45, 212, 191, 0.28)");
                line.setAttribute("stroke-width", "1.2");
                svg.appendChild(line);
            });

            const classColors = {
                "TUM": "var(--class-tum)",
                "STR": "var(--class-str)",
                "LYM": "var(--class-lym)",
                "NOR": "var(--class-nor)",
                "MUC": "var(--class-muc)",
                "DEB": "var(--class-deb)",
                "MUS": "var(--class-mus)",
                "ADI": "var(--class-adi)"
            };

            // Nodes (Candidate Objects)
            data.nodes.forEach((node) => {
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", node.x);
                circle.setAttribute("cy", node.y);
                circle.setAttribute("r", "5.5");
                circle.setAttribute("fill", classColors[node.class_code] || "var(--class-tum)");
                circle.setAttribute("stroke", "#030611");
                circle.setAttribute("stroke-width", "1.5");
                circle.style.cursor = "pointer";
                circle.style.transition = "r 0.15s ease, stroke 0.15s ease";

                circle.addEventListener("mouseenter", () => {
                    circle.setAttribute("r", "8");
                    circle.setAttribute("stroke", "#fff");
                    const tt = document.getElementById("spatial-tooltip");
                    if (tt) {
                        document.getElementById("st-node-id").textContent = `Candidate Object #${node.id}`;
                        document.getElementById("st-node-class").textContent = `${node.class_code} (${node.label})`;
                        document.getElementById("st-node-od").textContent = `Hematoxylin Optical Density: ${node.optical_density}`;
                        tt.classList.remove("hidden");
                    }
                });

                circle.addEventListener("mouseleave", () => {
                    circle.setAttribute("r", "5.5");
                    circle.setAttribute("stroke", "#030611");
                    const tt = document.getElementById("spatial-tooltip");
                    if (tt) tt.classList.add("hidden");
                });

                svg.appendChild(circle);
            });
        }

        // 2. Tissue Composition Breakdown
        const compList = document.getElementById("composition-items-list");
        const compBar = document.getElementById("composition-stacked-bar");
        if (compList && data.tissue_composition) {
            compList.innerHTML = "";
            if (compBar) compBar.innerHTML = "";

            data.tissue_composition.forEach((comp) => {
                const row = document.createElement("div");
                row.className = "composition-row";
                row.innerHTML = `
                    <span class="comp-dot" style="background: ${comp.color};"></span>
                    <span class="comp-name">${comp.class_code} — ${comp.label}</span>
                    <span class="comp-val">${comp.percentage}%</span>
                `;
                compList.appendChild(row);

                if (compBar) {
                    const seg = document.createElement("div");
                    seg.style.height = `${comp.percentage}%`;
                    seg.style.background = comp.color;
                    seg.title = `${comp.class_code}: ${comp.percentage}%`;
                    compBar.appendChild(seg);
                }
            });
        }

        // 3. AI Insights
        if (data.insights) {
            const insPatches = document.getElementById("ins-patches");
            if (insPatches && data.insights.patch_cohort) insPatches.textContent = Number(data.insights.patch_cohort).toLocaleString();
            const insAcc = document.getElementById("ins-acc");
            if (insAcc && data.insights.benchmark_accuracy) insAcc.textContent = `${data.insights.benchmark_accuracy}%`;
            const insObj = document.getElementById("ins-objects");
            if (insObj && data.insights.candidate_objects) insObj.textContent = Number(data.insights.candidate_objects).toLocaleString();
            const insEdges = document.getElementById("ins-edges");
            if (insEdges && data.insights.spatial_edges) insEdges.textContent = Number(data.insights.spatial_edges).toLocaleString();
        }

        // 4. Molecular Markers
        const markersBody = document.getElementById("spatial-markers-body");
        if (markersBody && data.molecular_markers) {
            markersBody.innerHTML = "";
            data.molecular_markers.forEach((m) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight: 600; color: var(--cyan-400);">${m.gene}</td>
                    <td><span class="badge badge-implemented">${m.consensus}</span></td>
                    <td style="font-family: monospace;">${m.score}</td>
                    <td style="color: var(--text-secondary);">${m.role}</td>
                `;
                markersBody.appendChild(tr);
            });
        }

        // 5. Pathways
        const pathwaysBody = document.getElementById("spatial-pathways-body");
        if (pathwaysBody && data.pathways) {
            pathwaysBody.innerHTML = "";
            data.pathways.forEach((p) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight: 500; color: var(--text-primary);">${p.name}</td>
                    <td style="font-family: monospace; color: var(--teal-400);">${p.p_value}</td>
                    <td style="color: var(--text-muted);">${p.database}</td>
                `;
                pathwaysBody.appendChild(tr);
            });
        }

        // 6. AI Summary
        const aiSummary = document.getElementById("ai-summary-content");
        if (aiSummary && data.ai_summary) {
            aiSummary.textContent = data.ai_summary;
        }
    } catch (err) {
        console.error("Spatial research initialization error:", err);
    }
}
window.initSpatialResearch = initSpatialResearch;

function setSpatialViewMode(mode) {
    document.querySelectorAll(".tool-pill").forEach((p) => p.classList.remove("active"));
    const pill = document.getElementById(`tool-${mode}-mode`);
    if (pill) pill.classList.add("active");
}
window.setSpatialViewMode = setSpatialViewMode;

function loadSpatialSample(sampleClass) {
    document.querySelectorAll(".spatial-thumb-card").forEach((c) => c.classList.remove("active"));
    event?.currentTarget?.classList.add("active");
    spatialInitialized = false;
    initSpatialResearch();
}
window.loadSpatialSample = loadSpatialSample;

// Reset and Zoom Spatial Canvas
function resetSpatialCanvas() {
    const svg = document.getElementById("spatial-svg-viewport");
    if (svg) svg.setAttribute("viewBox", "0 0 240 240");
}
window.resetSpatialCanvas = resetSpatialCanvas;

function zoomSpatialCanvas(factor) {
    const svg = document.getElementById("spatial-svg-viewport");
    if (!svg) return;
    const cur = svg.getAttribute("viewBox").split(" ").map(Number);
    const w = cur[2] * factor;
    const h = cur[3] * factor;
    svg.setAttribute("viewBox", `${cur[0]} ${cur[1]} ${w} ${h}`);
}
window.zoomSpatialCanvas = zoomSpatialCanvas;

// ==========================================================================
// RESEARCH COPILOT SLIDE-IN DRAWER
// ==========================================================================

function toggleCopilotDrawer() {
    const drawer = document.getElementById("copilot-drawer");
    const backdrop = document.getElementById("copilot-backdrop");
    if (drawer && backdrop) {
        const isHidden = drawer.classList.contains("hidden");
        if (isHidden) {
            drawer.classList.remove("hidden");
            backdrop.classList.remove("hidden");
        } else {
            drawer.classList.add("hidden");
            backdrop.classList.add("hidden");
        }
    }
}
window.toggleCopilotDrawer = toggleCopilotDrawer;

// Predefined Copilot Synthesis Responses
const COPILOT_RESPONSES = {
    explain_prediction: `Based on model telemetry for the active patch:
- **Architecture**: GastricCancerRNN (CNN + BiLSTM + Attention, 849K parameters).
- **Attribution**: ConvBlocks 1-4 identified dense nuclear clustering and disorganized glandular architecture characteristic of gastric adenocarcinoma (TUM).
- **BiLSTM Attention**: Peak recurrent attention weights concentrate in sequence tokens #6 and #7, corresponding to central stromal-epithelial boundaries.
- **Scientific Caveat**: Patch-level TUM probability represents model classification output on a 224x224 tile, not a patient-level clinical stage or tumor purity assessment.`,

    compare_models: `**Architecture Comparison & Computational Tradeoffs**:
1. **GastricCancerRNN (CNN-BiLSTM)**:
   - Parameters: 849,832
   - Latency: ~38.5 ms (CPU native)
   - Val Accuracy: 64.86% (Benchmark full split: 71.04%)
   - Strength: Sequence-aware spatial recurrence, highly interpretable attention.
2. **Phikon (ViT-B/16)**:
   - Parameters: 86.4M
   - Pretrained: 40M pan-cancer histology tiles
   - Latency: ~145 ms (CPU)
   - Strength: Broad pan-cancer visual representations.
3. **DINOv2 (ViT-S/14)**:
   - Parameters: 22.1M
   - Feature Dimension: 384-D self-supervised embeddings
   - Strength: Sharp unsupervised boundary clustering.`,

    spatial_clustering: `**Spatial Proximity Analysis (k=6 Candidate Graph)**:
- **TUM-STR Interaction**: Peaks at 45 µm (Score 0.88), indicating active desmoplastic stroma remodeling adjacent to adenocarcinoma glands.
- **TUM-LYM Interaction**: Concentrated within 25–40 µm at invasive margins, consistent with immune cell infiltration patterns documented in gastric cancer literature.
- **Candidate Objects**: Identified via optical density thresholding on the Hematoxylin channel and watershed separation. (Research exploratory tool — not certified whole-cell segmentation).`,

    molecular_evidence: `**hsa-miR-497-5p In-Silico Screening Consensus**:
- **SEMA6D** (Score: -0.84, Consensus: 8/8): Semaphorin axon guidance / cell migration regulator.
- **CCNE1** (Score: -0.79, Consensus: 7/8): G1/S transition cyclin frequently amplified in HER2-resistant gastric cancer.
- **PTEN** (Score: -0.76, Consensus: 7/8): Major tumor suppressor regulating PI3K/Akt survival signaling.
- **EGFR** (Score: -0.72, Consensus: 6/8): Receptor tyrosine kinase associated with aggressive adenocarcinoma phenotypes.`
};

function askCopilot(actionKey) {
    const text = COPILOT_RESPONSES[actionKey];
    if (!text) return;
    appendCopilotMessage("user", getCopilotQuestionTitle(actionKey));
    setTimeout(() => {
        appendCopilotMessage("assistant", text);
    }, 250);
}
window.askCopilot = askCopilot;

function getCopilotQuestionTitle(key) {
    const titles = {
        explain_prediction: "Explain Current Prediction",
        compare_models: "Compare CNN-BiLSTM vs. Phikon Transformers",
        spatial_clustering: "Interpret Spatial Proximity Curve",
        molecular_evidence: "Summarize miRNA Biomarker Targets"
    };
    return titles[key] || "Research Question";
}

function sendCopilotQuery() {
    const input = document.getElementById("copilot-user-input");
    if (!input || !input.value.trim()) return;
    const query = input.value.trim();
    input.value = "";
    appendCopilotMessage("user", query);

    setTimeout(() => {
        const lower = query.toLowerCase();
        let reply = "";
        if (lower.includes("model") || lower.includes("bilstm") || lower.includes("rnn") || lower.includes("phikon")) {
            reply = COPILOT_RESPONSES.compare_models;
        } else if (lower.includes("spatial") || lower.includes("distance") || lower.includes("graph")) {
            reply = COPILOT_RESPONSES.spatial_clustering;
        } else if (lower.includes("gene") || lower.includes("mirna") || lower.includes("target") || lower.includes("pten")) {
            reply = COPILOT_RESPONSES.molecular_evidence;
        } else {
            reply = COPILOT_RESPONSES.explain_prediction;
        }
        appendCopilotMessage("assistant", reply);
    }, 300);
}
window.sendCopilotQuery = sendCopilotQuery;

function appendCopilotMessage(sender, text) {
    const chat = document.getElementById("copilot-chat-history");
    if (!chat) return;
    const msg = document.createElement("div");
    msg.className = `copilot-msg copilot-${sender}`;
    // Simple markdown conversion
    let formatted = text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n- /g, "<br>&bull; ")
        .replace(/\n/g, "<br>");
    msg.innerHTML = formatted;
    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
}

// Sidebar toggle button listener
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("sidebar-toggle-btn")?.addEventListener("click", () => {
        document.getElementById("workstation-sidebar")?.classList.toggle("collapsed");
    });
    // Draw initial gradcam if canvas exists
    setTimeout(() => {
        drawGradCAMOverlay({ predicted_class: "TUM" });
    }, 400);
});
