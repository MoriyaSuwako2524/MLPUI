/**
 * app.js — MLPUI node-graph application.
 */

// Colour palette for connection wires by type
const TYPE_COLORS = {
    MLP_RESULT:      "#9c6fd6",
    ASE_ATOMS:       "#f0a500",
    ASE_ATOMS_LIST:  "#e07820",
    BATCH_RESULT:    "#c85cc8",
    DATASET_RESULT:  "#4ecb71",
    FLOAT:           "#b5cea8",
    STRING:          "#ce9178",
    INT:             "#b5cea8",
    BOOLEAN:         "#569cd6",
};

// Widget types that live inside the node (not connection slots)
const WIDGET_TYPES = new Set(["STRING", "INT", "FLOAT", "BOOLEAN"]);

class MLPUIApp {
    constructor() {
        this.graph           = new LGraph();
        this.canvas          = null;
        this.nodeTypes       = {};   // nodeId → object_info entry
        this.running         = false;
        this._dsCurrentName  = null;
        this._dsCurrentPage  = 0;
        this._dsPageSize     = 50;
        this._dsInfo         = null;
    }

    async setup() {
        // --- canvas ---
        const el = document.getElementById("graph-canvas");
        this.canvas = new LGraphCanvas(el, this.graph);
        this.canvas.background_image      = "";
        this.canvas.render_shadows        = false;
        this.canvas.render_canvas_border  = false;
        this.canvas.always_render_background = true;
        this.canvas.show_info             = false;

        // --- load node types ---
        const info = await api.getObjectInfo();
        this.nodeTypes = info;
        for (const [nodeId, nodeData] of Object.entries(info)) {
            this._registerNodeType(nodeId, nodeData);
        }

        // --- toolbar wiring ---
        document.getElementById("btn-queue").addEventListener("click",   () => this._queuePrompt());
        document.getElementById("btn-clear").addEventListener("click",   () => this._clearGraph());
        document.getElementById("btn-arrange").addEventListener("click", () => this._arrange());

        // --- output panel tab switching (Result / History) ---
        document.querySelectorAll(".panel-tab").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".panel-tab").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const view = btn.dataset.view;
                document.getElementById("result-view").style.display  = view === "result"  ? "flex" : "none";
                document.getElementById("history-view").style.display = view === "history" ? "flex" : "none";
                if (view === "history") this._loadHistory();
            });
        });

        // --- top toolbar tab switching (Inference / Training / Dataset) ---
        document.querySelectorAll(".tab-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const tab = btn.dataset.tab;
                document.getElementById("main").style.display          = tab === "inference" ? "flex" : "none";
                document.getElementById("training-main").style.display = tab === "training"  ? "flex" : "none";
                document.getElementById("dataset-main").style.display  = tab === "dataset"   ? "flex" : "none";
                if (tab === "inference") this.canvas.resize();
                if (tab === "dataset")  this._loadDatasetList();
            });
        });

        // --- dataset upload input ---
        document.getElementById("ds-upload-input").addEventListener("change", async e => {
            const file = e.target.files[0];
            if (!file) return;
            e.target.value = "";
            this._setStatus(`Uploading ${file.name}…`, "running");
            try {
                const { name } = await api.uploadDataset(file);
                this._setStatus("Ready");
                await this._loadDatasetList();
                this._openDataset(name);
            } catch (err) {
                this._setStatus(`Upload failed: ${err.message}`, "error");
            }
        });

        // --- right-click menu ---
        el.addEventListener("contextmenu", e => {
            e.preventDefault();
            // Convert screen → graph coords to check if we clicked an existing node
            const rect = el.getBoundingClientRect();
            const gx = (e.clientX - rect.left)  / this.canvas.ds.scale - this.canvas.ds.offset[0];
            const gy = (e.clientY - rect.top)   / this.canvas.ds.scale - this.canvas.ds.offset[1];
            const node = this.graph.getNodeOnPos(gx, gy);
            if (!node) this._showContextMenu(e.clientX, e.clientY);
        });
        document.addEventListener("click", () => this._hideContextMenu());

        // --- resize ---
        window.addEventListener("resize", () => this.canvas.resize());
        this.canvas.resize();

        // Start with the default graph
        this._loadDefaultGraph();
    }

    // ── Node type registration ────────────────────────────────────────────

    _registerNodeType(nodeId, nodeData) {
        const app = this;

        // Split inputs into connection slots vs. widget inputs
        const connInputs   = [];  // {name, type}
        const widgetInputs = [];  // {name, type, values?, config}

        for (const section of ["required", "optional"]) {
            const entries = nodeData.input?.[section];
            if (!entries) continue;
            for (const [name, [typeInfo, config]] of Object.entries(entries)) {
                if (Array.isArray(typeInfo)) {
                    widgetInputs.push({ name, type: "COMBO", values: typeInfo, config });
                } else if (WIDGET_TYPES.has(typeInfo)) {
                    widgetInputs.push({ name, type: typeInfo, config });
                } else {
                    connInputs.push({ name, type: typeInfo });
                }
            }
        }

        function NodeClass() {
            for (const { name, type } of connInputs) {
                this.addInput(name, type);
            }
            for (const w of widgetInputs) {
                if (w.type === "COMBO") {
                    this.addWidget("combo", w.name, w.values[0], null, { values: w.values });
                } else if (w.type === "INT") {
                    this.addWidget("number", w.name, w.config?.default ?? 0, null, {
                        min: w.config?.min ?? -1e9,
                        max: w.config?.max ?? 1e9,
                        step: (w.config?.step ?? 1) * 10,
                        precision: 0,
                    });
                } else if (w.type === "FLOAT") {
                    this.addWidget("number", w.name, w.config?.default ?? 0.0, null, {
                        min: w.config?.min ?? -1e9,
                        max: w.config?.max ?? 1e9,
                        step: (w.config?.step ?? 0.01) * 10,
                        precision: 4,
                    });
                } else if (w.type === "STRING") {
                    this.addWidget("text", w.name, w.config?.default ?? "");
                } else if (w.type === "BOOLEAN") {
                    this.addWidget("toggle", w.name, w.config?.default ?? false);
                }
            }
            for (let i = 0; i < nodeData.output.length; i++) {
                const type = nodeData.output[i];
                const name = nodeData.output_name?.[i] ?? type;
                this.addOutput(name, type);
            }

            this.color   = "#2d3748";
            this.bgcolor = "#1e2535";
        }

        NodeClass.title    = nodeData.display_name || nodeId;
        NodeClass.category = nodeData.category;
        NodeClass.prototype.comfy_class          = nodeId;
        NodeClass.prototype.comfy_conn_inputs    = connInputs;
        NodeClass.prototype.comfy_widget_inputs  = widgetInputs;
        NodeClass.prototype.comfy_is_output_node = nodeData.output_node;

        if (nodeData.output_node) {
            NodeClass.prototype.color   = "#2d3a2d";
            NodeClass.prototype.bgcolor = "#1e2a1e";
        }

        LiteGraph.registerNodeType(`${nodeData.category}/${nodeId}`, NodeClass);
    }

    // ── Graph serialisation ───────────────────────────────────────────────

    _serialiseGraph() {
        const prompt = {};

        for (const node of this.graph._nodes) {
            if (!node.comfy_class) continue;

            const inputs = {};

            let slotIdx = 0;
            for (const { name } of (node.comfy_conn_inputs ?? [])) {
                const slot = node.inputs?.[slotIdx];
                if (slot?.link != null) {
                    const link = this.graph.links[slot.link];
                    if (link) inputs[name] = [String(link.origin_id), link.origin_slot];
                }
                slotIdx++;
            }

            let widgetIdx = 0;
            for (const { name } of (node.comfy_widget_inputs ?? [])) {
                const widget = node.widgets?.[widgetIdx];
                if (widget !== undefined) inputs[name] = widget.value;
                widgetIdx++;
            }

            prompt[String(node.id)] = { class_type: node.comfy_class, inputs };
        }

        return prompt;
    }

    // ── Execution ─────────────────────────────────────────────────────────

    async _queuePrompt() {
        if (this.running) return;
        const prompt = this._serialiseGraph();
        if (Object.keys(prompt).length === 0) {
            this._setStatus("Graph is empty", "error");
            return;
        }

        this.running = true;
        document.getElementById("btn-queue").disabled = true;
        this._setStatus("Running…", "running");
        this._clearOutput();

        try {
            const result = await api.queuePrompt(prompt);
            this._switchOutputTab("result");
            this._displayResults(result.ui ?? {});
            this._setStatus("Done", "success");
        } catch (err) {
            this._setStatus(`Error: ${err.message}`, "error");
            this._showError(err.message);
        } finally {
            this.running = false;
            document.getElementById("btn-queue").disabled = false;
        }
    }

    // ── Output panel ──────────────────────────────────────────────────────

    _clearOutput() {
        document.getElementById("output-content").innerHTML =
            '<p class="empty-hint">Running…</p>';
    }

    _displayResults(ui) {
        const container = document.getElementById("output-content");
        container.innerHTML = "";

        if (Object.keys(ui).length === 0) {
            container.innerHTML = '<p class="empty-hint">No output nodes in graph.</p>';
            return;
        }

        for (const [nodeId, data] of Object.entries(ui)) {
            const card = document.createElement("div");
            card.className = "result-card";

            const title = document.createElement("div");
            title.className = "card-title";
            title.textContent = `${data.class_type}  (node ${nodeId})`;
            card.appendChild(title);

            const textParts = [];
            const images    = [];

            for (const out of data.outputs ?? []) {
                if (out.type === "STRING") {
                    if (out.value) textParts.push(out.value);
                } else if (out.type === "IMAGE") {
                    if (out.value && out.value.startsWith("data:")) {
                        images.push({ name: out.name, src: out.value });
                    }
                } else {
                    textParts.push(`${out.name}: ${out.value}`);
                }
            }

            if (textParts.length) {
                const body = document.createElement("div");
                body.className = "card-body";
                body.textContent = textParts.join("\n\n");
                card.appendChild(body);
            }

            for (const { name, src } of images) {
                const img = document.createElement("img");
                img.src       = src;
                img.alt       = name;
                img.className = "result-plot";
                card.appendChild(img);
            }

            container.appendChild(card);
        }
    }

    _showError(msg) {
        const container = document.getElementById("output-content");
        container.innerHTML = `<div class="result-card">
            <div class="card-title" style="color:#f44336">Error</div>
            <div class="card-body" style="color:#f44336;white-space:pre-wrap">${msg}</div>
        </div>`;
    }

    _setStatus(msg, cls = "") {
        const el = document.getElementById("status");
        el.textContent = msg;
        el.className = cls ? `status-text ${cls}` : "status-text";
    }

    // ── Toolbar actions ───────────────────────────────────────────────────

    _clearGraph() {
        if (!confirm("Clear the graph?")) return;
        this.graph.clear();
        document.getElementById("output-content").innerHTML =
            '<p class="empty-hint">Add nodes with right-click.</p>';
        this._setStatus("Ready");
    }

    _arrange() {
        this.graph.arrange();
        this.canvas.setDirty(true, true);
    }

    // ── Context menu (Add Node) ───────────────────────────────────────────

    _showContextMenu(x, y) {
        const menu = document.getElementById("ctx-menu");

        const byCategory = {};
        for (const [nodeId, data] of Object.entries(this.nodeTypes)) {
            const cat = data.category || "default";
            (byCategory[cat] = byCategory[cat] ?? []).push({ nodeId, data });
        }

        menu.innerHTML = "";
        for (const [cat, nodes] of Object.entries(byCategory)) {
            const sec = document.createElement("div");
            sec.className = "menu-section";
            sec.textContent = cat;
            menu.appendChild(sec);

            for (const { nodeId, data } of nodes) {
                const item = document.createElement("div");
                item.className = "menu-item";
                item.textContent = data.display_name || nodeId;
                item.addEventListener("click", () => {
                    this._addNode(nodeId, x, y);
                    this._hideContextMenu();
                });
                menu.appendChild(item);
            }
        }

        menu.style.left    = `${x}px`;
        menu.style.top     = `${y}px`;
        menu.style.display = "block";

        const rect = menu.getBoundingClientRect();
        if (rect.right  > window.innerWidth)  menu.style.left = `${x - rect.width}px`;
        if (rect.bottom > window.innerHeight) menu.style.top  = `${y - rect.height}px`;
    }

    _hideContextMenu() {
        document.getElementById("ctx-menu").style.display = "none";
    }

    _addNode(nodeId, screenX, screenY) {
        const data = this.nodeTypes[nodeId];
        if (!data) return;
        const type = `${data.category}/${nodeId}`;
        const node = LiteGraph.createNode(type);
        if (!node) return;

        const canvasRect = document.getElementById("graph-canvas").getBoundingClientRect();
        const gx = (screenX - canvasRect.left) / this.canvas.ds.scale - this.canvas.ds.offset[0];
        const gy = (screenY - canvasRect.top)  / this.canvas.ds.scale - this.canvas.ds.offset[1];
        node.pos = [gx, gy];

        this.graph.add(node);
        this.canvas.setDirty(true, true);
    }

    // ── Output panel tab helpers ──────────────────────────────────────────

    _switchOutputTab(view) {
        document.querySelectorAll(".panel-tab").forEach(b => {
            b.classList.toggle("active", b.dataset.view === view);
        });
        document.getElementById("result-view").style.display  = view === "result"  ? "flex" : "none";
        document.getElementById("history-view").style.display = view === "history" ? "flex" : "none";
    }

    async _loadHistory() {
        const container = document.getElementById("history-content");
        container.innerHTML = '<p class="empty-hint">Loading…</p>';
        try {
            const runs = await api.getHistory();
            container.innerHTML = "";
            if (!runs.length) {
                container.innerHTML = '<p class="empty-hint">No runs yet.</p>';
                return;
            }
            for (const run of runs) {
                const item = document.createElement("div");
                item.className = "history-item";
                item.dataset.runId = run.run_id;

                const ts = run.timestamp.replace("T", " ");
                item.innerHTML =
                    `<div class="hi-time">${ts}</div>` +
                    `<div class="hi-label">${run.label || "(no output)"}</div>`;

                item.addEventListener("click", () => this._showHistoryRun(run.run_id, item));
                container.appendChild(item);
            }
        } catch (err) {
            container.innerHTML = `<p class="empty-hint">Error: ${err.message}</p>`;
        }
    }

    async _showHistoryRun(runId, itemEl) {
        // Mark active item
        document.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"));
        itemEl.classList.add("active");

        try {
            const data = await api.getHistoryRun(runId);
            this._switchOutputTab("result");
            this._displayResults(data.result?.ui ?? {});
        } catch (err) {
            this._showError(`Failed to load run: ${err.message}`);
        }
    }

    // ── Dataset tab ───────────────────────────────────────────────────────

    async _loadDatasetList() {
        const list = document.getElementById("ds-list");
        list.innerHTML = '<p class="empty-hint" style="margin-top:20px">Loading…</p>';
        try {
            const datasets = await api.listDatasets();
            list.innerHTML = "";
            if (!datasets.length) {
                list.innerHTML = '<p class="empty-hint" style="margin-top:20px">No datasets.</p>';
                return;
            }
            for (const ds of datasets) {
                const item = document.createElement("div");
                item.className = "ds-list-item";
                if (ds.name === this._dsCurrentName) item.classList.add("active");
                item.dataset.name = ds.name;
                item.innerHTML =
                    `<span class="ds-name">${ds.name}</span>` +
                    `<span class="ds-meta">${ds.num_frames ?? "?"} frames</span>`;
                item.addEventListener("click", () => this._openDataset(ds.name));
                list.appendChild(item);
            }
        } catch (err) {
            list.innerHTML = `<p class="empty-hint" style="margin-top:20px">Error: ${err.message}</p>`;
        }
    }

    async _openDataset(name) {
        this._dsCurrentName = name;
        this._dsCurrentPage = 0;

        // Highlight active item in sidebar
        document.querySelectorAll(".ds-list-item").forEach(el => {
            el.classList.toggle("active", el.dataset.name === name);
        });

        document.getElementById("ds-placeholder").style.display = "none";
        const content = document.getElementById("ds-content");
        content.style.display = "flex";

        const header = document.getElementById("ds-header");
        header.innerHTML = `<span class="ds-hdr-title">${name}</span><span style="color:var(--text-dim);font-size:11px">Loading…</span>`;

        try {
            const info = await api.getDatasetInfo(name);
            this._dsInfo = info;
            header.innerHTML =
                `<span class="ds-hdr-title">${name}</span>` +
                `<span class="ds-hdr-badge">${info.num_frames} frames</span>` +
                `<span style="color:var(--text-dim);font-size:11px">fields: ${Object.keys(info.fields ?? {}).join(", ")}</span>`;
            await this._loadDatasetPage(0);
        } catch (err) {
            header.innerHTML = `<span class="ds-hdr-title">${name}</span><span style="color:#f44336"> Error: ${err.message}</span>`;
        }
    }

    async _loadDatasetPage(page) {
        const start = page * this._dsPageSize;
        const tbody = document.getElementById("ds-tbody");
        const thead = document.getElementById("ds-thead-row");
        tbody.innerHTML = '<tr><td colspan="99" style="text-align:center;padding:12px">Loading…</td></tr>';

        try {
            const frames = await api.getDatasetFrames(this._dsCurrentName, start, this._dsPageSize);
            if (!frames.length) {
                tbody.innerHTML = '<tr><td colspan="99" style="text-align:center;padding:12px">No frames.</td></tr>';
                return;
            }
            // Build header from first frame keys
            const cols = Object.keys(frames[0]).filter(k => k !== "index");
            thead.innerHTML = "<th>#</th>" + cols.map(c => `<th>${c}</th>`).join("");

            const INT_COLS   = new Set(["num_atoms"]);
            const FLOAT_COLS = new Set(["energy", "max_force", "dipole_norm", "total_charge"]);

            tbody.innerHTML = "";
            for (const frame of frames) {
                const tr = document.createElement("tr");
                tr.dataset.index = frame.index;
                tr.innerHTML = `<td>${frame.index}</td>` +
                    cols.map(c => {
                        const v = frame[c];
                        if (v == null) return "<td>—</td>";
                        if (typeof v === "number") {
                            if (INT_COLS.has(c))   return `<td>${v}</td>`;
                            if (FLOAT_COLS.has(c)) return `<td>${v.toFixed(6)}</td>`;
                            return `<td>${v}</td>`;
                        }
                        return `<td>${v}</td>`;
                    }).join("");
                tr.addEventListener("click", () => this._loadFrameDetail(frame.index, tr));
                tbody.appendChild(tr);
            }

            // Pagination
            const totalPages = this._dsInfo ? Math.ceil(this._dsInfo.num_frames / this._dsPageSize) : 1;
            const pg = document.getElementById("ds-pagination");
            pg.innerHTML = "";
            if (totalPages > 1) {
                const prev = document.createElement("button");
                prev.textContent = "← Prev";
                prev.disabled = page === 0;
                prev.addEventListener("click", () => { this._dsCurrentPage--; this._loadDatasetPage(this._dsCurrentPage); });
                pg.appendChild(prev);

                const label = document.createElement("span");
                label.textContent = ` Page ${page + 1} / ${totalPages} `;
                pg.appendChild(label);

                const next = document.createElement("button");
                next.textContent = "Next →";
                next.disabled = page >= totalPages - 1;
                next.addEventListener("click", () => { this._dsCurrentPage++; this._loadDatasetPage(this._dsCurrentPage); });
                pg.appendChild(next);
            }
        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="99" style="color:#f44336;padding:12px">Error: ${err.message}</td></tr>`;
        }
    }

    _fmtNum(x) {
        if (typeof x !== "number") return String(x);
        if (Number.isInteger(x))   return String(x);
        return x.toFixed(6);
    }

    _fmtArray(v) {
        // v is a JS array (possibly nested)
        const is2D = Array.isArray(v[0]);
        if (is2D) {
            const rows = v.slice(0, 30);
            const more = v.length > 30 ? `\n… (${v.length} rows total)` : "";
            return rows.map(row =>
                row.map(x => (typeof x === "number" ? x.toFixed(6).padStart(14) : String(x))).join("  ")
            ).join("\n") + more;
        }
        // 1-D
        const flat = v.slice(0, 60);
        const more = v.length > 60 ? `, … (${v.length} total)` : "";
        return flat.map(x => this._fmtNum(x)).join(", ") + more;
    }

    async _loadFrameDetail(index, rowEl) {
        document.querySelectorAll("#ds-tbody tr").forEach(r => r.classList.remove("active"));
        rowEl.classList.add("active");

        const panel = document.getElementById("ds-frame-panel");
        panel.innerHTML = '<div class="ds-frame-panel-hint">Loading frame…</div>';
        try {
            const frame = await api.getDatasetFrame(this._dsCurrentName, index);

            // Separate scalars from arrays for layout
            const scalars = [];
            const arrays  = [];
            for (const [k, v] of Object.entries(frame)) {
                if (k === "index") continue;
                if (Array.isArray(v)) arrays.push([k, v]);
                else scalars.push([k, v]);
            }

            // Title row
            const formula = frame.formula ?? "";
            let html = `<div class="ds-frame-title">Frame ${index}${formula ? "  " + formula : ""}</div>`;

            // Scalar grid
            if (scalars.length) {
                html += `<div class="ds-frame-scalars">`;
                for (const [k, v] of scalars) {
                    const display = v == null ? "—" : this._fmtNum(v);
                    html += `<div class="ds-scalar-cell"><span class="ds-field-key">${k}</span><span class="ds-scalar-val">${display}</span></div>`;
                }
                html += `</div>`;
            }

            // Array fields
            html += `<div class="ds-frame-fields">`;
            for (const [k, v] of arrays) {
                const shape = Array.isArray(v[0]) ? `${v.length}×${v[0].length}` : `${v.length}`;
                html += `<div class="ds-field-row">` +
                    `<span class="ds-field-key">${k} <span style="font-weight:400;text-transform:none">[${shape}]</span></span>` +
                    `<pre class="ds-field-val">${this._fmtArray(v)}</pre>` +
                    `</div>`;
            }
            html += `</div>`;

            panel.innerHTML = html;
        } catch (err) {
            panel.innerHTML = `<div class="ds-frame-panel-hint" style="color:#f44336">Error: ${err.message}</div>`;
        }
    }

    // ── Default graph ─────────────────────────────────────────────────────

    _loadDefaultGraph() {
        if (!this.nodeTypes["ReadXYZFile"]  ||
            !this.nodeTypes["RunMLPModel"] ||
            !this.nodeTypes["OutputEnergy"] ||
            !this.nodeTypes["OutputForces"]) return;

        // ReadXYZFile → atoms_list → RunMLPModel → result → OutputEnergy / OutputForces

        const xyzNode = LiteGraph.createNode("mlp/input/ReadXYZFile");
        xyzNode.pos = [60, 220];
        this.graph.add(xyzNode);

        const modelNode = LiteGraph.createNode("mlp/inference/RunMLPModel");
        modelNode.pos = [360, 200];
        this.graph.add(modelNode);

        const energyNode = LiteGraph.createNode("mlp/output/OutputEnergy");
        energyNode.pos = [660, 100];
        this.graph.add(energyNode);

        const forcesNode = LiteGraph.createNode("mlp/output/OutputForces");
        forcesNode.pos = [660, 280];
        this.graph.add(forcesNode);

        // ReadXYZFile.atoms (slot 0) → LoadMLPModel.atoms (slot 0)
        xyzNode.connect(0, modelNode, 0);
        // LoadMLPModel.result (slot 0) → OutputEnergy.result (slot 0)
        modelNode.connect(0, energyNode, 0);
        // LoadMLPModel.result (slot 0) → OutputForces.result (slot 0)
        modelNode.connect(0, forcesNode, 0);

        this.canvas.setDirty(true, true);
        document.getElementById("output-content").innerHTML =
            '<p class="empty-hint">Default graph loaded.<br>Set filepath, select model, click <b>Queue</b>.</p>';
    }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

const app = new MLPUIApp();

window.addEventListener("load", async () => {
    LiteGraph.NODE_TITLE_HEIGHT    = 24;
    LiteGraph.NODE_SLOT_HEIGHT     = 20;
    LiteGraph.NODE_WIDTH           = 220;
    LiteGraph.DEFAULT_SHADOW_COLOR = "rgba(0,0,0,0)";
    LiteGraph.LINK_COLOR           = "#9c6fd6";

    // Register wire colours per type.
    // Must include nodes:[] because registerNodeAndSlotType() pushes into it.
    for (const [type, color] of Object.entries(TYPE_COLORS)) {
        LiteGraph.registered_slot_in_types  = LiteGraph.registered_slot_in_types  || {};
        LiteGraph.registered_slot_out_types = LiteGraph.registered_slot_out_types || {};
        LiteGraph.registered_slot_in_types[type]  = { color, nodes: [] };
        LiteGraph.registered_slot_out_types[type] = { color, nodes: [] };
    }

    try {
        await app.setup();
        app._setStatus("Ready");
    } catch (err) {
        console.error("Failed to initialise:", err);
        app._setStatus("Init error", "error");
        document.getElementById("output-content").innerHTML =
            `<div class="result-card">
               <div class="card-title" style="color:#f44336">Init Error</div>
               <div class="card-body" style="color:#f44336;white-space:pre-wrap">${err.stack || err.message || String(err)}</div>
             </div>`;
    }
});
