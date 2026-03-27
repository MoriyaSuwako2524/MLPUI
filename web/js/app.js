/**
 * app.js — MLPUI node-graph application.
 */

// Colour palette for connection wires by type
const TYPE_COLORS = {
    MLP_RESULT:      "#9c6fd6",
    ASE_ATOMS:       "#f0a500",
    ASE_ATOMS_LIST:  "#e07820",
    BATCH_RESULT:    "#c85cc8",
    FLOAT:           "#b5cea8",
    STRING:          "#ce9178",
    INT:             "#b5cea8",
    BOOLEAN:         "#569cd6",
};

// Widget types that live inside the node (not connection slots)
const WIDGET_TYPES = new Set(["STRING", "INT", "FLOAT", "BOOLEAN"]);

class MLPUIApp {
    constructor() {
        this.graph      = new LGraph();
        this.canvas     = null;
        this.nodeTypes  = {};   // nodeId → object_info entry
        this.running    = false;
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

        // --- top toolbar tab switching (Inference / Training) ---
        document.querySelectorAll(".tab-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const tab = btn.dataset.tab;
                document.getElementById("main").style.display          = tab === "inference" ? "flex" : "none";
                document.getElementById("training-main").style.display = tab === "training"  ? "flex" : "none";
                if (tab === "inference") this.canvas.resize();
            });
        });

        // --- drag-and-drop XYZ files ---
        el.addEventListener("dragover", e => {
            const hasFiles = [...e.dataTransfer.types].includes("Files");
            if (!hasFiles) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            document.getElementById("canvas-wrap").classList.add("drag-over");
        });
        el.addEventListener("dragleave", () => {
            document.getElementById("canvas-wrap").classList.remove("drag-over");
        });
        el.addEventListener("drop", async e => {
            e.preventDefault();
            document.getElementById("canvas-wrap").classList.remove("drag-over");
            const files = [...e.dataTransfer.files].filter(
                f => /\.(xyz|extxyz)$/i.test(f.name)
            );
            if (!files.length) return;
            const rect = el.getBoundingClientRect();
            let offsetX = 0;
            for (const file of files) {
                try {
                    this._setStatus(`Uploading ${file.name}…`, "running");
                    const { filename } = await api.uploadFile(file);

                    // Re-fetch the updated node definition so the COMBO reflects new file
                    const info = await api.getObjectInfo();
                    if (info["ReadXYZFile"]) {
                        this.nodeTypes["ReadXYZFile"] = info["ReadXYZFile"];
                        this._registerNodeType("ReadXYZFile", info["ReadXYZFile"]);
                    }

                    const gx = (e.clientX - rect.left)  / this.canvas.ds.scale - this.canvas.ds.offset[0] + offsetX;
                    const gy = (e.clientY - rect.top)   / this.canvas.ds.scale - this.canvas.ds.offset[1];
                    this._createXYZNode(filename, gx, gy);
                    offsetX += 260;
                    this._setStatus("Ready");
                } catch (err) {
                    this._setStatus(`Upload failed: ${err.message}`, "error");
                }
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

            const body = document.createElement("div");
            body.className = "card-body";

            for (const out of data.outputs ?? []) {
                if (out.type === "STRING") {
                    // Pre-formatted text — show directly
                    if (body.textContent) body.textContent += "\n\n";
                    body.textContent += out.value;
                } else {
                    body.textContent += `${out.name}: ${out.value}\n`;
                }
            }

            card.appendChild(body);
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

    // ── XYZ drop helper ──────────────────────────────────────────────────

    _createXYZNode(filename, gx, gy) {
        const node = LiteGraph.createNode("mlp/input/ReadXYZFile");
        if (!node) return;
        node.pos   = [gx, gy];
        node.title = filename;
        // Select the just-uploaded file in the COMBO widget
        if (node.widgets?.[0]) node.widgets[0].value = filename;
        this.graph.add(node);
        this.canvas.setDirty(true, true);
    }

    // ── Default graph ─────────────────────────────────────────────────────

    _loadDefaultGraph() {
        if (!this.nodeTypes["ReadXYZFile"]  ||
            !this.nodeTypes["LoadMLPModel"] ||
            !this.nodeTypes["OutputEnergy"] ||
            !this.nodeTypes["OutputForces"]) return;

        // ReadXYZFile → atoms → LoadMLPModel → result → OutputEnergy
        //                                             ↘ OutputForces

        const xyzNode = LiteGraph.createNode("mlp/input/ReadXYZFile");
        xyzNode.pos = [60, 220];
        this.graph.add(xyzNode);

        const modelNode = LiteGraph.createNode("mlp/inference/LoadMLPModel");
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
