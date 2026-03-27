/**
 * app.js — MLPUI node-graph application.
 *
 * Design mirrors ComfyUI:
 *   - /object_info  → register LiteGraph node types
 *   - Graph JSON    → POST /prompt → execution result → output panel
 */

// Colour palette for connection wires by type
const TYPE_COLORS = {
    MLP_CALCULATOR: "#4a9eff",
    ASE_ATOMS:      "#f0a500",
    FLOAT:          "#b5cea8",
    STRING:         "#ce9178",
    INT:            "#b5cea8",
    BOOLEAN:        "#569cd6",
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
        this.canvas.background_image = "";
        this.canvas.render_shadows = false;
        this.canvas.render_canvas_border = false;
        this.canvas.always_render_background = true;
        this.canvas.show_info = false;

        // --- load node types ---
        const info = await api.getObjectInfo();
        this.nodeTypes = info;
        for (const [nodeId, nodeData] of Object.entries(info)) {
            this._registerNodeType(nodeId, nodeData);
        }

        // --- toolbar wiring ---
        document.getElementById("btn-queue").addEventListener("click", () => this._queuePrompt());
        document.getElementById("btn-clear").addEventListener("click", () => this._clearGraph());
        document.getElementById("btn-arrange").addEventListener("click", () => this._arrange());

        // --- right-click menu ---
        el.addEventListener("contextmenu", e => {
            e.preventDefault();
            // Convert screen coords → graph coords to check if we clicked a node
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

        // Start with an example graph
        this._loadDefaultGraph();

        console.log("MLPUI ready. Registered node types:", Object.keys(info));
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
            // Connection input slots
            for (const { name, type } of connInputs) {
                this.addInput(name, type);
            }
            // Widget inputs
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
            // Output slots
            for (let i = 0; i < nodeData.output.length; i++) {
                const type = nodeData.output[i];
                const name = nodeData.output_name?.[i] ?? type;
                this.addOutput(name, type);
            }

            this.color    = "#2d3748";
            this.bgcolor  = "#1e2535";
        }

        NodeClass.title        = nodeData.display_name || nodeId;
        NodeClass.category     = nodeData.category;
        NodeClass.prototype.comfy_class          = nodeId;
        NodeClass.prototype.comfy_conn_inputs    = connInputs;
        NodeClass.prototype.comfy_widget_inputs  = widgetInputs;
        NodeClass.prototype.comfy_is_output_node = nodeData.output_node;

        // Colour output-nodes differently
        if (nodeData.output_node) {
            NodeClass.prototype.color   = "#2d3a2d";
            NodeClass.prototype.bgcolor = "#1e2a1e";
        }

        // Wire type colours
        for (const type of nodeData.output) {
            if (TYPE_COLORS[type]) LiteGraph.registered_node_types; // ensure registered
        }

        LiteGraph.registerNodeType(`${nodeData.category}/${nodeId}`, NodeClass);
    }

    // ── Graph serialisation ───────────────────────────────────────────────

    _serialiseGraph() {
        const prompt = {};

        for (const node of this.graph._nodes) {
            if (!node.comfy_class) continue;

            const inputs = {};

            // Connection slot inputs
            let slotIdx = 0;
            for (const { name } of (node.comfy_conn_inputs ?? [])) {
                const slot = node.inputs?.[slotIdx];
                if (slot?.link != null) {
                    const link = this.graph.links[slot.link];
                    if (link) {
                        inputs[name] = [String(link.origin_id), link.origin_slot];
                    }
                }
                slotIdx++;
            }

            // Widget inputs
            let widgetIdx = 0;
            for (const { name } of (node.comfy_widget_inputs ?? [])) {
                const widget = node.widgets?.[widgetIdx];
                if (widget !== undefined) {
                    inputs[name] = widget.value;
                }
                widgetIdx++;
            }

            prompt[String(node.id)] = {
                class_type: node.comfy_class,
                inputs,
            };
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
            this._displayResults(result.ui ?? {});
            this._setStatus("Done ✓", "success");
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
                    body.textContent = out.value;
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
        el.className = cls;
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

        // Group node types by category
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

        // Keep inside viewport
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
        const type  = `${data.category}/${nodeId}`;
        const node  = LiteGraph.createNode(type);
        if (!node) return;

        // Convert screen coords to graph coords
        const canvasRect = document.getElementById("graph-canvas").getBoundingClientRect();
        const gx = (screenX - canvasRect.left) / this.canvas.ds.scale - this.canvas.ds.offset[0];
        const gy = (screenY - canvasRect.top)  / this.canvas.ds.scale - this.canvas.ds.offset[1];
        node.pos = [gx, gy];

        this.graph.add(node);
        this.canvas.setDirty(true, true);
    }

    // ── Default graph ─────────────────────────────────────────────────────

    _loadDefaultGraph() {
        if (!this.nodeTypes["LoadMLPModel"] ||
            !this.nodeTypes["CreateAtomsFromFormula"] ||
            !this.nodeTypes["CalculateEnergyForces"]) return;

        const loaderNode = LiteGraph.createNode("mlp/loaders/LoadMLPModel");
        loaderNode.pos = [60, 120];
        this.graph.add(loaderNode);

        const atomsNode = LiteGraph.createNode("mlp/input/CreateAtomsFromFormula");
        atomsNode.pos = [60, 320];
        this.graph.add(atomsNode);

        const calcNode = LiteGraph.createNode("mlp/calculate/CalculateEnergyForces");
        calcNode.pos = [420, 200];
        this.graph.add(calcNode);

        // Connect: loader.calculator → calc.calculator
        loaderNode.connect(0, calcNode, 0);
        // Connect: atoms.atoms → calc.atoms
        atomsNode.connect(0, calcNode, 1);

        this.canvas.setDirty(true, true);
        document.getElementById("output-content").innerHTML =
            '<p class="empty-hint">Default graph loaded.<br>Click <b>Queue</b> to run.</p>';
    }
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

const app = new MLPUIApp();

window.addEventListener("load", async () => {
    // Set litegraph defaults
    LiteGraph.NODE_TITLE_HEIGHT   = 24;
    LiteGraph.NODE_SLOT_HEIGHT    = 20;
    LiteGraph.NODE_WIDTH          = 220;
    LiteGraph.DEFAULT_SHADOW_COLOR = "rgba(0,0,0,0)";
    LiteGraph.LINK_COLOR          = "#4a9eff";

    // Register wire colours per type
    for (const [type, color] of Object.entries(TYPE_COLORS)) {
        LiteGraph.registered_slot_in_types  = LiteGraph.registered_slot_in_types  || {};
        LiteGraph.registered_slot_out_types = LiteGraph.registered_slot_out_types || {};
        LiteGraph.registered_slot_in_types[type]  = { color };
        LiteGraph.registered_slot_out_types[type] = { color };
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
