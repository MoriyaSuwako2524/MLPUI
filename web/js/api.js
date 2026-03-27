/**
 * api.js — thin wrapper around the MLPUI backend REST API.
 */
const api = {
    async getObjectInfo() {
        const r = await fetch("/object_info");
        if (!r.ok) throw new Error(`/object_info failed: ${r.status}`);
        return r.json();
    },

    async getHistory() {
        const r = await fetch("/history");
        if (!r.ok) throw new Error(`/history failed: ${r.status}`);
        return r.json();   // [{run_id, timestamp, label}]
    },

    async getHistoryRun(runId) {
        const r = await fetch(`/history/${encodeURIComponent(runId)}`);
        if (!r.ok) throw new Error(`/history/${runId} failed: ${r.status}`);
        return r.json();   // {run_id, timestamp, prompt, result}
    },

    async getXyzFiles() {
        const r = await fetch("/xyz_files");
        if (!r.ok) throw new Error(`/xyz_files failed: ${r.status}`);
        return r.json();   // string[]
    },

    async uploadFile(file) {
        const form = new FormData();
        form.append("file", file);
        const r = await fetch("/upload", { method: "POST", body: form });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
        return data;   // { path, filename }
    },

    async queuePrompt(prompt) {
        const r = await fetch("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
        return data;
    },
};
