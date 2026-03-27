/**
 * api.js — thin wrapper around the MLPUI backend REST API.
 */
const api = {
    async getObjectInfo() {
        const r = await fetch("/object_info");
        if (!r.ok) throw new Error(`/object_info failed: ${r.status}`);
        return r.json();
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
