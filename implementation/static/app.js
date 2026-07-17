/**
 * app.js — Frontend logic for the Petroleum Engineering RAG chat interface.
 * Vanilla JavaScript, no frameworks.
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("question-input");
    const askBtn = document.getElementById("ask-btn");
    const btnText = askBtn.querySelector(".btn-text");
    const btnLoading = askBtn.querySelector(".btn-loading");

    const loadingSection = document.getElementById("loading-section");
    const errorSection = document.getElementById("error-section");
    const errorMessage = document.getElementById("error-message");
    const answerSection = document.getElementById("answer-section");
    const answerText = document.getElementById("answer-text");
    const sourcesSection = document.getElementById("sources-section");
    const sourcesList = document.getElementById("sources-list");
    const chunksSection = document.getElementById("chunks-section");
    const chunksList = document.getElementById("chunks-list");
    const toggleChunksBtn = document.getElementById("toggle-chunks");

    let chunksVisible = true;

    // --- Form submission ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = input.value.trim();
        if (!question) {
            showError("Please enter a question.");
            return;
        }
        await askQuestion(question);
    });

    // --- Enter to submit, Shift+Enter for newline ---
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event("submit"));
        }
    });

    // --- Toggle chunks visibility ---
    toggleChunksBtn.addEventListener("click", () => {
        chunksVisible = !chunksVisible;
        chunksList.style.display = chunksVisible ? "flex" : "none";
        toggleChunksBtn.textContent = chunksVisible ? "▼" : "▶";
    });

    /**
     * Send a question to the /chat endpoint and display the results.
     */
    async function askQuestion(question) {
        // Show loading state
        setLoading(true);
        hideAll();

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(err.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    }

    /**
     * Display the RAG response in the UI.
     */
    function displayResults(data) {
        // Answer
        answerText.textContent = data.answer;
        answerSection.hidden = false;

        // Sources
        if (data.sources && data.sources.length > 0) {
            sourcesList.innerHTML = "";
            const seen = new Set();
            data.sources.forEach((src) => {
                const key = `${src.page}-${src.chapter}`;
                if (seen.has(key)) return;
                seen.add(key);

                const tag = document.createElement("span");
                tag.className = "source-tag";
                tag.innerHTML = `<span class="page-num">p.${src.page}</span> ${escapeHtml(src.chapter)}`;
                sourcesList.appendChild(tag);
            });
            sourcesSection.hidden = false;
        }

        // Retrieved chunks
        if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
            chunksList.innerHTML = "";
            data.retrieved_chunks.forEach((chunk) => {
                const item = document.createElement("div");
                item.className = "chunk-item";
                item.innerHTML = `
                    <div class="chunk-header">
                        <span class="chunk-id">${escapeHtml(chunk.chunk_id)}</span>
                        <span class="chunk-score">Score: ${chunk.score.toFixed(4)}</span>
                    </div>
                    <div class="chunk-meta">Page ${chunk.page} &mdash; ${escapeHtml(chunk.chapter)}</div>
                    <div class="chunk-text">${escapeHtml(chunk.text)}</div>
                `;
                chunksList.appendChild(item);
            });
            chunksSection.hidden = false;
            chunksList.style.display = "flex";
            chunksVisible = true;
            toggleChunksBtn.textContent = "▼";
        }
    }

    /**
     * Show an error message.
     */
    function showError(message) {
        errorMessage.textContent = message;
        errorSection.hidden = false;
    }

    /**
     * Hide all result sections.
     */
    function hideAll() {
        errorSection.hidden = true;
        answerSection.hidden = true;
        sourcesSection.hidden = true;
        chunksSection.hidden = true;
    }

    /**
     * Toggle loading state on the button and loading section.
     */
    function setLoading(isLoading) {
        askBtn.disabled = isLoading;
        btnText.hidden = isLoading;
        btnLoading.hidden = !isLoading;
        loadingSection.hidden = !isLoading;
    }

    /**
     * Escape HTML to prevent XSS.
     */
    function escapeHtml(text) {
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }
});
