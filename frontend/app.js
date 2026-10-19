/**
 * RAGNova NLP Document Assistant - Interactive Frontend Engine
 */

document.addEventListener("DOMContentLoaded", () => {
    // App State
    const state = {
        documents: [],
        currentDocId: null,
        chunks: [],
        chatHistory: [],
        settings: {
            chunk_size: 600,
            chunk_overlap: 120,
            top_k: 4,
            retrieval_mode: "hybrid",
            alpha: 0.5,
            gemini_model: "gemini-2.5-flash",
            has_api_key: false
        },
        citationsCache: {}
    };

    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadProgressBar = document.getElementById("upload-progress-bar");
    const uploadProgressFill = document.getElementById("upload-progress-fill");
    const btnLoadSample = document.getElementById("btn-load-sample");
    const documentList = document.getElementById("document-list");
    const emptyDocsState = document.getElementById("empty-docs-state");
    const docCounter = document.getElementById("doc-counter");
    const footerChunkCount = document.getElementById("footer-chunk-count");
    const footerRetrievalMode = document.getElementById("footer-retrieval-mode");
    const headerProviderBadge = document.getElementById("header-provider-badge");

    // Tabs
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Chat Elements
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const btnSend = document.getElementById("btn-send");
    const btnClearChat = document.getElementById("btn-clear-chat");
    const promptsChips = document.getElementById("prompts-chips");
    const chatLatencyIndicator = document.getElementById("chat-latency-indicator");

    // Analytics Elements
    const analyticsDocSelect = document.getElementById("analytics-doc-select");
    const analyticsEmptyState = document.getElementById("analytics-empty-state");
    const analyticsContentView = document.getElementById("analytics-content-view");
    const statWords = document.getElementById("stat-words");
    const statDiversity = document.getElementById("stat-diversity");
    const statReadability = document.getElementById("stat-readability");
    const statReadingTime = document.getElementById("stat-reading-time");
    const nlpSummaryText = document.getElementById("nlp-summary-text");
    const nlpKeywordsList = document.getElementById("nlp-keywords-list");
    const nlpEntitiesList = document.getElementById("nlp-entities-list");

    // Chunk Explorer
    const chunksList = document.getElementById("chunks-list");
    const simQueryInput = document.getElementById("sim-query-input");
    const btnSimSearch = document.getElementById("btn-sim-search");

    // Citation Drawer
    const citationDrawer = document.getElementById("citation-drawer");
    const btnCloseDrawer = document.getElementById("btn-close-drawer");
    const drawerDocName = document.getElementById("drawer-doc-name");
    const drawerPagePill = document.getElementById("drawer-page-pill");
    const drawerChunkIdPill = document.getElementById("drawer-chunk-id-pill");
    const drawerScorePill = document.getElementById("drawer-score-pill");
    const drawerFullText = document.getElementById("drawer-full-text");

    // Settings Modal
    const settingsModal = document.getElementById("settings-modal");
    const btnOpenSettings = document.getElementById("btn-open-settings");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const btnCancelSettings = document.getElementById("btn-cancel-settings");
    const settingsForm = document.getElementById("settings-form");
    const inputApiKey = document.getElementById("input-api-key");
    const selectGeminiModel = document.getElementById("select-gemini-model");
    const selectRetrievalMode = document.getElementById("select-retrieval-mode");
    const rangeAlpha = document.getElementById("range-alpha");
    const valAlpha = document.getElementById("val-alpha");
    const rangeChunkSize = document.getElementById("range-chunk-size");
    const valChunkSize = document.getElementById("val-chunk-size");
    const rangeChunkOverlap = document.getElementById("range-chunk-overlap");
    const valChunkOverlap = document.getElementById("val-chunk-overlap");
    const rangeTopK = document.getElementById("range-top-k");
    const valTopK = document.getElementById("val-top-k");

    // Initialize App
    init();

    async function init() {
        setupEventListeners();
        await fetchSettings();
        await fetchDocuments();
    }

    function setupEventListeners() {
        // Tab switching
        tabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                tabButtons.forEach(b => b.classList.remove("active"));
                tabPanes.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                document.getElementById(targetTab).classList.add("active");
                if (targetTab === "tab-chunks") {
                    renderAllChunks();
                }
            });
        });

        // Dropzone interactions
        dropZone.addEventListener("click", () => fileInput.click());
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-active");
        });
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-active"));
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-active");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files);
            }
        });
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files);
            }
        });

        // 1-Click Load Sample Docs
        btnLoadSample.addEventListener("click", loadSampleDocuments);

        // Chat Form
        chatForm.addEventListener("submit", handleChatSubmit);
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
        btnClearChat.addEventListener("click", clearChatHistory);

        // Prompt Chips Click
        promptsChips.addEventListener("click", (e) => {
            const chip = e.target.closest(".prompt-chip");
            if (chip) {
                const text = chip.getAttribute("data-prompt") || chip.textContent;
                chatInput.value = text;
                chatInput.focus();
                chatForm.dispatchEvent(new Event("submit"));
            }
        });

        // Analytics Document Select
        analyticsDocSelect.addEventListener("change", (e) => {
            const docId = e.target.value;
            renderAnalytics(docId);
        });

        // Citation Drawer Close
        btnCloseDrawer.addEventListener("click", () => citationDrawer.classList.remove("open"));

        // Settings Modal
        btnOpenSettings.addEventListener("click", () => {
            syncSettingsToUI();
            settingsModal.style.display = "flex";
        });
        btnCloseSettings.addEventListener("click", () => settingsModal.style.display = "none");
        btnCancelSettings.addEventListener("click", () => settingsModal.style.display = "none");
        settingsModal.addEventListener("click", (e) => {
            if (e.target === settingsModal) settingsModal.style.display = "none";
        });
        settingsForm.addEventListener("submit", handleSettingsSubmit);

        // Range slider live feedback
        rangeAlpha.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            valAlpha.textContent = `${val.toFixed(2)} (${val < 0.4 ? "Vector Heavy" : val > 0.6 ? "BM25 Heavy" : "Balanced"})`;
        });
        rangeChunkSize.addEventListener("input", (e) => {
            valChunkSize.textContent = `${e.target.value} chars`;
        });
        rangeChunkOverlap.addEventListener("input", (e) => {
            valChunkOverlap.textContent = `${e.target.value} chars`;
        });
        rangeTopK.addEventListener("input", (e) => {
            valTopK.textContent = `${e.target.value} chunks`;
        });

        // Simulator search
        btnSimSearch.addEventListener("click", handleSimulateSearch);
        simQueryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") handleSimulateSearch();
        });
    }

    // =========================================================================
    // API Calls
    // =========================================================================
    async function fetchSettings() {
        try {
            const res = await fetch("/api/settings");
            const data = await res.json();
            state.settings = data;
            updateHeaderAndFooterBadges();
        } catch (err) {
            console.error("Failed to load settings:", err);
        }
    }

    async function fetchDocuments() {
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            state.documents = data.documents || [];
            renderDocumentList();
            updateDocSelector();
            updateHeaderAndFooterBadges();
        } catch (err) {
            console.error("Failed to load documents:", err);
        }
    }

    async function handleFileUpload(fileList) {
        uploadProgressBar.style.display = "block";
        uploadProgressFill.style.width = "20%";

        for (let i = 0; i < fileList.length; i++) {
            const file = fileList[i];
            const formData = new FormData();
            formData.append("file", file);

            try {
                uploadProgressFill.style.width = `${Math.round(((i + 1) / fileList.length) * 80)}%`;
                const res = await fetch("/api/documents/upload", {
                    method: "POST",
                    body: formData
                });
                if (!res.ok) {
                    const err = await res.json();
                    alert(`Error uploading ${file.name}: ${err.error}`);
                }
            } catch (err) {
                console.error("Upload error:", err);
            }
        }

        uploadProgressFill.style.width = "100%";
        setTimeout(() => {
            uploadProgressBar.style.display = "none";
            uploadProgressFill.style.width = "0%";
        }, 500);

        fileInput.value = "";
        await fetchDocuments();
        await fetchSettings();
    }

    async function loadSampleDocuments() {
        btnLoadSample.disabled = true;
        btnLoadSample.innerHTML = `<span>Loading Samples...</span>`;

        try {
            const res = await fetch("/api/documents/sample", { method: "POST" });
            const data = await res.json();
            await fetchDocuments();
            await fetchSettings();
            
            // Switch to chat and notify
            appendSystemNotice(`✅ Loaded ${data.documents.length} sample document(s). Ready to query!`);
        } catch (err) {
            console.error("Error loading samples:", err);
        } finally {
            btnLoadSample.disabled = false;
            btnLoadSample.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                </svg>
                <span>Load Sample Documents</span>
            `;
        }
    }

    async function deleteDocument(docId, filename) {
        if (!confirm(`Are you sure you want to remove "${filename}" from the RAG index?`)) return;

        try {
            const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
            if (res.ok) {
                await fetchDocuments();
                await fetchSettings();
                if (state.currentDocId === docId) {
                    state.currentDocId = null;
                    analyticsDocSelect.value = "";
                    renderAnalytics(null);
                }
            }
        } catch (err) {
            console.error("Error deleting doc:", err);
        }
    }

    // =========================================================================
    // Chat & RAG Generation
    // =========================================================================
    async function handleChatSubmit(e) {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Append user bubble
        appendUserMessage(query);
        chatInput.value = "";
        btnSend.disabled = true;

        // Append loading assistant placeholder
        const assistantPlaceholderId = appendAssistantLoadingBubble();
        chatLatencyIndicator.textContent = "🔍 Retrieving across BM25 & Dense Vector indices...";

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    history: state.chatHistory
                })
            });

            const data = await res.json();
            
            if (data.error) {
                updateAssistantBubble(assistantPlaceholderId, `⚠️ ${data.error}`, []);
            } else {
                // Cache citations
                if (data.citations) {
                    data.citations.forEach(c => {
                        state.citationsCache[c.chunk_id] = c;
                    });
                }

                updateAssistantBubble(assistantPlaceholderId, data.answer, data.citations);

                // Update latency indicator
                chatLatencyIndicator.textContent = `⚡ Latency: ${data.latency_ms}ms | Confidence: ${data.confidence.toUpperCase()} | Engine: ${data.provider_used || "Local"}`;

                // Update dynamic prompts
                if (data.suggested_questions && data.suggested_questions.length > 0) {
                    updateSuggestedQuestions(data.suggested_questions);
                }

                // Add to conversational history
                state.chatHistory.push({ role: "user", content: query });
                state.chatHistory.push({ role: "assistant", content: data.answer });
            }
        } catch (err) {
            updateAssistantBubble(assistantPlaceholderId, "❌ Network connection error while communicating with RAG server.", []);
            console.error(err);
        } finally {
            btnSend.disabled = false;
            chatInput.focus();
        }
    }

    function appendUserMessage(text) {
        const row = document.createElement("div");
        row.className = "message-row user-row";
        row.innerHTML = `
            <div class="avatar-user">You</div>
            <div class="message-bubble user-bubble">
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
    }

    function appendAssistantLoadingBubble() {
        const id = "msg_" + Math.random().toString(36).substr(2, 9);
        const row = document.createElement("div");
        row.className = "message-row assistant-row";
        row.id = id;
        row.innerHTML = `
            <div class="avatar-assistant">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="12 2 2 7 12 12 22 7 12 2" />
                    <polyline points="2 17 12 22 22 17" />
                    <polyline points="2 12 12 17 22 12" />
                </svg>
            </div>
            <div class="message-bubble assistant-bubble">
                <div class="message-header">
                    <span class="sender-name">RAGNova Assistant</span>
                    <span class="grounding-badge">Grounding Query...</span>
                </div>
                <div class="message-text">
                    <p><em>Synthesizing evidence and consulting indexed knowledge...</em></p>
                </div>
            </div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
        return id;
    }

    function updateAssistantBubble(elementId, markdownText, citations) {
        const row = document.getElementById(elementId);
        if (!row) return;

        const formattedHtml = parseMarkdownToHtml(markdownText);
        
        let citationsHtml = "";
        if (citations && citations.length > 0) {
            const chips = citations.map(c => `
                <button class="citation-chip" data-chunk-id="${c.chunk_id}" title="Click to view full verified excerpt">
                    <span>📄</span>
                    <span>${escapeHtml(c.filename)} (p. ${c.page_num})</span>
                </button>
            `).join("");

            citationsHtml = `
                <div class="citations-wrapper">
                    <span class="citations-label">Verified Grounding Sources:</span>
                    <div class="citation-chips-row">${chips}</div>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="avatar-assistant">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="12 2 2 7 12 12 22 7 12 2" />
                    <polyline points="2 17 12 22 22 17" />
                    <polyline points="2 12 12 17 22 12" />
                </svg>
            </div>
            <div class="message-bubble assistant-bubble">
                <div class="message-header">
                    <span class="sender-name">RAGNova Assistant</span>
                    <span class="grounding-badge">Evidence-Grounded</span>
                </div>
                <div class="message-text">${formattedHtml}</div>
                ${citationsHtml}
            </div>
        `;

        // Add click listeners to citation chips
        row.querySelectorAll(".citation-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const chunkId = chip.getAttribute("data-chunk-id");
                openCitationDrawer(chunkId);
            });
        });

        scrollToBottom();
    }

    function appendSystemNotice(text) {
        const div = document.createElement("div");
        div.style.textAlign = "center";
        div.style.fontSize = "0.8rem";
        div.style.color = "var(--cyan)";
        div.style.padding = "6px 0";
        div.innerHTML = `<em>${escapeHtml(text)}</em>`;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function clearChatHistory() {
        if (!confirm("Clear conversation history?")) return;
        state.chatHistory = [];
        chatMessages.innerHTML = `
            <div class="message-row assistant-row">
                <div class="avatar-assistant">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="12 2 2 7 12 12 22 7 12 2" />
                        <polyline points="2 17 12 22 22 17" />
                        <polyline points="2 12 12 17 22 12" />
                    </svg>
                </div>
                <div class="message-bubble assistant-bubble">
                    <div class="message-header">
                        <span class="sender-name">RAGNova Assistant</span>
                        <span class="grounding-badge">Context Cleared</span>
                    </div>
                    <div class="message-text">
                        <p>Conversation history has been reset. What would you like to explore next?</p>
                    </div>
                </div>
            </div>
        `;
        chatLatencyIndicator.textContent = "⚡ Grounded in uploaded context";
    }

    function openCitationDrawer(chunkId) {
        const citation = state.citationsCache[chunkId];
        if (!citation) return;

        drawerDocName.textContent = citation.filename;
        drawerPagePill.textContent = `Page ${citation.page_num}`;
        drawerChunkIdPill.textContent = citation.chunk_id;
        drawerScorePill.textContent = `Relevance: ${Math.round(citation.score * 100)}% (${citation.match_type})`;
        drawerFullText.textContent = citation.full_text || citation.snippet;

        citationDrawer.classList.add("open");
    }

    function updateSuggestedQuestions(questions) {
        promptsChips.innerHTML = questions.map(q => `
            <button class="prompt-chip" data-prompt="${escapeHtml(q)}">${escapeHtml(q)}</button>
        `).join("");
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // =========================================================================
    // Document List & NLP Analytics Views
    // =========================================================================
    function renderDocumentList() {
        docCounter.textContent = `${state.documents.length} Docs`;
        
        if (state.documents.length === 0) {
            emptyDocsState.style.display = "block";
            documentList.innerHTML = "";
            documentList.appendChild(emptyDocsState);
            return;
        }

        emptyDocsState.style.display = "none";
        documentList.innerHTML = "";

        state.documents.forEach(doc => {
            const ext = doc.metadata.extension ? doc.metadata.extension.replace(".", "") : "txt";
            const card = document.createElement("div");
            card.className = "doc-item-card";
            card.innerHTML = `
                <div class="doc-header-row">
                    <div class="doc-title-group">
                        <span class="doc-ext-icon ext-${ext}">${ext}</span>
                        <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
                    </div>
                    <div class="doc-actions">
                        <button class="icon-btn btn-view-nlp" title="Inspect Document NLP Analytics">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="16" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                        </button>
                        <button class="icon-btn btn-delete-doc" title="Remove Document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="doc-meta-row">
                    <span>${doc.metadata.word_count || 0} words</span>
                    <span>${doc.metadata.pages || 1} pg</span>
                    <span>${doc.analysis?.readability?.grade || "N/A"}</span>
                </div>
            `;

            // Inspect NLP button
            card.querySelector(".btn-view-nlp").addEventListener("click", () => {
                document.getElementById("tab-btn-analytics").click();
                analyticsDocSelect.value = doc.doc_id;
                renderAnalytics(doc.doc_id);
            });

            // Delete button
            card.querySelector(".btn-delete-doc").addEventListener("click", () => {
                deleteDocument(doc.doc_id, doc.filename);
            });

            documentList.appendChild(card);
        });
    }

    function updateDocSelector() {
        analyticsDocSelect.innerHTML = `<option value="">Select a document to inspect...</option>`;
        state.documents.forEach(doc => {
            const opt = document.createElement("option");
            opt.value = doc.doc_id;
            opt.textContent = `${doc.filename} (${doc.metadata.word_count} words)`;
            analyticsDocSelect.appendChild(opt);
        });

        if (state.documents.length > 0 && !state.currentDocId) {
            state.currentDocId = state.documents[0].doc_id;
            analyticsDocSelect.value = state.currentDocId;
            renderAnalytics(state.currentDocId);
        }
    }

    function renderAnalytics(docId) {
        if (!docId) {
            analyticsEmptyState.style.display = "block";
            analyticsContentView.style.display = "none";
            return;
        }

        const doc = state.documents.find(d => d.doc_id === docId);
        if (!doc || !doc.analysis) {
            analyticsEmptyState.style.display = "block";
            analyticsContentView.style.display = "none";
            return;
        }

        state.currentDocId = docId;
        analyticsEmptyState.style.display = "none";
        analyticsContentView.style.display = "flex";

        const analysis = doc.analysis;
        const stats = analysis.stats || {};
        const readability = analysis.readability || {};

        statWords.textContent = (stats.words || 0).toLocaleString();
        statDiversity.textContent = `${stats.lexical_diversity || 0}%`;
        statReadability.textContent = `${readability.score || 0} (${readability.grade || "Standard"})`;
        const readMins = Math.max(1, Math.round((stats.words || 0) / 200));
        statReadingTime.textContent = `${readMins} min`;

        nlpSummaryText.textContent = analysis.summary || "No summary available.";

        // Keywords Bar Chart
        const keywords = analysis.keywords || [];
        if (keywords.length === 0) {
            nlpKeywordsList.innerHTML = `<p class="empty-docs-state">No significant keyphrases extracted.</p>`;
        } else {
            nlpKeywordsList.innerHTML = keywords.map(k => `
                <div class="keyword-item">
                    <span class="keyword-name" title="${escapeHtml(k.phrase)}">${escapeHtml(k.phrase)}</span>
                    <div class="keyword-bar-track">
                        <div class="keyword-bar-fill" style="width: ${Math.round(k.weight * 100)}%;"></div>
                    </div>
                    <span class="keyword-count">${k.count}</span>
                </div>
            `).join("");
        }

        // Entities Lists
        const entities = analysis.entities || {};
        let entitiesHtml = "";

        if (entities.technologies && entities.technologies.length > 0) {
            entitiesHtml += `
                <div class="entity-group">
                    <div class="entity-group-title">Technical Terms & Acronyms</div>
                    <div class="entity-tags-row">
                        ${entities.technologies.map(t => `<span class="entity-tag tag-tech">${escapeHtml(t)}</span>`).join("")}
                    </div>
                </div>
            `;
        }

        if (entities.metrics && entities.metrics.length > 0) {
            entitiesHtml += `
                <div class="entity-group">
                    <div class="entity-group-title">Metrics & Quantitative Targets</div>
                    <div class="entity-tags-row">
                        ${entities.metrics.map(m => `<span class="entity-tag tag-metric">${escapeHtml(m)}</span>`).join("")}
                    </div>
                </div>
            `;
        }

        if (entities.organizations && entities.organizations.length > 0) {
            entitiesHtml += `
                <div class="entity-group">
                    <div class="entity-group-title">Organizations & Bodies</div>
                    <div class="entity-tags-row">
                        ${entities.organizations.map(o => `<span class="entity-tag tag-org">${escapeHtml(o)}</span>`).join("")}
                    </div>
                </div>
            `;
        }

        if (entities.dates_years && entities.dates_years.length > 0) {
            entitiesHtml += `
                <div class="entity-group">
                    <div class="entity-group-title">Temporal & Milestones</div>
                    <div class="entity-tags-row">
                        ${entities.dates_years.map(d => `<span class="entity-tag tag-date">${escapeHtml(d)}</span>`).join("")}
                    </div>
                </div>
            `;
        }

        if (!entitiesHtml) {
            entitiesHtml = `<p class="empty-docs-state">No named entities discovered in this text.</p>`;
        }

        nlpEntitiesList.innerHTML = entitiesHtml;
    }

    // =========================================================================
    // Chunk & Vector Explorer
    // =========================================================================
    async function renderAllChunks() {
        if (state.documents.length === 0) {
            chunksList.innerHTML = `<div class="empty-panel-state"><p>No chunks indexed. Upload documents first.</p></div>`;
            return;
        }

        chunksList.innerHTML = `<div class="empty-panel-state"><p>Loading indexed chunks...</p></div>`;
        let allChunks = [];

        for (const doc of state.documents) {
            try {
                const res = await fetch(`/api/documents/${doc.doc_id}/chunks`);
                const data = await res.json();
                if (data.chunks) {
                    allChunks = allChunks.concat(data.chunks);
                }
            } catch (err) {
                console.error(err);
            }
        }

        state.chunks = allChunks;
        footerChunkCount.textContent = allChunks.length;

        if (allChunks.length === 0) {
            chunksList.innerHTML = `<div class="empty-panel-state"><p>No chunks generated.</p></div>`;
            return;
        }

        chunksList.innerHTML = allChunks.slice(0, 30).map((ch, idx) => `
            <div class="chunk-card">
                <div class="chunk-header-row">
                    <span class="chunk-id-tag">${escapeHtml(ch.chunk_id)} | ${escapeHtml(ch.filename)}</span>
                    <div class="chunk-metrics">
                        <span>Page ${ch.page_num}</span>
                        <span>${ch.char_count} chars</span>
                    </div>
                </div>
                <div class="chunk-text-body">${escapeHtml(ch.text)}</div>
            </div>
        `).join("");
    }

    async function handleSimulateSearch() {
        const query = simQueryInput.value.trim();
        if (!query) return;

        btnSimSearch.disabled = true;
        btnSimSearch.textContent = "Simulating...";

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });
            const data = await res.json();

            if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
                chunksList.innerHTML = data.retrieved_chunks.map((ch, idx) => `
                    <div class="chunk-card" style="border-color: var(--border-glow);">
                        <div class="chunk-header-row">
                            <span class="chunk-id-tag">Rank #${idx + 1}: ${escapeHtml(ch.chunk_id)} (${escapeHtml(ch.filename)})</span>
                            <div class="chunk-metrics">
                                <span class="rank-badge">Fused RRF: ${(ch.fused_score * 100).toFixed(1)}%</span>
                                <span>BM25: ${(ch.bm25_score * 100).toFixed(1)}%</span>
                                <span>Vector: ${(ch.vector_score * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="chunk-text-body">${escapeHtml(ch.text)}</div>
                    </div>
                `).join("");
            } else {
                chunksList.innerHTML = `<div class="empty-panel-state"><p>No matching chunks above threshold for query: "${escapeHtml(query)}"</p></div>`;
            }
        } catch (err) {
            console.error("Simulation error:", err);
        } finally {
            btnSimSearch.disabled = false;
            btnSimSearch.textContent = "Simulate Retrieval";
        }
    }

    // =========================================================================
    // Settings Handling
    // =========================================================================
    function syncSettingsToUI() {
        const s = state.settings;
        rangeAlpha.value = s.alpha !== undefined ? s.alpha : 0.5;
        valAlpha.textContent = `${parseFloat(rangeAlpha.value).toFixed(2)}`;
        rangeChunkSize.value = s.chunk_size || 600;
        valChunkSize.textContent = `${s.chunk_size || 600} chars`;
        rangeChunkOverlap.value = s.chunk_overlap || 120;
        valChunkOverlap.textContent = `${s.chunk_overlap || 120} chars`;
        rangeTopK.value = s.top_k || 4;
        valTopK.textContent = `${s.top_k || 4} chunks`;
        selectRetrievalMode.value = s.retrieval_mode || "hybrid";
        selectGeminiModel.value = s.gemini_model || "gemini-2.5-flash";
    }

    async function handleSettingsSubmit(e) {
        e.preventDefault();
        const payload = {
            chunk_size: parseInt(rangeChunkSize.value),
            chunk_overlap: parseInt(rangeChunkOverlap.value),
            top_k: parseInt(rangeTopK.value),
            retrieval_mode: selectRetrievalMode.value,
            alpha: parseFloat(rangeAlpha.value),
            gemini_model: selectGeminiModel.value
        };

        const apiKey = inputApiKey.value.trim();
        if (apiKey) {
            payload.gemini_api_key = apiKey;
        }

        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            state.settings = data.settings;
            settingsModal.style.display = "none";
            updateHeaderAndFooterBadges();
            await fetchDocuments();
            alert("Settings applied and indices re-synchronized!");
        } catch (err) {
            console.error("Error updating settings:", err);
        }
    }

    function updateHeaderAndFooterBadges() {
        const s = state.settings;
        footerChunkCount.textContent = s.total_chunks || 0;
        footerRetrievalMode.textContent = s.retrieval_mode ? s.retrieval_mode.toUpperCase() : "HYBRID";
        if (s.has_api_key) {
            headerProviderBadge.textContent = "Google Gemini";
            headerProviderBadge.style.color = "var(--emerald)";
        } else {
            headerProviderBadge.textContent = "Local NLP";
            headerProviderBadge.style.color = "var(--cyan)";
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================
    function escapeHtml(text) {
        if (!text) return "";
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function parseMarkdownToHtml(markdown) {
        if (!markdown) return "";
        let html = markdown;

        // Escape raw HTML tags
        html = escapeHtml(html);

        // Headers
        html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

        // Bold & Italic
        html = html.replace(/\*\*\*(.*?)\*\*\*/gim, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

        // Blockquotes
        html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

        // Inline Code
        html = html.replace(/`(.*?)`/gim, '<code>$1</code>');

        // Horizontal rules
        html = html.replace(/^---$/gim, '<hr>');

        // Unordered lists
        html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>');

        // Paragraphs
        html = html.split(/\n\n+/).map(p => {
            if (p.startsWith('<h') || p.startsWith('<ul>') || p.startsWith('<blockquote') || p.startsWith('<hr')) {
                return p;
            }
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');

        return html;
    }
});
