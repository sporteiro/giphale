const providerSelect = document.getElementById('provider');
const modelInput = document.getElementById('model');
const useRagCheckbox = document.getElementById('useRag');
const ragSection = document.getElementById('ragSection');
const ragSourceSelect = document.getElementById('ragSource');
const themeToggle = document.getElementById('themeToggle');

// Theme toggle functionality
themeToggle.addEventListener('click', function() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    themeToggle.textContent = newTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
});

// Load saved theme preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeToggle.textContent = savedTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
}

const defaultModels = {
    'openrouter': 'meta-llama/llama-3-8b-instruct:free',
    'local': 'qwen2.5-coder:7b',
    'groq': 'llama3-70b-8192',
    'huggingface': 'google/flan-t5-base'
};

providerSelect.addEventListener('change', function() {
    const selectedProvider = this.value;
    if (defaultModels[selectedProvider]) {
        modelInput.value = defaultModels[selectedProvider];
    } else {
        modelInput.value = '';
    }
});

useRagCheckbox.addEventListener('change', async function() {
    if (this.checked) {
        ragSection.style.display = 'block';
        await loadRagSources();
    } else {
        ragSection.style.display = 'none';
        ragSourceSelect.value = '';
    }
});

async function loadRagSources() {
    try {
        const res = await fetch('/rag_sources');
        const data = await res.json();

        ragSourceSelect.innerHTML = '<option value="">Select a source...</option>';

        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const option = document.createElement('option');
                option.value = source;
                option.textContent = source;
                ragSourceSelect.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No sources available';
            ragSourceSelect.appendChild(option);
        }
    } catch (err) {
        console.error('Error loading RAG sources:', err);
        ragSourceSelect.innerHTML = '<option value="">Error loading sources</option>';
    }
}

document.getElementById('aiForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const prompt = document.getElementById('prompt').value;
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value || null;
    const useRag = useRagCheckbox.checked;
    const ragSource = ragSourceSelect.value || null;

    const payload = { prompt };
    if (provider) payload.provider = provider;
    if (model) payload.model = model;
    if (useRag && ragSource) {
        payload.use_rag = true;
        payload.rag_source = ragSource;
    }

    const responseDiv = document.getElementById('response');
    responseDiv.textContent = 'Cargando...';

    try {
        const res = await fetch('/ask_ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) {
            responseDiv.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
        } else {
            responseDiv.textContent = data.response || 'No response content';
        }
    } catch (err) {
        responseDiv.innerHTML = `<span style="color:red;">Network error: ${err.message}</span>`;
    }
});
