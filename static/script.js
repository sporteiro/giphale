document.getElementById('aiForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const prompt = document.getElementById('prompt').value;
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value || null;

    const payload = { prompt };
    if (provider) payload.provider = provider;
    if (model) payload.model = model;

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