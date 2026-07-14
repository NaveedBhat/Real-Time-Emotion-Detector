// app.js

const emotionColors = {
    happy: '#00d25a',
    neutral: '#969696',
    surprise: '#00c8e6',
    sad: '#1450c8',
    angry: '#dc1e1e',
    fear: '#be28be',
    disgust: '#50aa28'
};

const emotionOrder = ['happy', 'neutral', 'surprise', 'sad', 'angry', 'fear', 'disgust'];

let ws;
let emotionChart;

// --- Chart Setup ---
function initChart() {
    const ctx = document.getElementById('emotionChart').getContext('2d');

    const datasets = emotionOrder.map(emotion => ({
        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
        data: [],
        borderColor: emotionColors[emotion],
        backgroundColor: emotionColors[emotion] + '2A', // ~16% opacity, filled area
        borderWidth: 2,
        tension: 0.4,
        fill: 'origin',
        pointRadius: 0,
        pointHitRadius: 10,
    }));

    emotionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 }, // turn off animation for live updates
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#6b7789', font: { family: "'JetBrains Mono', monospace", size: 11 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#6b7789', maxTicksLimit: 10, font: { family: "'JetBrains Mono', monospace", size: 11 } }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Build custom legend
    const legendEl = document.getElementById('emotion-legend');
    emotionOrder.forEach(emotion => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-dot" style="background:${emotionColors[emotion]}"></span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`;
        legendEl.appendChild(item);
    });
}

// --- WebSocket Setup ---
function connectWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

    ws.onopen = () => {
        document.querySelector('.indicator-dot').classList.add('active');
        document.getElementById('ws-status').innerText = 'Connected';
    };

    ws.onclose = () => {
        document.querySelector('.indicator-dot').classList.remove('active');
        document.getElementById('ws-status').innerText = 'Disconnected';
        setTimeout(connectWebSocket, 2000); // Reconnect loop
    };

    // Network errors don't trigger onclose — forward them so the reconnect fires
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // 1. Update Chart
        if (data.chart) {
            updateChart(data.chart);
        }

        // 2. Update Stats
        if (data.stats) {
            updateStats(data.stats);
        }

        // 3. Update HUD & Faces
        if (data.hud) {
            document.getElementById('fps-val').innerText = data.hud.fps.toFixed(1);
            document.getElementById('face-count-badge').innerText =
                `${data.hud.n_faces} Face${data.hud.n_faces !== 1 ? 's' : ''}`;
            updateFaces(data.hud.faces);
        }
    };
}

// --- Updaters ---
function updateChart(chartData) {
    if (!emotionChart) return;

    emotionChart.data.labels = chartData.labels;

    emotionOrder.forEach((emotion, index) => {
        emotionChart.data.datasets[index].data = chartData.datasets[emotion];
    });

    emotionChart.update();
}

function updateStats(stats) {
    document.getElementById('stat-dominant').innerText = stats.dominant || '--';
    document.getElementById('stat-dominant').style.color = stats.dominant ? emotionColors[stats.dominant] : '#fff';

    document.getElementById('stat-confidence').innerText = stats.avg_confidence ? `${stats.avg_confidence}%` : '--%';
    document.getElementById('stat-records').innerText = (stats.total_records ?? 0).toLocaleString();
}

function updateFaces(faces) {
    const container = document.getElementById('faces-container');
    const msg = document.getElementById('no-faces-msg');

    if (!faces || faces.length === 0) {
        msg.style.display = 'block';
        Array.from(container.children).forEach(child => {
            if (child.id !== 'no-faces-msg') child.remove();
        });
        return;
    }

    msg.style.display = 'none';

    // We want to reuse existing cards if possible to avoid flicker
    const existingIds = Array.from(container.children)
        .filter(c => c.id !== 'no-faces-msg')
        .map(c => c.dataset.faceId);

    const newIds = faces.map(f => f.id.toString());

    // Remove old cards
    existingIds.forEach(id => {
        if (!newIds.includes(id)) {
            container.querySelector(`[data-face-id="${id}"]`).remove();
        }
    });

    // Update or create cards
    faces.forEach(face => {
        let card = container.querySelector(`[data-face-id="${face.id}"]`);

        const confStr = face.confidence ? (face.confidence * 100).toFixed(0) : 0;
        const color = face.label ? emotionColors[face.label] : '#969696';
        const labelText = face.label || 'Detecting...';

        if (!card) {
            // Create
            card = document.createElement('div');
            card.className = 'face-card';
            card.dataset.faceId = face.id;

            card.innerHTML = `
                <div class="face-card-header">
                    <span>Face <span class="face-id">#${face.id}</span></span>
                    <span class="face-conf">${confStr}%</span>
                </div>
                <div class="face-emotion" style="color: ${color}">${labelText}</div>
                <div class="face-prob-bar">
                    <div class="face-prob-fill" style="width: ${confStr}%; background-color: ${color}"></div>
                </div>
            `;
            card.style.borderLeftColor = color;
            container.appendChild(card);
        } else {
            // Update
            card.querySelector('.face-conf').innerText = `${confStr}%`;

            const elEmotion = card.querySelector('.face-emotion');
            elEmotion.innerText = labelText;
            elEmotion.style.color = color;

            const elFill = card.querySelector('.face-prob-fill');
            elFill.style.width = `${confStr}%`;
            elFill.style.backgroundColor = color;

            card.style.borderLeftColor = color;
        }
    });
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    connectWebSocket();

    // Export CSV
    document.getElementById('btn-export').addEventListener('click', async () => {
        const resp = await fetch('/export_csv');
        if (resp.ok) {
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'session_export.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            alert('Export failed.');
        }
    });
});