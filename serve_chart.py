import os
import glob
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from tensorboard.backend.event_processing import event_accumulator

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Bidding Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, sans-serif; background: #1e1e1e; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .dashboard { display: flex; width: 90vw; max-width: 1200px; justify-content: space-around; align-items: center; margin-top: 20px; }
        .chart-container { width: 45%; }
        h1 { font-weight: 300; margin-bottom: 10px; }
        .totals-banner { font-size: 1.1rem; color: #ccc; margin-bottom: 20px; padding: 15px; background: #2a2a2a; border-radius: 8px; width: 90vw; max-width: 1000px; display: flex; justify-content: space-around; }
        h2 { text-align: center; font-weight: 300; font-size: 1.2rem; margin-bottom: 15px; color: #aaa; }
    </style>
</head>
<body>
    <h1>Live Agent Dashboard</h1>
    <div class="totals-banner" id="totalsBanner">
        <span>Passz: 0</span>
        <span>Ulti: 0</span>
        <span>Betli: 0</span>
        <span>Durchmars: 0</span>
        <span style="color: white; font-weight: bold;">Total: 0</span>
    </div>
    <div class="dashboard">
        <div class="chart-container">
            <h2>Bidding Distribution</h2>
            <canvas id="pieChart"></canvas>
        </div>
        <div class="chart-container">
            <h2>Win Rates</h2>
            <canvas id="barChart"></canvas>
        </div>
    </div>

    <script>
        const ctxPie = document.getElementById('pieChart').getContext('2d');
        const bidNames = ['Passz', 'Piros passz', '40-100', 'Piros 40-100', 'Ulti', 'Piros ulti', 'Betli', 'Piros betli', 'Durchmars', 'Piros durchmars'];
        const bgColors = ['#cccccc', '#ff6666', '#ffb366', '#ff8000', '#ff9999', '#cc0000', '#66b3ff', '#0055ff', '#99ff99', '#00cc00'];

        const pieChart = new Chart(ctxPie, {
            type: 'pie',
            data: {
                labels: bidNames,
                datasets: [{
                    data: new Array(10).fill(0),
                    backgroundColor: bgColors,
                    borderWidth: 1,
                    borderColor: '#111'
                }]
            },
            options: {
                responsive: true,
                animation: { duration: 500 },
                plugins: {
                    legend: { position: 'right', labels: { color: 'white', font: { size: 14 } } },
                    tooltip: { callbacks: { label: function(c) { return c.label + ': ' + c.raw.toFixed(1) + '%'; } } }
                }
            }
        });

        const ctxBar = document.getElementById('barChart').getContext('2d');
        const barChart = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: bidNames,
                datasets: [{
                    label: 'Win Rate %',
                    data: new Array(10).fill(0),
                    backgroundColor: bgColors,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                animation: { duration: 500 },
                scales: {
                    y: { beginAtZero: true, max: 100, ticks: { color: 'white' }, grid: { color: '#333' } },
                    x: { ticks: { color: 'white', font: { size: 12 } }, grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: function(c) { return c.raw.toFixed(1) + '% Win Rate'; } } }
                }
            }
        });

        function updateData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    const p = data.percentages || {};
                    const w = data.win_rates || {};
                    const t = data.totals || {};
                    
                    pieChart.data.datasets[0].data = bidNames.map(name => p[name] || 0);
                    pieChart.update();
                    
                    barChart.data.datasets[0].data = bidNames.map(name => w[name] || 0);
                    barChart.update();
                    
                    let totalsHtml = '';
                    let totalGames = 0;
                    bidNames.forEach(name => {
                        const count = t[name] || 0;
                        totalGames += count;
                        totalsHtml += `<span>${name}: ${count.toLocaleString()}</span>`;
                    });
                    
                    document.getElementById('totalsBanner').innerHTML = totalsHtml + `<span><b>Total: ${totalGames.toLocaleString()}</b></span>`;
                })
                .catch(err => console.error(err));
        }

        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
"""

def get_latest_percentages():
    try:
        with open('C:/ulti_ai/logs/bidding_percentages.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {}

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = get_latest_percentages()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass # Suppress logging

if __name__ == '__main__':
    server = HTTPServer(('localhost', 5000), ChartHandler)
    print("Serving live chart at http://localhost:5000")
    server.serve_forever()
