import os
import glob
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from tensorboard.backend.event_processing import event_accumulator

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Bidding Distribution</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, sans-serif; background: #1e1e1e; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .chart-container { width: 50vw; max-width: 600px; }
        h1 { font-weight: 300; margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Live Bidding Distribution</h1>
    <div class="chart-container">
        <canvas id="myChart"></canvas>
    </div>

    <script>
        const ctx = document.getElementById('myChart').getContext('2d');
        const myChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Passz', 'Ulti', 'Betli', 'Durchmars'],
                datasets: [{
                    data: [1, 1, 1, 1], // Initial dummy data
                    backgroundColor: ['#cccccc', '#ff9999', '#66b3ff', '#99ff99'],
                    borderColor: '#1e1e1e',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                animation: { duration: 500 },
                plugins: {
                    legend: { position: 'bottom', labels: { color: 'white', font: { size: 16 } } },
                    tooltip: { callbacks: { label: function(context) { return context.label + ': ' + context.raw.toFixed(1) + '%'; } } }
                }
            }
        });

        function updateData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    myChart.data.datasets[0].data = [
                        data.Normal || 0,
                        data.Ulti || 0,
                        data.Betli || 0,
                        data.Durchmars || 0
                    ];
                    myChart.update();
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
    tb_dirs = glob.glob('C:/ulti_ai/logs/tb/bidding_phase_*')
    if not tb_dirs: return {}
    latest_dir = max(tb_dirs, key=os.path.getctime)
    
    ea = event_accumulator.EventAccumulator(latest_dir, size_guidance={'scalars': 1})
    ea.Reload()
    
    modes = ["Normal", "Ulti", "Betli", "Durchmars"]
    data = {}
    for m in modes:
        key = f"Metrics/Percentage_{m}"
        if key in ea.Tags().get('scalars', []):
            events = ea.Scalars(key)
            if events:
                data[m] = events[-1].value
    return data

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
