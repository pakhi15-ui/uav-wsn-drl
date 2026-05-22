from flask import Flask, render_template_string, jsonify
from env.multi_uav_env import MultiUAVEnv
from stable_baselines3 import PPO
import numpy as np
import threading
import time

app = Flask(__name__)

# Global state
state = {
    "uav_positions": [[50, 50], [50, 50]],
    "sensors": [],
    "obstacles": [],
    "step": 0,
    "reward": 0,
    "visited": 0,
    "dead": 0,
    "running": False
}

env = MultiUAVEnv(n_sensors=10, grid_size=100, max_steps=200)

try:
    model = PPO.load("models/trained_uav_agent_v2")
    print("Loaded trained multi-UAV model")
except:
    model = None
    print("No trained model found — using random policy")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>UAV WSN Dashboard</title>
    <style>
        body { font-family: Arial; background: #0a0a1a; color: #fff; margin: 0; padding: 20px; }
        h1 { color: #4fc3f7; text-align: center; }
        .container { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
        canvas { background: #111; border: 2px solid #4fc3f7; border-radius: 8px; }
        .stats { background: #1a1a2e; border-radius: 8px; padding: 20px; min-width: 200px; }
        .stat { margin: 10px 0; font-size: 18px; }
        .val { color: #4fc3f7; font-weight: bold; font-size: 24px; }
        button { background: #4fc3f7; color: #000; border: none; padding: 12px 28px;
                 font-size: 16px; border-radius: 6px; cursor: pointer; margin: 5px; }
        button:hover { background: #81d4fa; }
        .legend { display: flex; gap: 20px; justify-content: center; margin: 10px 0; }
        .dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    </style>
</head>
<body>
    <h1>UAV-WSN Deep Reinforcement Learning — Live Dashboard</h1>
    <div class="legend">
        <span><span class="dot" style="background:#ef5350"></span>Sensors</span>
        <span><span class="dot" style="background:#42a5f5"></span>UAV 1</span>
        <span><span class="dot" style="background:#ab47bc"></span>UAV 2</span>
        <span><span class="dot" style="background:#ff7043"></span>Obstacles</span>
        <span><span class="dot" style="background:#66bb6a"></span>Visited</span>
    </div>
    <div class="container">
        <canvas id="canvas" width="500" height="500"></canvas>
        <div class="stats">
            <div class="stat">Step<br><span class="val" id="step">0</span></div>
            <div class="stat">Reward<br><span class="val" id="reward">0</span></div>
            <div class="stat">Sensors Visited<br><span class="val" id="visited">0/10</span></div>
            <div class="stat">Dead Sensors<br><span class="val" id="dead">0</span></div>
            <br>
            <button onclick="startSim()">▶ Start</button>
            <button onclick="stopSim()">⏹ Stop</button>
            <button onclick="resetSim()">↺ Reset</button>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const scale = 5;
        let interval = null;

        function draw(data) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Grid
            ctx.strokeStyle = '#1a1a2e';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 100; i += 10) {
                ctx.beginPath(); ctx.moveTo(i*scale, 0); ctx.lineTo(i*scale, 500); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, i*scale); ctx.lineTo(500, i*scale); ctx.stroke();
            }

            // Sensors
            data.sensors.forEach((s, i) => {
                ctx.beginPath();
                ctx.arc(s.x * scale, s.y * scale, 8, 0, Math.PI*2);
                const isDead = s.energy <= 0;
                const isVisited = s.visited;
                ctx.fillStyle = isDead ? '#555' : isVisited ? '#66bb6a' : '#ef5350';
                ctx.fill();
                ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();

                // Energy bar
                const barW = 20, barH = 4;
                ctx.fillStyle = '#333';
                ctx.fillRect(s.x*scale - barW/2, s.y*scale + 10, barW, barH);
                ctx.fillStyle = s.energy > 0.5 ? '#66bb6a' : s.energy > 0.2 ? '#ffa726' : '#ef5350';
                ctx.fillRect(s.x*scale - barW/2, s.y*scale + 10, barW * s.energy, barH);
            });

            // UAVs
            const uavColors = ['#42a5f5', '#ab47bc'];
            data.uav_positions.forEach((pos, i) => {
                ctx.beginPath();
                ctx.arc(pos[0]*scale, pos[1]*scale, 10, 0, Math.PI*2);
                ctx.fillStyle = uavColors[i];
                ctx.fill();
                ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
                ctx.fillStyle = '#fff'; ctx.font = 'bold 10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('U'+(i+1), pos[0]*scale, pos[1]*scale+4);

                // Collection radius
                ctx.beginPath();
                ctx.arc(pos[0]*scale, pos[1]*scale, 20*scale/10, 0, Math.PI*2);
                ctx.strokeStyle = uavColors[i] + '44';
                ctx.lineWidth = 1; ctx.stroke();
            });

            // Stats
            document.getElementById('step').textContent = data.step;
            document.getElementById('reward').textContent = Math.round(data.reward);
            document.getElementById('visited').textContent = data.visited + '/10';
            document.getElementById('dead').textContent = data.dead;
        }

        function fetchAndDraw() {
            fetch('/state').then(r => r.json()).then(draw);
        }

        function startSim() {
            fetch('/start');
            if (!interval) interval = setInterval(fetchAndDraw, 100);
        }

        function stopSim() {
            fetch('/stop');
            clearInterval(interval); interval = null;
        }

        function resetSim() {
            fetch('/reset');
            fetchAndDraw();
        }

        fetchAndDraw();
    </script>
</body>
</html>
"""

sim_running = False

def run_simulation():
    global state, sim_running
    obs, _ = env.reset()
    total_reward = 0
    visited_set = set()

    while sim_running and state["step"] < 200:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        visited_set = env.visited_sensors

        sensor_data = []
        for s in env.sensors:
            sensor_data.append({
                "x": s.x, "y": s.y,
                "energy": s.energy / s.max_energy,
                "visited": s.node_id in visited_set
            })

        state.update({
            "uav_positions": [list(p) for p in env.uav_positions],
            "sensors": sensor_data,
            "step": env.current_step,
            "reward": round(total_reward, 1),
            "visited": len(visited_set),
            "dead": len(env.dead_sensors)
        })
        time.sleep(0.05)
        if done:
            break
    sim_running = False

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/start')
def start():
    global sim_running
    if not sim_running:
        sim_running = True
        t = threading.Thread(target=run_simulation, daemon=True)
        t.start()
    return 'ok'

@app.route('/stop')
def stop():
    global sim_running
    sim_running = False
    return 'ok'

@app.route('/reset')
def reset():
    global state, sim_running
    sim_running = False
    obs, _ = env.reset()
    state = {
        "uav_positions": [list(p) for p in env.uav_positions],
        "sensors": [{"x": s.x, "y": s.y, "energy": 1.0, "visited": False}
                    for s in env.sensors],
        "step": 0, "reward": 0, "visited": 0, "dead": 0
    }
    return 'ok'

if __name__ == '__main__':
    print("\n" + "="*50)
    print("UAV WSN Dashboard running!")
    print("Open your browser at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
