import os
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorboard.backend.event_processing import event_accumulator

def get_latest_percentages():
    tb_dirs = glob.glob('C:/ulti_ai/logs/tb/bidding_phase_*')
    if not tb_dirs:
        return None
        
    latest_dir = max(tb_dirs, key=os.path.getctime)
    
    # We only care about the most recent scalar value
    ea = event_accumulator.EventAccumulator(latest_dir, size_guidance={'scalars': 1})
    ea.Reload()
    
    modes = ["Normal", "Ulti", "Betli", "Durchmars"]
    percentages = []
    labels = []
    colors = ['#cccccc', '#ff9999', '#66b3ff', '#99ff99']
    
    for i, m in enumerate(modes):
        key = f"Metrics/Percentage_{m}"
        if key in ea.Tags().get('scalars', []):
            events = ea.Scalars(key)
            if events:
                val = events[-1].value
                if val > 0:
                    percentages.append(val)
                    # For Normal, label it "Passz"
                    display_name = "Passz" if m == "Normal" else m
                    labels.append(f"{display_name}\n({val:.1f}%)")
                    
    # Ensure colors match the labels
    final_colors = []
    for lbl in labels:
        if "Passz" in lbl: final_colors.append('#cccccc')
        elif "Ulti" in lbl: final_colors.append('#ff9999')
        elif "Betli" in lbl: final_colors.append('#66b3ff')
        elif "Durchmars" in lbl: final_colors.append('#99ff99')
        
    if not percentages:
        # If no percentages logged yet, return default
        return [100], ["Waiting for data..."], ['#ffffff']
        
    return percentages, labels, final_colors

# Setup the figure
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title('Live Bidding Distribution')

def animate(i):
    data = get_latest_percentages()
    if data:
        percentages, labels, colors = data
        ax.clear()
        
        # Don't plot if empty
        if percentages:
            ax.pie(percentages, labels=labels, colors=colors, autopct='', 
                   startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title(f'Live Bidding Distribution\n(Updates every 2s)', fontsize=14, pad=20)

# Update every 2000 milliseconds (2 seconds)
ani = animation.FuncAnimation(fig, animate, interval=2000, cache_frame_data=False)

plt.show()
