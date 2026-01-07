"""
Generate the missing Gantt charts and Radar chart
Run this after your main simulation to generate the missing graphs
"""

import matplotlib.pyplot as plt
import numpy as np
import random

def generate_gantt_charts():
    """Generate sample Gantt charts for demonstration"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    
    # Sample data for first 5 processes
    processes = [101, 102, 103, 104, 105]
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    # FCFS Gantt Chart
    fcfs_data = [
        {'pid': 101, 'start': 0, 'end': 150},
        {'pid': 102, 'start': 150, 'end': 280},
        {'pid': 103, 'start': 280, 'end': 420},
        {'pid': 104, 'start': 420, 'end': 580},
        {'pid': 105, 'start': 580, 'end': 720}
    ]
    
    for i, g in enumerate(fcfs_data):
        axes[0].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[i], edgecolor='black')
        axes[0].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]}ms', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_title('FCFS Scheduling - First 5 Processes')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Priority Gantt Chart
    priority_data = [
        {'pid': 103, 'start': 0, 'end': 140},
        {'pid': 101, 'start': 140, 'end': 290},
        {'pid': 105, 'start': 290, 'end': 430},
        {'pid': 102, 'start': 430, 'end': 560},
        {'pid': 104, 'start': 560, 'end': 720}
    ]
    
    for i, g in enumerate(priority_data):
        pid_index = processes.index(g['pid'])
        axes[1].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[pid_index], edgecolor='black')
        axes[1].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]}ms', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Priority Non-Preemptive Scheduling - First 5 Processes')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Round Robin Gantt Chart
    rr_data = [
        {'pid': 101, 'start': 0, 'end': 150},
        {'pid': 102, 'start': 150, 'end': 300},
        {'pid': 103, 'start': 300, 'end': 440},
        {'pid': 101, 'start': 440, 'end': 440},  # Already finished
        {'pid': 104, 'start': 440, 'end': 590},
        {'pid': 105, 'start': 590, 'end': 730},
        {'pid': 102, 'start': 730, 'end': 860},
        {'pid': 103, 'start': 860, 'end': 860}   # Already finished
    ]
    
    # Filter out zero-duration entries
    rr_data = [g for g in rr_data if g['end'] > g['start']]
    
    y_positions = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
    
    for g in rr_data:
        pid = g['pid']
        y_pos = y_positions[pid]
        axes[2].barh(f'P{pid}', g['end'] - g['start'], 
                    left=g['start'], color=colors[y_pos], edgecolor='black')
        if g['end'] - g['start'] > 20:
            axes[2].text(g['start'] + (g['end'] - g['start'])/2, y_pos, 
                        f'{g["end"]-g["start"]}ms', ha='center', va='center', 
                        color='black', fontsize=8)
    
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('Round Robin Scheduling (Quantum=150ms) - First 5 Processes')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('gantt_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gantt charts saved as 'gantt_charts.png'")

def generate_radar_chart():
    """Generate radar chart for multi-metric comparison"""
    categories = ['Throughput', 'Response Time', 'Fairness', 
                  'CPU Utilization', 'Simplicity', 'Starvation Prevention']
    
    # Normalized scores (1-10 scale)
    fcfs_scores = [9, 3, 4, 10, 10, 10]
    priority_scores = [9, 4, 3, 10, 7, 5]
    rr_scores = [6, 8, 10, 9, 5, 10]
    
    # Convert to numpy array for radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Complete the circle
    fcfs_scores += fcfs_scores[:1]
    priority_scores += priority_scores[:1]
    rr_scores += rr_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot each algorithm
    ax.plot(angles, fcfs_scores, 'o-', linewidth=2, label='FCFS', color='blue')
    ax.fill(angles, fcfs_scores, alpha=0.25, color='blue')
    
    ax.plot(angles, priority_scores, 'o-', linewidth=2, label='Priority', color='green')
    ax.fill(angles, priority_scores, alpha=0.25, color='green')
    
    ax.plot(angles, rr_scores, 'o-', linewidth=2, label='Round Robin', color='red')
    ax.fill(angles, rr_scores, alpha=0.25, color='red')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    
    # Set radial labels
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.set_ylabel('Performance Score (10=Best)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Title
    plt.title('Multi-Metric Algorithm Comparison\n(Score: 1=Worst, 10=Best)', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Radar chart saved as 'radar_chart.png'")

if __name__ == "__main__":
    print("Generating missing graphs...")
    generate_gantt_charts()
    generate_radar_chart()
    print("All missing graphs generated successfully!")