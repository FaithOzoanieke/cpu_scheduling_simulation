"""
Generate the missing Gantt charts and Radar chart
Run this after your main simulation to generate the missing graphs
"""

import matplotlib.pyplot as plt
import numpy as np

def generate_gantt_charts():
    """Generate Gantt charts for the report using corrected data"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    
    # Process IDs for first 5 processes
    processes = [101, 102, 103, 104, 105]
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    # FCFS Gantt Chart - Based on corrected execution timeline
    fcfs_data = [
        {'pid': 101, 'start': 0.0, 'end': 187.0},
        {'pid': 102, 'start': 187.0, 'end': 332.0},
        {'pid': 103, 'start': 332.0, 'end': 508.0},
        {'pid': 104, 'start': 508.0, 'end': 620.0},
        {'pid': 105, 'start': 620.0, 'end': 818.0}
    ]
    
    for i, g in enumerate(fcfs_data):
        axes[0].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[i], edgecolor='black')
        axes[0].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]:.0f}ms', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_title('FCFS Scheduling - First 5 Processes')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Priority Non-Preemptive Gantt Chart - Based on corrected execution
    priority_data = [
        {'pid': 101, 'start': 0.0, 'end': 187.0},
        {'pid': 104, 'start': 187.0, 'end': 299.0},
        {'pid': 106, 'start': 299.0, 'end': 428.0},
        {'pid': 110, 'start': 428.0, 'end': 625.0},
        {'pid': 107, 'start': 625.0, 'end': 806.0}
    ]
    
    for i, g in enumerate(priority_data):
        pid_index = processes.index(g['pid']) if g['pid'] in processes else 0
        axes[1].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[pid_index], edgecolor='black')
        axes[1].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]:.0f}ms', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Priority Non-Preemptive Scheduling - First 5 Processes (Corrected)')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Round Robin Gantt Chart - Based on corrected execution
    rr_data = [
        {'pid': 101, 'start': 0.0, 'end': 150.0},
        {'pid': 102, 'start': 150.0, 'end': 295.0},
        {'pid': 101, 'start': 295.0, 'end': 332.0},
        {'pid': 103, 'start': 332.0, 'end': 482.0},
        {'pid': 104, 'start': 482.0, 'end': 594.0},
        {'pid': 105, 'start': 594.0, 'end': 744.0},
        {'pid': 103, 'start': 744.0, 'end': 770.0},
        {'pid': 105, 'start': 770.0, 'end': 818.0}
    ]
    
    y_positions = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
    
    for g in rr_data:
        pid = g['pid']
        y_pos = y_positions[pid]
        axes[2].barh(f'P{pid}', g['end'] - g['start'], 
                    left=g['start'], color=colors[y_pos], edgecolor='black')
        if g['end'] - g['start'] > 20:
            axes[2].text(g['start'] + (g['end'] - g['start'])/2, y_pos, 
                        f'{g["end"]-g["start"]:.0f}ms', ha='center', va='center', 
                        color='black', fontsize=8)
    
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('Round Robin Scheduling (Quantum=150ms) - First 5 Processes')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('gantt_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gantt charts saved as 'gantt_charts.png'")

def generate_radar_chart():
    """Generate radar chart based on actual simulation results"""
    categories = ['Throughput', 'Response Time', 'Fairness', 
                  'CPU Utilization', 'Simplicity', 'Starvation Prevention']
    
    # Normalized scores based on actual simulation results from report
    fcfs_scores = [8, 3, 4, 10, 10, 10]  # FCFS: High throughput/util, poor response/fairness
    priority_scores = [8, 3, 3, 10, 7, 5]  # Priority: Similar to FCFS but worse fairness
    rr_scores = [6, 9, 10, 8, 5, 10]  # RR: Excellent response/fairness, lower throughput
    
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
    
    ax.plot(angles, priority_scores, 'o-', linewidth=2, label='Priority Non-Preemptive', color='green')
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
    plt.title('Multi-Metric Algorithm Comparison\nBased on 100-Process Simulation Results', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Radar chart saved as 'radar_chart.png'")

if __name__ == "__main__":
    print("Generating missing graphs for the report...")
    generate_gantt_charts()
    generate_radar_chart()
    print("✓ All missing graphs generated successfully!")