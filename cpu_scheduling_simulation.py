"""
CPU Scheduling Algorithms Simulation with Automatic Graph Generation
Author: [Your Name]
Course: CSC 805 - Advanced Operating Systems
Run this script to generate all required graphs automatically
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# ==================== 1. PROCESS GENERATION ====================
def generate_processes(num_processes=100, lambda_rate=150):
    """Generate processes with Poisson arrivals"""
    processes = []
    current_time = 0
    
    for pid in range(101, 101 + num_processes):
        # Poisson arrival using exponential distribution
        inter_arrival = random.expovariate(1/lambda_rate)
        current_time += inter_arrival
        
        processes.append({
            'pid': pid,
            'arrival': current_time,
            'burst': random.randint(101, 200),
            'priority': random.randint(101, 200),
            'remaining': 0,
            'waiting': 0,
            'turnaround': 0,
            'response': 0,
            'completion': 0,
            'start': -1
        })
    
    return processes

# ==================== 2. ALGORITHM SIMULATIONS ====================
def fcfs_simulation(processes):
    """FCFS Scheduling"""
    sorted_procs = sorted(processes, key=lambda x: x['arrival'])
    current_time = 0
    gantt_data = []
    
    for p in sorted_procs:
        if current_time < p['arrival']:
            current_time = p['arrival']
        
        p['start'] = current_time
        p['response'] = current_time - p['arrival']
        p['waiting'] = current_time - p['arrival']
        
        gantt_data.append({
            'pid': p['pid'],
            'start': current_time,
            'end': current_time + p['burst'],
            'burst': p['burst']
        })
        
        current_time += p['burst']
        p['completion'] = current_time
        p['turnaround'] = current_time - p['arrival']
    
    return gantt_data, sorted_procs

def priority_simulation(processes):
    """Priority Non-Preemptive Scheduling with Aging"""
    sorted_procs = sorted(processes.copy(), key=lambda x: x['arrival'])
    ready_queue = []
    current_time = 0
    gantt_data = []
    completed = []
    
    while sorted_procs or ready_queue:
        while sorted_procs and sorted_procs[0]['arrival'] <= current_time:
            ready_queue.append(sorted_procs.pop(0))
        
        if ready_queue:
            # Apply aging
            for p in ready_queue:
                if current_time - p['arrival'] > 100:
                    p['priority'] = max(101, p['priority'] - 1)
            
            # Sort by priority (ascending)
            ready_queue.sort(key=lambda x: x['priority'])
            p = ready_queue.pop(0)
            
            if current_time < p['arrival']:
                current_time = p['arrival']
            
            p['start'] = current_time
            p['response'] = current_time - p['arrival']
            p['waiting'] = current_time - p['arrival']
            
            gantt_data.append({
                'pid': p['pid'],
                'start': current_time,
                'end': current_time + p['burst'],
                'burst': p['burst']
            })
            
            current_time += p['burst']
            p['completion'] = current_time
            p['turnaround'] = current_time - p['arrival']
            completed.append(p)
        else:
            current_time = sorted_procs[0]['arrival'] if sorted_procs else current_time
    
    return gantt_data, completed

def round_robin_simulation(processes, quantum=150):
    """Round Robin Scheduling"""
    sorted_procs = sorted(processes.copy(), key=lambda x: x['arrival'])
    ready_queue = []
    current_time = 0
    gantt_data = []
    completed = []
    
    for p in sorted_procs:
        p['remaining'] = p['burst']
        p['first_response'] = -1
    
    while sorted_procs or ready_queue:
        while sorted_procs and sorted_procs[0]['arrival'] <= current_time:
            p = sorted_procs.pop(0)
            ready_queue.append(p)
        
        if not ready_queue:
            current_time = sorted_procs[0]['arrival'] if sorted_procs else current_time
            continue
        
        p = ready_queue.pop(0)
        
        if p['first_response'] == -1:
            p['first_response'] = max(0, current_time - p['arrival'])
            p['response'] = p['first_response']
        
        exec_time = min(quantum, p['remaining'])
        
        gantt_data.append({
            'pid': p['pid'],
            'start': current_time,
            'end': current_time + exec_time,
            'burst': exec_time
        })
        
        current_time += exec_time
        p['remaining'] -= exec_time
        
        while sorted_procs and sorted_procs[0]['arrival'] <= current_time:
            ready_queue.append(sorted_procs.pop(0))
        
        if p['remaining'] > 0:
            ready_queue.append(p)
        else:
            p['completion'] = current_time
            p['turnaround'] = current_time - p['arrival']
            p['waiting'] = p['turnaround'] - p['burst']
            completed.append(p)
    
    return gantt_data, completed

# ==================== 3. METRICS CALCULATION ====================
def calculate_metrics(processes, total_time, algorithm_name):
    """Calculate performance metrics"""
    if not processes:
        return {}
    
    avg_waiting = sum(p['waiting'] for p in processes) / len(processes)
    avg_turnaround = sum(p['turnaround'] for p in processes) / len(processes)
    
    if 'response' in processes[0]:
        avg_response = sum(p['response'] for p in processes) / len(processes)
    else:
        avg_response = avg_waiting  # For non-preemptive
    
    total_busy = sum(p['burst'] for p in processes)
    cpu_util = (total_busy / total_time) * 100 if total_time > 0 else 0
    throughput = len(processes) / (total_time / 1000)  # processes per second
    
    return {
        'algorithm': algorithm_name,
        'avg_waiting': avg_waiting,
        'avg_turnaround': avg_turnaround,
        'avg_response': avg_response,
        'cpu_util': cpu_util,
        'throughput': throughput
    }

# ==================== 4. GRAPH GENERATION ====================
def generate_comparison_graph(results):
    """Generate bar chart comparison"""
    algorithms = [r['algorithm'] for r in results]
    
    # Prepare data
    waiting_times = [r['avg_waiting'] for r in results]
    turnaround_times = [r['avg_turnaround'] for r in results]
    response_times = [r['avg_response'] for r in results]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart 1: Time metrics
    ax1.bar(x - width, waiting_times, width, label='Avg Waiting Time', color='skyblue')
    ax1.bar(x, turnaround_times, width, label='Avg Turnaround Time', color='lightgreen')
    ax1.bar(x + width, response_times, width, label='Avg Response Time', color='salmon')
    
    ax1.set_xlabel('Scheduling Algorithm')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (wt, tt, rt) in enumerate(zip(waiting_times, turnaround_times, response_times)):
        ax1.text(i - width, wt + 10, f'{wt:.0f}', ha='center', fontsize=9)
        ax1.text(i, tt + 10, f'{tt:.0f}', ha='center', fontsize=9)
        ax1.text(i + width, rt + 10, f'{rt:.0f}', ha='center', fontsize=9)
    
    # Bar chart 2: Throughput and CPU Utilization
    throughputs = [r['throughput'] for r in results]
    cpu_utils = [r['cpu_util'] for r in results]
    
    x2 = np.arange(len(algorithms))
    
    ax2_throughput = ax2.twinx()
    
    bars1 = ax2.bar(x2 - width/2, throughputs, width, label='Throughput', color='royalblue')
    bars2 = ax2_throughput.bar(x2 + width/2, cpu_utils, width, label='CPU Utilization', color='orange')
    
    ax2.set_xlabel('Scheduling Algorithm')
    ax2.set_ylabel('Throughput (processes/sec)', color='royalblue')
    ax2_throughput.set_ylabel('CPU Utilization (%)', color='orange')
    ax2.set_title('System Efficiency Metrics')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(algorithms)
    
    # Add value labels
    for bar, value in zip(bars1, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{value:.2f}', 
                ha='center', va='bottom', color='royalblue', fontsize=9)
    
    for bar, value in zip(bars2, cpu_utils):
        height = bar.get_height()
        ax2_throughput.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{value:.1f}%', 
                          ha='center', va='bottom', color='orange', fontsize=9)
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Graph 1 saved as 'performance_comparison.png'")

def generate_gantt_charts(fcfs_gantt, priority_gantt, rr_gantt):
    """Generate Gantt charts for first 5 processes"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    
    # Filter for first 5 processes
    target_pids = [101, 102, 103, 104, 105]
    
    # FCFS Gantt
    fcfs_filtered = [g for g in fcfs_gantt if g['pid'] in target_pids]
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for i, g in enumerate(fcfs_filtered):
        axes[0].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[i], edgecolor='black')
        # Add time labels
        axes[0].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]}ms', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_title('FCFS Scheduling - First 5 Processes')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Priority Gantt
    priority_filtered = [g for g in priority_gantt if g['pid'] in target_pids]
    priority_sorted = sorted(priority_filtered, key=lambda x: x['start'])
    
    for i, g in enumerate(priority_sorted):
        axes[1].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[i], edgecolor='black')
        axes[1].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]}ms', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Priority Non-Preemptive Scheduling - First 5 Processes')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Round Robin Gantt
    rr_filtered = [g for g in rr_gantt if g['pid'] in target_pids]
    
    # Group by process ID
    pid_groups = {}
    for g in rr_filtered:
        pid = g['pid']
        if pid not in pid_groups:
            pid_groups[pid] = []
        pid_groups[pid].append(g)
    
    y_positions = {pid: i for i, pid in enumerate(target_pids)}
    
    for pid, executions in pid_groups.items():
        y_pos = y_positions[pid]
        for exec in executions:
            axes[2].barh(f'P{pid}', exec['end'] - exec['start'], 
                        left=exec['start'], color=colors[y_pos], edgecolor='black')
            if exec['end'] - exec['start'] > 20:  # Only label if enough space
                axes[2].text(exec['start'] + (exec['end'] - exec['start'])/2, y_pos, 
                            f'{exec["end"]-exec["start"]}ms', ha='center', va='center', 
                            color='black', fontsize=8)
    
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('Round Robin Scheduling (Quantum=150ms) - First 5 Processes')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('gantt_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Graph 2 saved as 'gantt_charts.png'")

def generate_radar_chart(results):
    """Generate radar chart for multi-metric comparison"""
    # Categories for radar chart
    categories = ['Throughput', 'Response Time', 'Fairness', 
                  'CPU Utilization', 'Simplicity', 'Starvation Prevention']
    
    # Normalized scores (1-10 scale)
    # Based on our analysis
    fcfs_scores = [9, 3, 4, 10, 10, 10]      # Scale: 1=Worst, 10=Best
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
    plt.show()
    
    print("✓ Graph 3 saved as 'radar_chart.png'")

# ==================== 5. MAIN EXECUTION ====================
def main():
    print("=" * 70)
    print("CPU SCHEDULING ALGORITHMS SIMULATION WITH GRAPH GENERATION")
    print("=" * 70)
    
    # Step 1: Generate processes
    print("\n📊 Generating 100 processes with Poisson distribution (λ=150)...")
    processes = generate_processes(100, 150)
    print("✓ Process generation complete")
    
    # Step 2: Run simulations
    print("\n⚙️ Running algorithm simulations...")
    
    # FCFS
    print("  • Running FCFS simulation...")
    fcfs_gantt, fcfs_procs = fcfs_simulation(processes.copy())
    fcfs_total_time = max(p['completion'] for p in fcfs_procs)
    fcfs_results = calculate_metrics(fcfs_procs, fcfs_total_time, "FCFS")
    
    # Priority
    print("  • Running Priority Non-Preemptive simulation...")
    priority_gantt, priority_procs = priority_simulation(processes.copy())
    priority_total_time = max(p['completion'] for p in priority_procs)
    priority_results = calculate_metrics(priority_procs, priority_total_time, "Priority")
    
    # Round Robin
    print("  • Running Round Robin simulation...")
    rr_gantt, rr_procs = round_robin_simulation(processes.copy(), 150)
    rr_total_time = max(p['completion'] for p in rr_procs)
    rr_results = calculate_metrics(rr_procs, rr_total_time, "Round Robin")
    
    # Step 3: Display results
    print("\n📈 Performance Results:")
    print("-" * 80)
    
    results = [fcfs_results, priority_results, rr_results]
    
    table_data = []
    for r in results:
        table_data.append([
            r['algorithm'],
            f"{r['avg_waiting']:.1f} ms",
            f"{r['avg_turnaround']:.1f} ms",
            f"{r['avg_response']:.1f} ms",
            f"{r['cpu_util']:.1f}%",
            f"{r['throughput']:.2f}"
        ])
    
    print(tabulate(table_data, 
                   headers=["Algorithm", "Avg Waiting", "Avg Turnaround", 
                           "Avg Response", "CPU Util", "Throughput"],
                   tablefmt="grid"))
    
    # Step 4: Generate graphs
    print("\n🎨 Generating graphs...")
    
    # Graph 1: Comparison Bar Chart
    print("  • Creating performance comparison chart...")
    generate_comparison_graph(results)
    
    # Graph 2: Gantt Charts
    print("  • Creating Gantt charts...")
    generate_gantt_charts(fcfs_gantt, priority_gantt, rr_gantt)
    
    # Graph 3: Radar Chart
    print("  • Creating radar chart...")
    generate_radar_chart(results)
    
    print("\n" + "=" * 70)
    print("✅ SIMULATION COMPLETE!")
    print("Three graphs have been generated:")
    print("  1. performance_comparison.png - Bar chart comparison")
    print("  2. gantt_charts.png - Gantt charts for each algorithm")
    print("  3. radar_chart.png - Multi-metric radar chart")
    print("=" * 70)

if __name__ == "__main__":
    main()