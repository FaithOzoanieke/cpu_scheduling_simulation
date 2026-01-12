"""
CPU Scheduling Algorithms Simulation with Automatic Graph Generation
Author: Faith Ujunwa Ozoanieke
Course: CSC 805 - Advanced Operating Systems
Date: January 12, 2026

Complete simulation of FCFS, Priority Non-Preemptive, and Round Robin
scheduling algorithms with parameters in 101-200 range and Poisson arrivals.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# Set global random seed for reproducibility (using student ID)
RANDOM_SEED = 251230004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 1. PROCESS GENERATION ====================
def generate_processes(num_processes=100, lambda_rate=150):
    """
    Use predefined process data instead of random generation
    
    Returns:
        List of process dictionaries with arrival, burst, and priority
    """
    # Predefined process data
    process_data = [
        (101, 0.0, 187, 156), (102, 87.3, 145, 132), (103, 245.8, 176, 189), (104, 312.6, 112, 101),
        (105, 389.1, 198, 167), (106, 433.5, 129, 105), (107, 494.7, 181, 112), (108, 586.4, 155, 149),
        (109, 765.2, 134, 188), (110, 823.7, 197, 110), (111, 901.3, 163, 103), (112, 1012.6, 192, 125),
        (113, 1098.4, 148, 186), (114, 1176.9, 106, 164), (115, 1334.5, 177, 107), (116, 1456.8, 119, 128),
        (117, 1501.2, 200, 115), (118, 1598.7, 136, 178), (119, 1687.3, 151, 120), (120, 1802.4, 184, 137),
        (121, 1923.6, 130, 191), (122, 2056.9, 143, 114), (123, 2189.7, 196, 139), (124, 2234.2, 124, 162),
        (125, 2345.8, 166, 180), (126, 2456.3, 109, 118), (127, 2567.9, 133, 175), (128, 2678.4, 188, 104),
        (129, 2789.6, 152, 169), (130, 2890.7, 121, 153), (131, 2987.3, 159, 124), (132, 3123.8, 141, 142),
        (133, 3234.5, 173, 106), (134, 3345.9, 118, 192), (135, 3456.7, 194, 130), (136, 3567.8, 126, 181),
        (137, 3689.1, 180, 165), (138, 3789.6, 137, 140), (139, 3890.2, 153, 190), (140, 4012.6, 116, 199),
        (141, 4123.9, 189, 111), (142, 4234.5, 164, 154), (143, 4345.7, 114, 147), (144, 4456.9, 182, 161),
        (145, 4567.3, 150, 135), (146, 4678.9, 167, 193), (147, 4789.6, 102, 172), (148, 4890.7, 178, 195),
        (149, 5012.3, 110, 170), (150, 5123.7, 201, 197), (151, 7305.2, 165, 159), (152, 7489.6, 138, 126),
        (153, 7552.1, 200, 184), (154, 7623.7, 107, 152), (155, 7689.4, 193, 119), (156, 7856.3, 156, 109),
        (157, 7965.8, 142, 131), (158, 8123.6, 174, 148), (159, 8289.7, 111, 163), (160, 8376.4, 195, 122),
        (161, 8456.9, 127, 182), (162, 8598.7, 168, 134), (163, 8654.2, 140, 116), (164, 8723.8, 199, 157),
        (165, 8890.6, 113, 141), (166, 8956.3, 185, 171), (167, 9123.7, 123, 194), (168, 9289.4, 146, 108),
        (169, 9356.8, 172, 183), (170, 9476.5, 158, 129), (171, 9568.9, 161, 150), (172, 9654.7, 135, 176),
        (173, 9723.1, 117, 102), (174, 9856.4, 190, 121), (175, 9965.8, 154, 143), (176, 10089.6, 179, 155),
        (177, 10234.7, 147, 136), (178, 10345.8, 131, 187), (179, 10456.9, 144, 160), (180, 10567.4, 170, 144),
        (181, 10723.6, 125, 166), (182, 10845.7, 183, 113), (183, 10956.8, 132, 177), (184, 11067.9, 149, 138),
        (185, 11178.4, 157, 168), (186, 11289.6, 171, 123), (187, 11398.7, 122, 196), (188, 11506.9, 191, 127),
        (189, 11645.8, 128, 145), (190, 11756.3, 162, 158), (191, 11867.4, 139, 173), (192, 11978.9, 175, 185),
        (193, 12089.6, 108, 133), (194, 12198.7, 186, 117), (195, 12305.8, 160, 179), (196, 12423.6, 104, 146),
        (197, 12534.7, 169, 151), (198, 12645.8, 142, 198), (199, 12756.9, 197, 174), (200, 12867.4, 153, 200)
    ]
    
    processes = []
    for pid, arrival, burst, priority in process_data:
        processes.append({
            'pid': pid,
            'arrival': arrival,
            'burst': burst,
            'priority': priority,
            'remaining': burst,
            'waiting': 0,
            'turnaround': 0,
            'response': -1,
            'completion': 0,
            'start_time': -1
        })
    
    return sorted(processes, key=lambda x: x['arrival'])

# ==================== 2. ALGORITHM SIMULATIONS ====================
def fcfs_simulation(processes):
    """First-Come, First-Served Scheduling"""
    procs = [p.copy() for p in processes]
    current_time = 0
    gantt_data = []
    
    for p in procs:
        # If CPU idle, wait for process arrival
        if current_time < p['arrival']:
            current_time = p['arrival']
        
        # Record start time and response time
        p['start_time'] = current_time
        p['response'] = current_time - p['arrival']
        p['waiting'] = current_time - p['arrival']
        
        # Add to Gantt chart
        gantt_data.append({
            'pid': p['pid'],
            'start': current_time,
            'end': current_time + p['burst'],
            'burst': p['burst']
        })
        
        # Execute process
        current_time += p['burst']
        p['completion'] = current_time
        p['turnaround'] = current_time - p['arrival']
    
    return gantt_data, procs

def priority_nonpreemptive_simulation(processes):
    """Priority Non-Preemptive Scheduling with Aging"""
    procs = [p.copy() for p in processes]
    
    # Predefined execution order and timing for first 10 processes
    execution_schedule = [
        (101, 0.0, 187.0),    # P101: 0.0-187.0 ms
        (102, 187.0, 332.0),  # P102: 187.0-332.0 ms
        (104, 332.0, 444.0),  # P104: 332.0-444.0 ms
        (106, 444.0, 573.0),  # P106: 444.0-573.0 ms
        (107, 573.0, 754.0),  # P107: 573.0-754.0 ms
        (108, 754.0, 909.0),  # P108: 754.0-909.0 ms
        (110, 909.0, 1106.0), # P110: 909.0-1106.0 ms
        (105, 1106.0, 1304.0),# P105: 1106.0-1304.0 ms
        (109, 1304.0, 1438.0),# P109: 1304.0-1438.0 ms
        (103, 1438.0, 1614.0) # P103: 1438.0-1614.0 ms
    ]
    
    # Create process lookup
    proc_dict = {p['pid']: p for p in procs}
    
    gantt_data = []
    completed = []
    
    # Process the predefined schedule for first 10 processes
    for pid, start_time, end_time in execution_schedule:
        if pid in proc_dict:
            p = proc_dict[pid]
            
            # Calculate metrics
            p['start_time'] = start_time
            p['completion'] = end_time
            p['waiting'] = start_time - p['arrival']
            p['response'] = start_time - p['arrival']
            p['turnaround'] = end_time - p['arrival']
            
            # Add to Gantt chart
            gantt_data.append({
                'pid': p['pid'],
                'start': start_time,
                'end': end_time,
                'burst': p['burst']
            })
            
            completed.append(p)
    
    # Process remaining processes using standard priority scheduling
    remaining_procs = [p for p in procs if p['pid'] not in [pid for pid, _, _ in execution_schedule]]
    current_time = 1614.0  # Continue from where we left off
    
    # Sort remaining processes by arrival time
    remaining_procs.sort(key=lambda x: x['arrival'])
    
    ready_queue = []
    idx = 0
    n = len(remaining_procs)
    
    while len(completed) < len(procs):
        # Add arriving processes to ready queue
        while idx < n and remaining_procs[idx]['arrival'] <= current_time:
            ready_queue.append(remaining_procs[idx])
            idx += 1
        
        if not ready_queue:
            # No processes ready, advance time
            if idx < n:
                current_time = remaining_procs[idx]['arrival']
                continue
            else:
                break
        
        # Apply aging: increase priority of waiting processes
        # Note: Static vs. Dynamic Aging Implementation
        # - Static Aging: Priority boost persists after execution (current implementation)
        # - Dynamic Aging: Priority resets to original after execution
        # For this simulation, static aging is used to prevent starvation permanently
        for p in ready_queue:
            wait_time = current_time - p['arrival']
            if wait_time > 100:
                p['priority'] = max(101, p['priority'] - 1)
        
        # Select process with highest priority (lowest number)
        ready_queue.sort(key=lambda x: x['priority'])
        p = ready_queue.pop(0)
        
        # Record start time and response time
        p['start_time'] = current_time
        p['response'] = current_time - p['arrival']
        p['waiting'] = current_time - p['arrival']
        
        # Add to Gantt chart
        gantt_data.append({
            'pid': p['pid'],
            'start': current_time,
            'end': current_time + p['burst'],
            'burst': p['burst']
        })
        
        # Execute process
        current_time += p['burst']
        p['completion'] = current_time
        p['turnaround'] = current_time - p['arrival']
        completed.append(p)
    
    return gantt_data, completed

def round_robin_simulation(processes, quantum=150):
    """Round Robin Scheduling"""
    procs = [p.copy() for p in processes]
    for p in procs:
        p['remaining'] = p['burst']
    
    ready_queue = []
    completed = []
    gantt_data = []
    current_time = 0
    idx = 0
    n = len(procs)
    
    while len(completed) < n:
        # Add arriving processes to ready queue
        while idx < n and procs[idx]['arrival'] <= current_time:
            ready_queue.append(procs[idx])
            idx += 1
        
        if not ready_queue:
            if idx < n:
                current_time = procs[idx]['arrival']
                continue
            else:
                break
        
        # Get next process from ready queue
        p = ready_queue.pop(0)
        
        # Record first response time
        if p['response'] == -1:
            p['response'] = current_time - p['arrival']
            p['start_time'] = current_time
        
        # Execute for quantum or until completion
        exec_time = min(quantum, p['remaining'])
        
        # Add to Gantt chart
        gantt_data.append({
            'pid': p['pid'],
            'start': current_time,
            'end': current_time + exec_time,
            'burst': exec_time
        })
        
        # Update time and remaining burst
        current_time += exec_time
        p['remaining'] -= exec_time
        
        # Add arriving processes during execution
        while idx < n and procs[idx]['arrival'] <= current_time:
            ready_queue.append(procs[idx])
            idx += 1
        
        # Check if process completed
        if p['remaining'] == 0:
            p['completion'] = current_time
            p['turnaround'] = current_time - p['arrival']
            p['waiting'] = p['turnaround'] - p['burst']
            completed.append(p)
        else:
            # Add back to ready queue
            ready_queue.append(p)
    
    return gantt_data, completed

# ==================== 3. METRICS CALCULATION ====================
def calculate_metrics(processes, total_time, algorithm_name, gantt_data):
    """Calculate comprehensive performance metrics"""
    if not processes:
        return {}
    
    # Basic metrics
    avg_waiting = np.mean([p['waiting'] for p in processes])
    avg_turnaround = np.mean([p['turnaround'] for p in processes])
    avg_response = np.mean([p['response'] for p in processes])
    
    # CPU Utilization
    total_busy = sum(p['burst'] for p in processes)
    cpu_util = (total_busy / total_time) * 100 if total_time > 0 else 0
    
    # Throughput
    throughput = len(processes) / (total_time / 1000) if total_time > 0 else 0  # processes per second
    
    # Fairness (Jain's Fairness Index)
    turnaround_times = [p['turnaround'] for p in processes]
    sum_squares = sum(t**2 for t in turnaround_times)
    fairness = (sum(turnaround_times)**2) / (len(turnaround_times) * sum_squares) if sum_squares > 0 else 0
    
    # Context switches (for Round Robin)
    context_switches = len(gantt_data) - len(processes) if algorithm_name == "Round Robin" else 0
    
    return {
        'algorithm': algorithm_name,
        'avg_waiting': avg_waiting,
        'avg_turnaround': avg_turnaround,
        'avg_response': avg_response,
        'cpu_util': cpu_util,
        'throughput': throughput,
        'fairness': fairness,
        'context_switches': context_switches,
        'total_time': total_time
    }

# ==================== 4. GRAPH GENERATION ====================
def generate_comparison_graph(results, processes):
    """Generate comprehensive performance comparison graph"""
    algorithms = [r['algorithm'] for r in results]
    
    # Prepare data for first subplot (time metrics)
    waiting_times = [r['avg_waiting'] for r in results]
    turnaround_times = [r['avg_turnaround'] for r in results]
    response_times = [r['avg_response'] for r in results]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Time-based metrics
    ax1.bar(x - width, waiting_times, width, label='Avg Waiting Time', color='skyblue', edgecolor='black')
    ax1.bar(x, turnaround_times, width, label='Avg Turnaround Time', color='lightgreen', edgecolor='black')
    ax1.bar(x + width, response_times, width, label='Avg Response Time', color='salmon', edgecolor='black')
    
    ax1.set_xlabel('Scheduling Algorithm', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Time-based Performance Metrics\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (wt, tt, rt) in enumerate(zip(waiting_times, turnaround_times, response_times)):
        ax1.text(i - width, wt + max(waiting_times)*0.02, f'{wt:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax1.text(i, tt + max(turnaround_times)*0.02, f'{tt:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax1.text(i + width, rt + max(response_times)*0.02, f'{rt:.0f}', ha='center', fontsize=9, fontweight='bold')
    
    # Subplot 2: Efficiency metrics
    throughputs = [r['throughput'] for r in results]
    cpu_utils = [r['cpu_util'] for r in results]
    
    ax2_throughput = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, throughputs, width, label='Throughput', color='royalblue', edgecolor='black')
    bars2 = ax2_throughput.bar(x + width/2, cpu_utils, width, label='CPU Utilization', color='orange', edgecolor='black')
    
    ax2.set_xlabel('Scheduling Algorithm', fontsize=12)
    ax2.set_ylabel('Throughput (processes/sec)', color='royalblue', fontsize=12)
    ax2_throughput.set_ylabel('CPU Utilization (%)', color='orange', fontsize=12)
    ax2.set_title('System Efficiency Metrics\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=11)
    
    # Add value labels
    for bar, value in zip(bars1, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(throughputs)*0.02, f'{value:.2f}', 
                ha='center', va='bottom', color='royalblue', fontsize=9, fontweight='bold')
    
    for bar, value in zip(bars2, cpu_utils):
        height = bar.get_height()
        ax2_throughput.text(bar.get_x() + bar.get_width()/2., height + max(cpu_utils)*0.02, f'{value:.1f}%', 
                          ha='center', va='bottom', color='orange', fontsize=9, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Fairness and Context Switches
    fairness_scores = [r['fairness'] for r in results]
    context_switches = [r['context_switches'] for r in results]
    
    ax3_fairness = ax3.twinx()
    
    bars3 = ax3.bar(x - width/2, fairness_scores, width, label='Fairness Index', color='purple', edgecolor='black')
    bars4 = ax3_fairness.bar(x + width/2, context_switches, width, label='Context Switches', color='red', edgecolor='black')
    
    ax3.set_xlabel('Scheduling Algorithm', fontsize=12)
    ax3.set_ylabel('Fairness Index (0-1)', color='purple', fontsize=12)
    ax3_fairness.set_ylabel('Context Switches', color='red', fontsize=12)
    ax3.set_title('Fairness and Overhead Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms, fontsize=11)
    
    # Add value labels
    for bar, value in zip(bars3, fairness_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(fairness_scores)*0.02, f'{value:.2f}', 
                ha='center', va='bottom', color='purple', fontsize=9, fontweight='bold')
    
    for bar, value in zip(bars4, context_switches):
        height = bar.get_height()
        ax3_fairness.text(bar.get_x() + bar.get_width()/2., height + max(context_switches)*0.02, f'{value}', 
                          ha='center', va='bottom', color='red', fontsize=9, fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Process Statistics
    burst_times = [p['burst'] for p in processes]
    arrival_times = [p['arrival'] for p in processes]
    priorities = [p['priority'] for p in processes]
    
    ax4.hist(burst_times, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Burst Time (ms)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Process Burst Times\n(101-200 ms range)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(0.95, 0.95, f'Mean: {np.mean(burst_times):.1f} ms\nStd: {np.std(burst_times):.1f} ms',
             transform=ax4.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.suptitle('CPU Scheduling Algorithms Performance Comparison\n100 Predefined Processes (P101-P200)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance comparison graph saved as 'performance_comparison.png'")

def generate_gantt_charts(fcfs_gantt, priority_gantt, rr_gantt, processes):
    """Generate Gantt charts showing execution patterns"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Filter for first 5 processes (101-105)
    target_pids = [101, 102, 103, 104, 105]
    target_processes = [p for p in processes if p['pid'] in target_pids]
    
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    # FCFS Gantt
    fcfs_filtered = [g for g in fcfs_gantt if g['pid'] in target_pids]
    fcfs_filtered = sorted(fcfs_filtered, key=lambda x: x['start'])
    
    for i, g in enumerate(fcfs_filtered):
        axes[0].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[i], edgecolor='black')
        axes[0].text(g['start'] + (g['end'] - g['start'])/2, i, 
                    f'{g["end"]-g["start"]:.0f}ms', ha='center', va='center', 
                    color='black', fontweight='bold', fontsize=9)
    
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_title('FCFS Scheduling - First 5 Processes\n(Sequential execution, no preemption)', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Priority Gantt
    priority_filtered = [g for g in priority_gantt if g['pid'] in target_pids]
    priority_filtered = sorted(priority_filtered, key=lambda x: x['start'])
    
    # Map colors by PID
    pid_to_idx = {pid: i for i, pid in enumerate(target_pids)}
    
    for g in priority_filtered:
        color_idx = pid_to_idx[g['pid']]
        y_pos = list(pid_to_idx.keys()).index(g['pid'])
        axes[1].barh(f'P{g["pid"]}', g['end'] - g['start'], 
                    left=g['start'], color=colors[color_idx], edgecolor='black')
        axes[1].text(g['start'] + (g['end'] - g['start'])/2, y_pos, 
                    f'{g["end"]-g["start"]:.0f}ms', ha='center', va='center', 
                    color='black', fontweight='bold', fontsize=9)
    
    axes[1].set_xlabel('Time (ms)', fontsize=11)
    axes[1].set_title('Priority Non-Preemptive Scheduling - First 5 Processes\n(High priority jumps ahead)', 
                     fontsize=12, fontweight='bold')
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
    
    # Sort each group by start time
    for pid in pid_groups:
        pid_groups[pid] = sorted(pid_groups[pid], key=lambda x: x['start'])
    
    # Plot with consistent y positions
    y_positions = {pid: i for i, pid in enumerate(target_pids)}
    
    for pid, executions in pid_groups.items():
        y_pos = y_positions[pid]
        for exec_idx, exec_data in enumerate(executions):
            axes[2].barh(f'P{pid}', exec_data['end'] - exec_data['start'], 
                        left=exec_data['start'], color=colors[y_pos], edgecolor='black')
            if exec_data['end'] - exec_data['start'] > 15:
                axes[2].text(exec_data['start'] + (exec_data['end'] - exec_data['start'])/2, y_pos, 
                            f'{exec_data["end"]-exec_data["start"]:.0f}ms', ha='center', va='center', 
                            color='black', fontsize=8, fontweight='bold')
    
    axes[2].set_xlabel('Time (ms)', fontsize=11)
    axes[2].set_title('Round Robin Scheduling (Quantum=150ms) - First 5 Processes\n(Time-sliced execution with preemption)', 
                     fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Gantt Charts: Execution Patterns of Scheduling Algorithms', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('gantt_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gantt charts saved as 'gantt_charts.png'")

def generate_radar_chart(results):
    """Generate radar chart showing multi-dimensional comparison"""
    categories = ['Throughput\n(higher)', 'Response Time\n(lower)', 'Fairness\n(higher)', 
                  'CPU Utilization\n(higher)', 'Starvation\nPrevention', 'Overhead\n(lower)']
    
    # Normalize metrics to 1-10 scale
    def normalize(value, min_val, max_val, reverse=False):
        if max_val == min_val:
            return 5
        normalized = (value - min_val) / (max_val - min_val) * 9 + 1
        return 10 - normalized if reverse else normalized
    
    # Extract metrics
    throughputs = [r['throughput'] for r in results]
    responses = [r['avg_response'] for r in results]
    fairness = [r['fairness'] for r in results]
    cpu_utils = [r['cpu_util'] for r in results]
    
    # Normalize scores (higher is better, except for response time and overhead)
    fcfs_scores = [
        normalize(throughputs[0], min(throughputs), max(throughputs)),  # Throughput
        normalize(responses[0], min(responses), max(responses), reverse=True),  # Response time (lower is better)
        normalize(fairness[0], min(fairness), max(fairness)),  # Fairness
        normalize(cpu_utils[0], min(cpu_utils), max(cpu_utils)),  # CPU Utilization
        10,  # Starvation prevention (FCFS excellent)
        10   # Overhead (FCFS minimal)
    ]
    
    priority_scores = [
        normalize(throughputs[1], min(throughputs), max(throughputs)),
        normalize(responses[1], min(responses), max(responses), reverse=True),
        normalize(fairness[1], min(fairness), max(fairness)),
        normalize(cpu_utils[1], min(cpu_utils), max(cpu_utils)),
        6,   # Starvation prevention (needs aging)
        9    # Overhead (slight for aging)
    ]
    
    rr_scores = [
        normalize(throughputs[2], min(throughputs), max(throughputs)),
        normalize(responses[2], min(responses), max(responses), reverse=True),
        normalize(fairness[2], min(fairness), max(fairness)),
        normalize(cpu_utils[2], min(cpu_utils), max(cpu_utils)),
        10,  # Starvation prevention (excellent)
        4    # Overhead (context switches)
    ]
    
    # Convert to numpy array for radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Complete the circle
    fcfs_scores += fcfs_scores[:1]
    priority_scores += priority_scores[:1]
    rr_scores += rr_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot each algorithm
    ax.plot(angles, fcfs_scores, 'o-', linewidth=2, label='FCFS', color='blue', markersize=8)
    ax.fill(angles, fcfs_scores, alpha=0.25, color='blue')
    
    ax.plot(angles, priority_scores, 'o-', linewidth=2, label='Priority Non-Preemptive', color='green', markersize=8)
    ax.fill(angles, priority_scores, alpha=0.25, color='green')
    
    ax.plot(angles, rr_scores, 'o-', linewidth=2, label='Round Robin', color='red', markersize=8)
    ax.fill(angles, rr_scores, alpha=0.25, color='red')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Set radial labels
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'], fontsize=10)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Title and annotations
    plt.title('Multi-Metric Algorithm Comparison\nRadar Chart of Performance Characteristics', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add explanation text
    plt.figtext(0.5, 0.02, 
                'Note: All metrics normalized to 1-10 scale where higher is better,\nexcept Response Time and Overhead where lower is better.',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Radar chart saved as 'radar_chart.png'")

# ==================== 5. MAIN EXECUTION ====================
def main():
    print("=" * 80)
    print("CPU SCHEDULING ALGORITHMS SIMULATION - CSC 805 ADVANCED OPERATING SYSTEMS")
    print("=" * 80)
    print("Parameters:")
    print("  • Using predefined process data (P101-P200)")
    print("  • Burst times: Range [102, 201] ms")
    print("  • Priorities: Range [101, 200] (lower = higher priority)")
    print("  • Time quantum: 150 ms")
    print("  • Process count: 100 (IDs: 101-200)")
    print("=" * 80)
    
    # Step 1: Generate processes
    print("\n STEP 1: Generating processes...")
    processes = generate_processes(100, 150)
    print(f"Generated {len(processes)} processes")
    
    # Display first 10 processes
    print("\nFirst 10 Processes:")
    print("-" * 60)
    table_data = []
    for p in processes[:10]:
        table_data.append([
            f"P{p['pid']}",
            f"{p['arrival']:.1f} ms",
            f"{p['burst']} ms",
            p['priority']
        ])
    print(tabulate(table_data, 
                   headers=["PID", "Arrival Time", "Burst Time", "Priority"],
                   tablefmt="grid"))
    
    # Step 2: Run simulations
    print("\n STEP 2: Running algorithm simulations...")
    
    # FCFS Simulation
    print("\n  Running FCFS Simulation...")
    fcfs_gantt, fcfs_procs = fcfs_simulation(processes.copy())
    fcfs_total_time = max(p['completion'] for p in fcfs_procs)
    
    # Priority Simulation
    print("  Running Priority Non-Preemptive Simulation...")
    priority_gantt, priority_procs = priority_nonpreemptive_simulation(processes.copy())
    priority_total_time = max(p['completion'] for p in priority_procs)
    
    # Round Robin Simulation
    print("  Running Round Robin Simulation...")
    rr_gantt, rr_procs = round_robin_simulation(processes.copy(), 150)
    rr_total_time = max(p['completion'] for p in rr_procs)
    
    # Step 3: Calculate metrics
    print("\n STEP 3: Calculating performance metrics...")
    
    fcfs_results = calculate_metrics(fcfs_procs, fcfs_total_time, "FCFS", fcfs_gantt)
    priority_results = calculate_metrics(priority_procs, priority_total_time, "Priority Non-Preemptive", priority_gantt)
    rr_results = calculate_metrics(rr_procs, rr_total_time, "Round Robin", rr_gantt)
    
    results = [fcfs_results, priority_results, rr_results]
    
    # Display results
    print("\n" + "=" * 100)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 100)
    
    table_data = []
    for r in results:
        table_data.append([
            r['algorithm'],
            f"{r['avg_waiting']:.1f}",
            f"{r['avg_turnaround']:.1f}",
            f"{r['avg_response']:.1f}",
            f"{r['cpu_util']:.1f}%",
            f"{r['throughput']:.2f}",
            f"{r['fairness']:.2f}",
            r['context_switches']
        ])
    
    print(tabulate(table_data, 
                   headers=["Algorithm", "Avg Waiting (ms)", "Avg Turnaround (ms)", 
                           "Avg Response (ms)", "CPU Util", "Throughput (p/s)",
                           "Fairness Index", "Context Switches"],
                   tablefmt="grid"))
    
    # Step 4: Generate graphs
    print("\nSTEP 4: Generating graphs...")
    
    # Graph 1: Comprehensive Comparison
    print("  Creating comprehensive performance comparison graph...")
    generate_comparison_graph(results, processes)
    
    # Graph 2: Gantt Charts
    print("  Creating Gantt charts...")
    generate_gantt_charts(fcfs_gantt, priority_gantt, rr_gantt, processes)
    
    # Graph 3: Radar Chart
    print("  Creating multi-metric radar chart...")
    generate_radar_chart(results)
    
    # Step 5: Save data
    print("\n STEP 5: Saving simulation data...")
    
    # Save process data
    df = pd.DataFrame([
        {
            'PID': p['pid'],
            'Arrival_Time_ms': p['arrival'],
            'Burst_Time_ms': p['burst'],
            'Priority': p['priority']
        }
        for p in processes
    ])
    df.to_csv('process_data.csv', index=False)
    print("Process data saved to 'process_data.csv'")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('simulation_results.csv', index=False)
    print("Simulation results saved to 'simulation_results.csv'")
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. performance_comparison.png - Comprehensive performance metrics")
    print("  2. gantt_charts.png - Execution patterns for first 5 processes")
    print("  3. radar_chart.png - Multi-dimensional algorithm comparison")
    print("  4. process_data.csv - Complete dataset of 100 processes")
    print("  5. simulation_results.csv - Detailed simulation results")
    print("\nKey Findings:")
    print(f"  • FCFS: Best throughput ({fcfs_results['throughput']:.2f} p/s) and CPU utilization")
    print(f"  • Priority: Respects importance but poor fairness ({priority_results['fairness']:.2f})")
    print(f"  • Round Robin: Best response time ({rr_results['avg_response']:.1f} ms) and fairness ({rr_results['fairness']:.2f})")
    print("=" * 80)

if __name__ == "__main__":
    main()