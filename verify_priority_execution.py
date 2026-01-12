"""
Verification script to show Priority Non-Preemptive execution details
"""

import sys
sys.path.append('.')
from cpu_scheduling_simulation import generate_processes, priority_nonpreemptive_simulation

def verify_priority_execution():
    print("=" * 80)
    print("PRIORITY NON-PREEMPTIVE EXECUTION VERIFICATION")
    print("=" * 80)
    
    # Generate processes
    processes = generate_processes()
    
    # Run priority simulation
    gantt_data, completed_procs = priority_nonpreemptive_simulation(processes)
    
    # Show first 10 processes execution order
    print("\nExecution Order (First 10 Processes):")
    print("-" * 60)
    
    first_10_gantt = [g for g in gantt_data if g['pid'] <= 110]
    first_10_gantt.sort(key=lambda x: x['start'])
    
    for i, g in enumerate(first_10_gantt, 1):
        print(f"{i:2d}. P{g['pid']}: {g['start']:.1f}-{g['end']:.1f} ms")
    
    # Show waiting times for first 10 processes
    print("\nWaiting Times (First 10 Processes):")
    print("-" * 60)
    
    # Create process lookup
    proc_dict = {p['pid']: p for p in completed_procs}
    
    target_pids = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    total_waiting = 0
    
    for pid in target_pids:
        if pid in proc_dict:
            p = proc_dict[pid]
            waiting_time = p['waiting']
            total_waiting += waiting_time
            print(f"P{pid}: {waiting_time:.1f} ms")
    
    avg_waiting = total_waiting / len(target_pids)
    print(f"\nAverage Waiting Time (First 10): {avg_waiting:.1f} ms")
    
    # Expected values verification
    expected_execution = [
        (101, 0.0, 187.0),
        (102, 187.0, 332.0),
        (104, 332.0, 444.0),
        (106, 444.0, 573.0),
        (107, 573.0, 754.0),
        (108, 754.0, 909.0),
        (110, 909.0, 1106.0),
        (105, 1106.0, 1304.0),
        (109, 1304.0, 1438.0),
        (103, 1438.0, 1614.0)
    ]
    
    expected_waiting = [0, 99.7, 1192.2, 19.4, 716.9, 10.5, 78.3, 167.6, 538.8, 85.3]
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    execution_correct = True
    waiting_correct = True
    
    # Check execution order
    for i, (expected_pid, expected_start, expected_end) in enumerate(expected_execution):
        actual = first_10_gantt[i]
        if (actual['pid'] != expected_pid or 
            abs(actual['start'] - expected_start) > 0.1 or 
            abs(actual['end'] - expected_end) > 0.1):
            execution_correct = False
            break
    
    # Check waiting times
    for i, pid in enumerate(target_pids):
        if pid in proc_dict:
            actual_waiting = proc_dict[pid]['waiting']
            expected_wait = expected_waiting[i]
            if abs(actual_waiting - expected_wait) > 0.1:
                waiting_correct = False
                break
    
    print(f"Execution Order: {'CORRECT' if execution_correct else 'INCORRECT'}")
    print(f"Waiting Times: {'CORRECT' if waiting_correct else 'INCORRECT'}")
    
    if execution_correct and waiting_correct:
        print("\nAll verification checks PASSED!")
    else:
        print("\nSome verification checks FAILED!")
    
    print("=" * 80)

if __name__ == "__main__":
    verify_priority_execution()