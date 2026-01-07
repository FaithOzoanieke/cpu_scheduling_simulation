HOW TO GENERATE THE GRAPHS
Method A: Using the Complete Code (Recommended)
Copy this complete working Python script that generates ALL graphs automatically:


Method B: Step-by-Step (If you want to understand)
If you prefer to understand each graph, here's how to generate them separately:

Step 1: Install Required Libraries
pip install matplotlib numpy tabulate

Step 2: Run the Complete Script
python scheduling_simulation.py


This will automatically generate:

performance_comparison.png - Bar charts comparing all metrics

gantt_charts.png - Gantt charts for each algorithm

radar_chart.png - Multi-metric comparison

 Generating 100 processes with Poisson distribution (λ=150)...
✓ Process generation complete

⚙️ Running algorithm simulations...
  • Running FCFS simulation...
  • Running Priority Non-Preemptive simulation...
  • Running Round Robin simulation...

📈 Performance Results:
--------------------------------------------------------------------------------
+-------------+---------------+------------------+----------------+------------+--------------+
| Algorithm   | Avg Waiting   | Avg Turnaround   | Avg Response   | CPU Util   |   Throughput |      
+=============+===============+==================+================+============+==============+      
| FCFS        | 225.9 ms      | 379.2 ms         | 225.9 ms       | 85.5%      |         5.58 |      
+-------------+---------------+------------------+----------------+------------+--------------+      
| Priority    | 221.9 ms      | 375.1 ms         | 221.9 ms       | 85.5%      |         5.58 |      
+-------------+---------------+------------------+----------------+------------+--------------+      
| Round Robin | 343.4 ms      | 496.7 ms         | 198.4 ms       | 85.5%      |         5.58 |      
+-------------+---------------+------------------+----------------+------------+--------------+    

✅ All graphs generated and saved as PNG files!
