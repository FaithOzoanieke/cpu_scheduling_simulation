"""
Wrapper script to run the complete simulation
Author: Faith Ujunwa Ozoanieke
Student ID: 251230004
"""

import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Requirements installed")

def run_main_simulation():
    """Run the main CPU scheduling simulation"""
    print("\nRunning CPU scheduling simulation...")
    from cpu_scheduling_simulation import main
    main()

def run_graph_generation():
    """Generate missing graphs"""
    print("\nGenerating additional graphs...")
    from generate_missing_graphs import generate_gantt_charts, generate_radar_chart
    generate_gantt_charts()
    generate_radar_chart()

if __name__ == "__main__":
    print("=" * 70)
    print("CPU Scheduling Simulation Suite")
    print("Nile University of Nigeria - CSC 805 Advanced Operating Systems")
    print("=" * 70)
    
    try:
        # Optionally install requirements
        # install_requirements()
        
        # Run main simulation
        run_main_simulation()
        
        # Generate additional graphs
        run_graph_generation()
        
        print("\n" + "=" * 70)
        print("All tasks completed successfully!")
        print("Files available in current directory:")
        print("  1. performance_comparison.png")
        print("  2. gantt_charts.png")
        print("  3. radar_chart.png")
        print("  4. process_data.csv")
        print("  5. simulation_results.csv")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure required packages are installed: pip install numpy matplotlib pandas tabulate")
        print("  2. Check Python version (3.8+ recommended)")
        print("  3. Ensure you have write permissions in the current directory")
        sys.exit(1)