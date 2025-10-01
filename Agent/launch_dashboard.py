#!/usr/bin/env python3
"""
Dashboard Launcher - Integrated with Agent Workflow
Usage: 
  python launch_dashboard.py                    # Start dashboard only
  python launch_dashboard.py --after-agent     # Start dashboard after agent run
  python launch_dashboard.py --file "Mails.csv"  # Run agent on file, then show dashboard
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_agent_on_file(file_path, context=None):
    """Run the agent on a specific file"""
    print(f"🤖 Running agent analysis on: {file_path}")
    
    # Check if file exists in the workspace root
    workspace_root = Path(__file__).parent.parent
    full_file_path = workspace_root / file_path
    
    if not full_file_path.exists():
        print(f"❌ File not found: {full_file_path}")
        return False
    
    try:
        # Import and run the agent
        from agent import build_graph
        from tools import database_ingestion_orchestrator
        
        print("📋 Starting database ingestion orchestrator...")
        
        # Run the orchestrator tool directly
        result = database_ingestion_orchestrator.invoke({
            "file_path": str(full_file_path),
            "user_context": context,
            "table_name_preference": None
        })
        
        print("✅ Agent analysis completed!")
        print(f"📄 Analysis result saved to results_for_agent/")
        return True
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return False

def start_dashboard():
    """Start the dashboard"""
    try:
        from dashboard import start_dashboard
        print("🚀 Launching dashboard...")
        start_dashboard()
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Data Mapping Dashboard Launcher")
    parser.add_argument("--file", help="File to analyze with agent before launching dashboard")
    parser.add_argument("--context", help="Context to provide to the agent")
    parser.add_argument("--after-agent", action="store_true", help="Start dashboard after running agent")
    parser.add_argument("--port", type=int, default=8050, help="Port for dashboard (default: 8050)")
    
    args = parser.parse_args()
    
    print("🎯 Data Mapping Dashboard Launcher")
    print("=" * 50)
    
    # If file specified, run agent first
    if args.file:
        success = run_agent_on_file(args.file, args.context)
        if not success:
            print("❌ Agent run failed. Dashboard will show existing analyses.")
        else:
            print("⏳ Waiting 2 seconds for file system to update...")
            time.sleep(2)
    
    # Start dashboard
    print("\n🚀 Starting Dashboard...")
    start_dashboard()

if __name__ == "__main__":
    main()