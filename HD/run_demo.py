#!/usr/bin/env python3
import sys
import os
import asyncio

# Fix the path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
hd_path = os.path.join(project_root, "HD")

# Add both paths to ensure imports work
sys.path.insert(0, project_root)
sys.path.insert(0, hd_path)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    try:
        from HD.demo.main import DemoApplication
        asyncio.run(DemoApplication.run())
    except KeyboardInterrupt:
        print("\nDemo terminated by user")
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()