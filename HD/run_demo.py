#!/usr/bin/env python3
import sys
import os
import asyncio

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.abspath(r"c:/Users/ymeny/Desktop/HADER/HD"))

# Add the project root to the Python path
sys.path.insert(0, project_root)

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
    main();
