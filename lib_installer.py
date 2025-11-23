import subprocess
import sys

def check_and_install_libraries():
    """
    Simple function to check and install required libraries.
    """
    libraries = ['numpy', 'matplotlib', 'pandas']
    
    print("Checking libraries...")
    print()
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"{lib} is installed")
        except ImportError:
            print(f"Installing {lib}...", end=" ", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print("Done")
    
    print()
    print("All libraries ready")
