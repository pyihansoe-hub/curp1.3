"""
test_installation.py
Currency Recognition System - Installation Test Script
Verifies all components are properly installed and working
"""

import sys
import os
import importlib
import subprocess
import platform

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print_info(f"Python version: {sys.version}")
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} is below minimum required 3.8")
        return False

def check_pip():
    """Check if pip is installed"""
    try:
        import pip
        print_success(f"pip version: {pip.__version__}")
        return True
    except ImportError:
        print_error("pip is not installed")
        return False

def check_virtual_env():
    """Check if running in virtual environment"""
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print_success(f"Running in virtual environment: {sys.prefix}")
    else:
        print_warning("Not running in virtual environment (recommended)")
    return in_venv

def check_package(package_name, min_version=None):
    """Check if a package is installed and version"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif package_name == 'cv2':
            version = module.__version__
        else:
            version = "unknown"
        
        if min_version:
            from packaging import version as version_parser
            if version_parser.parse(version) >= version_parser.parse(min_version):
                print_success(f"{package_name} {version} (✓ meets minimum {min_version})")
            else:
                print_warning(f"{package_name} {version} (⬇ below recommended {min_version})")
        else:
            print_success(f"{package_name} {version}")
        return True
    except ImportError as e:
        print_error(f"{package_name} not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking {package_name}: {e}")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow has GPU support"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_success(f"TensorFlow GPU support: Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print_info(f"  GPU {i}: {gpu.name}")
        else:
            print_warning("TensorFlow running on CPU (GPU not available)")
        return True
    except:
        return False

def check_camera():
    """Check if camera is accessible"""
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
            if ret:
                print_success("Camera is accessible and working")
                print_info(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print_warning("Camera opened but cannot read frames")
            camera.release()
            return True
        else:
            print_warning("Camera not accessible (may be in use or not connected)")
            return False
    except Exception as e:
        print_warning(f"Camera check failed: {e}")
        return False

def check_folders():
    """Check required folders exist"""
    required_folders = ['dataset', 'models', 'templates', 'captured_images']
    optional_folders = ['static', 'results', 'static/img']
    
    print_info("\nChecking required folders:")
    all_ok = True
    for folder in required_folders:
        if os.path.exists(folder):
            if os.path.isdir(folder):
                print_success(f"Folder '{folder}' exists")
                # Count files in dataset subfolders
                if folder == 'dataset':
                    classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
                    print_info(f"  Found {len(classes)} class folders in dataset")
                    for cls in classes[:3]:  # Show first 3
                        class_path = os.path.join(folder, cls)
                        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        print_info(f"    {cls}: {len(images)} images")
            else:
                print_error(f"'{folder}' exists but is not a folder")
                all_ok = False
        else:
            print_warning(f"Folder '{folder}' not found (will be created when needed)")
    
    print_info("\nChecking optional folders:")
    for folder in optional_folders:
        if os.path.exists(folder):
            print_success(f"Folder '{folder}' exists")
    
    return all_ok

def check_model_files():
    """Check if model files exist"""
    print_info("\nChecking model files:")
    
    model_path = 'models/currency_model.h5'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print_success(f"Model file found: {model_path} ({size_mb:.1f} MB)")
    else:
        print_warning(f"Model file not found. Run train_advanced.py first")
    
    class_names_path = 'models/class_names.txt'
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print_success(f"Class names file found with {len(classes)} classes: {classes}")
    else:
        print_warning(f"Class names file not found")

def check_requirements_file():
    """Check if requirements.txt exists"""
    if os.path.exists('requirements.txt'):
        print_success("requirements.txt found")
        # Count packages
        with open('requirements.txt', 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print_info(f"  Contains {len(packages)} packages")
        return True
    else:
        print_error("requirements.txt not found")
        return False

def check_app_file():
    """Check if app.py exists"""
    if os.path.exists('app.py'):
        print_success("app.py found")
        return True
    else:
        print_error("app.py not found")
        return False

def check_training_script():
    """Check if training script exists"""
    if os.path.exists('train_advanced.py'):
        print_success("train_advanced.py found")
        return True
    else:
        print_warning("train_advanced.py not found (training will not be possible)")
        return False

def check_port_availability(port=5000):
    """Check if port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result != 0:
        print_success(f"Port {port} is available")
        return True
    else:
        print_warning(f"Port {port} is already in use")
        return False

def system_info():
    """Display system information"""
    print_info(f"\nSystem: {platform.system()} {platform.release()}")
    print_info(f"Machine: {platform.machine()}")
    print_info(f"Processor: {platform.processor()}")
    print_info(f"Current directory: {os.getcwd()}")

def main():
    """Main test function"""
    print_header("CURRENCY RECOGNITION SYSTEM - INSTALLATION TEST")
    print_info("Testing all components...\n")
    
    # System info
    system_info()
    
    # Python environment
    print_header("1. PYTHON ENVIRONMENT")
    python_ok = check_python_version()
    pip_ok = check_pip()
    venv_ok = check_virtual_env()
    
    # Core packages
    print_header("2. CORE PACKAGES")
    packages_ok = []
    packages_ok.append(check_package('tensorflow', '2.10.0'))
    packages_ok.append(check_package('keras', '2.10.0'))
    packages_ok.append(check_package('cv2', '4.5.0'))  # OpenCV
    packages_ok.append(check_package('PIL', '9.0.0'))  # Pillow
    packages_ok.append(check_package('numpy', '1.21.0'))
    packages_ok.append(check_package('flask', '2.3.0'))
    packages_ok.append(check_package('flask_cors'))
    packages_ok.append(check_package('matplotlib', '3.5.0'))
    
    # TensorFlow GPU
    check_tensorflow_gpu()
    
    # Project structure
    print_header("3. PROJECT STRUCTURE")
    folders_ok = check_folders()
    req_ok = check_requirements_file()
    app_ok = check_app_file()
    train_ok = check_training_script()
    check_model_files()
    
    # Hardware
    print_header("4. HARDWARE CHECK")
    camera_ok = check_camera()
    port_ok = check_port_availability()
    
    # Summary
    print_header("INSTALLATION TEST SUMMARY")
    
    all_checks = [
        python_ok,
        pip_ok,
        all(packages_ok),
        folders_ok,
        req_ok,
        app_ok,
        camera_ok
    ]
    
    passed = sum(1 for check in all_checks if check)
    total = len(all_checks)
    
    print_info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print_success("\n✓ ALL CHECKS PASSED!")
        print_info("Your Currency Recognition System is ready to use!")
        print_info("\nNext steps:")
        print_info("1. Train the model: python train_advanced.py")
        print_info("2. Run the app: python app.py")
        print_info("3. Open browser: http://localhost:5000")
    else:
        print_warning(f"\n⚠️  {total - passed} checks failed. Please fix the issues above.")
        
        if not python_ok:
            print_info("  - Install Python 3.8 or higher")
        if not pip_ok:
            print_info("  - Install pip: python -m ensurepip")
        if not all(packages_ok):
            print_info("  - Run: pip install -r requirements.txt")
        if not folders_ok:
            print_info("  - Create missing folders or run app.py to auto-create them")
        if not req_ok:
            print_info("  - requirements.txt is missing")
        if not app_ok:
            print_info("  - app.py is missing")
        if not camera_ok:
            print_info("  - Check camera connection and permissions")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("\n\nTest interrupted by user")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")