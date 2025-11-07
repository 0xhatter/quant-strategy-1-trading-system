"""
Setup Verification Script
Checks if all dependencies are installed and modules can be imported.
"""

import sys

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 70)
    print(" " * 20 + "SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    print("Checking required packages...")
    print("-" * 70)
    
    for import_name, package_name in required_packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name:20s} {version}")
        except ImportError:
            print(f"✗ {package_name:20s} NOT INSTALLED")
            missing_packages.append(package_name)
    
    print()
    
    if missing_packages:
        print("❌ Missing packages:", ", ".join(missing_packages))
        print()
        print("To install missing packages, run:")
        print(f"  pip install {' '.join(missing_packages)}")
        print()
        return False
    
    print("✅ All dependencies are installed!")
    print()
    
    return True


def check_project_files():
    """Check if all project files exist."""
    import os
    
    print("Checking project files...")
    print("-" * 70)
    
    required_files = [
        'data_collection.py',
        'asset_selection.py',
        'feature_engineering.py',
        'ml_models.py',
        'risk_management.py',
        'backtesting.py',
        'main_trading_system.py',
        'example_workflow.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} NOT FOUND")
            missing_files.append(filename)
    
    print()
    
    if missing_files:
        print("❌ Missing files:", ", ".join(missing_files))
        print()
        return False
    
    print("✅ All project files are present!")
    print()
    
    return True


def check_imports():
    """Check if project modules can be imported."""
    print("Checking project modules...")
    print("-" * 70)
    
    modules = [
        'data_collection',
        'asset_selection',
        'feature_engineering',
        'ml_models',
        'risk_management',
        'backtesting',
        'main_trading_system'
    ]
    
    failed_imports = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name} - {str(e)[:50]}")
            failed_imports.append(module_name)
    
    print()
    
    if failed_imports:
        print("❌ Some modules failed to import:", ", ".join(failed_imports))
        print()
        return False
    
    print("✅ All project modules can be imported!")
    print()
    
    return True


def main():
    """Run all verification checks."""
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("=" * 70)
        print("Please install missing dependencies before proceeding.")
        print("=" * 70)
        sys.exit(1)
    
    # Check project files
    files_ok = check_project_files()
    
    if not files_ok:
        print("=" * 70)
        print("Some project files are missing.")
        print("=" * 70)
        sys.exit(1)
    
    # Check imports
    imports_ok = check_imports()
    
    if not imports_ok:
        print("=" * 70)
        print("Some modules failed to import. Check error messages above.")
        print("=" * 70)
        sys.exit(1)
    
    # All checks passed
    print("=" * 70)
    print(" " * 15 + "🎉 SETUP VERIFICATION COMPLETE! 🎉")
    print("=" * 70)
    print()
    print("Your quantitative trading system is ready to use!")
    print()
    print("Next steps:")
    print("  1. Run the example workflow:")
    print("     python3 example_workflow.py")
    print()
    print("  2. Or test individual modules:")
    print("     python3 data_collection.py")
    print("     python3 asset_selection.py")
    print("     python3 ml_models.py")
    print()
    print("  3. Read the documentation:")
    print("     - README.md for comprehensive guide")
    print("     - QUICK_START.md for quick reference")
    print("     - PROJECT_SUMMARY.md for overview")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
