
if __name__ == "__main__":
    # Add parent directory to path for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Now run test
    test_env_registration()