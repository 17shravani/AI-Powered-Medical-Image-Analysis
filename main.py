import os
import sys

def main():
    print("="*60)
    print("   AI-POWERED MEDICAL IMAGE ANALYSIS PROJECT - ORCHESTRATOR")
    print("="*60)
    print("Select an action to perform:")
    print("1. Setup Dummy Data (Simulate obtaining medical dataset)")
    print("2. Train AI Model (CNN) on Data")
    print("3. Evaluate Model (Generate Accuracy & Confusion Matrix)")
    print("4. Launch Web API (Flask Server for Doctor's UI)")
    print("5. Exit")
    print("="*60)
    
    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        from src.data_generation import generate_dummy_data
        generate_dummy_data()
    elif choice == '2':
        from src.train import train_model
        train_model()
    elif choice == '3':
        from src.evaluate import evaluate_model
        evaluate_model()
    elif choice == '4':
        print("\nStarting the Flask Web API... Please visit http://127.0.0.1:5000 in your browser to test.")
        # Execute the flask app
        os.system(f"{sys.executable} src/api.py")
    elif choice == '5':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Please run python main.py again.")

if __name__ == "__main__":
    main()
