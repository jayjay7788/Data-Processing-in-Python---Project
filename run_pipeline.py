from src import download_data
from src import data_processing
from src import train_pipeline

def run_all():
    print("--------------------------------------------------")
    print("Starting Data Pipeline...")
    print("--------------------------------------------------")

    # Step 1: Downloading
    print("\n[1/3] Step 1: Downloading Raw Data")
    dl_choice = input("Download fresh raw data from CHMI? (y/n) [Default: n]: ").strip().lower()
    if dl_choice in ['y', 'yes']:
        print("Downloading raw data...")
        download_data.main()
    else:
        print("Skipping download. (Using existing raw data)")

    # Step 2: Processing
    print("\n[2/3] Step 2: Processing and Cleaning Data")
    pr_choice = input("Re-process the raw data? (y/n) [Default: n]: ").strip().lower()
    if pr_choice in ['y', 'yes']:
        print("Processing raw data...")
        data_processing.main()
    else:
        print("Skipping processing. (Using existing processed_data.csv)")

    # Step 3: Training
    print("\n[3/3] Step 3: Training Random Forest Model")
    tr_choice = input("Re-train the machine learning model? (y/n) [Default: n]: ").strip().lower()
    if tr_choice in ['y', 'yes']:
        print("Training model...")
        train_pipeline.main()
    else:
        print("Skipping training. (Using existing compressed model bundle)")

    print("Pipeline Complete!")
    print("👉 You can now launch your dashboard by typing: streamlit run app.py")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_all()