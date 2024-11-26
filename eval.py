import os
import pandas as pd
import keras
from datetime import datetime

def analyze_model_files(directory):
    keras_files = [f for f in os.listdir(directory) if f.endswith('.keras')]
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    print("Model Files Analysis:")
    print("-------------------")
    
    for keras_file in keras_files:
        try:
            # Load the model
            model_path = os.path.join(directory, keras_file)
            model = keras.models.load_model(model_path)
            
            # Find corresponding CSV file
            matching_csv = [csv for csv in csv_files if keras_file.split('.')[0] in csv]
            
            if matching_csv:
                csv_path = os.path.join(directory, matching_csv[0])
                df = pd.read_csv(csv_path)
                
                print(f"\nModel File: {keras_file}")
                print(f"CSV Log: {matching_csv[0]}")
                print("Final Training Metrics:")
                print(f"- Final Training Loss: {df['loss'].iloc[-1]:.4f}")
                print(f"- Final Training Accuracy: {df['accuracy'].iloc[-1]:.4f}")
                print(f"- Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
                print(f"- Final Validation Accuracy: {df['val_accuracy'].iloc[-1]:.4f}")
                
                # File creation time
                file_ctime = os.path.getctime(model_path)
                print(f"Created: {datetime.fromtimestamp(file_ctime)}")
            
            # Model summary
            print("\nModel Summary:")
            model.summary()
            
        except Exception as e:
            print(f"Error processing {keras_file}: {e}")

# Run the analysis
analyze_model_files('.')



