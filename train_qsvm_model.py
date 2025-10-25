import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def generate_training_data(n_samples: int = 500):
    """
    Generates synthetic data that mimics the characteristics of quantum device metrics
    to train a fitness prediction model.
    """
    np.random.seed(42)
    
    # Core device metrics
    error_rate = np.random.uniform(0.001, 0.08, n_samples)
    readout_error = np.random.uniform(0.01, 0.1, n_samples)
    t1_time = np.random.uniform(70, 250, n_samples)
    t2_time = np.random.uniform(50, 200, n_samples)
    pending_jobs = np.random.randint(0, 800, n_samples)
    
    # Engineered features (often used in model input)
    gate_fidelity = 1 - error_rate
    readout_fidelity = 1 - readout_error
    
    # --- Define the Target Variable: 'fitness_score' ---
    # This formula creates a score from 1-10 where high scores are better.
    # A good device has low errors, high coherence times, and a short queue.
    fidelity_component = (gate_fidelity * 0.7 + readout_fidelity * 0.3) * 5
    coherence_component = ((t1_time / 250) * 0.6 + (t2_time / 200) * 0.4) * 4
    queue_penalty = (pending_jobs / 800) * 2.5
    
    # Introduce some noise to make the relationship more realistic for the model
    noise = np.random.normal(0, 0.3, n_samples)
    
    fitness_score = fidelity_component + coherence_component - queue_penalty + noise
    # Clip the score to ensure it stays within the 1-10 range
    fitness_score = np.clip(fitness_score, 1, 10)
    
    # Create the DataFrame for training
    df = pd.DataFrame({
        'error_rate': error_rate,
        'readout_error': readout_error,
        't1_time': t1_time,
        't2_time': t2_time,
        'gate_fidelity': gate_fidelity,
        'readout_fidelity': readout_fidelity,
        'pending_jobs': pending_jobs,
        'fitness_score': fitness_score # This is our target 'y' value
    })
    
    return df

def train_and_save_model():
    """
    Trains an SVR model on synthetic data and saves it to a pickle file.
    """
    print("🚀 Starting model training process...")
    
    # 1. Generate the training data
    training_df = generate_training_data()
    # Define the features your main application will provide to the model
    features = [
        'error_rate', 'readout_error', 't1_time', 't2_time',
        'gate_fidelity', 'readout_fidelity', 'pending_jobs'
    ]
    target = 'fitness_score'
    
    X = training_df[features]
    y = training_df[target]
    
    print(f"✅ Generated {len(training_df)} samples for training.")
    
    # 2. Define the Model Pipeline
    # A pipeline is a best practice that bundles a pre-processing step (like scaling)
    # with the actual model. SVR (Support Vector Regressor) is a good classical
    # substitute for a QSVM for this type of prediction task.
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVR(kernel='rbf', C=1.0, epsilon=0.2))
    ])
    
    # 3. Train the Model on the data
    print("⏳ Training the SVM model...")
    model_pipeline.fit(X, y)
    print("✅ Model training complete.")
    
    # 4. Save the trained pipeline to the final file
    model_filename = 'qsvm_fitness_model.pkl'
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model_pipeline, f)
        print(f"💾 Model successfully saved to '{model_filename}'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

if __name__ == '__main__':
    train_and_save_model()