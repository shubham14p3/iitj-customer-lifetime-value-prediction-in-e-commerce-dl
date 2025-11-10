"""
Flask Web Application for Customer Lifetime Value (CLV) Prediction
This application provides a web interface for the CLV prediction model.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from flask_cors import CORS  

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.clv_model import get_model
from data.data_loader import CLVDataProcessor
from utils.evaluator import CLVEvaluator

# -------------------------------------------------------------------
# Flask app setup
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://13.50.9.79:5173"
]}})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['SECRET_KEY'] = 'clv_prediction_secret_key_2024'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# -------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------
model = None
processor = None
model_loaded = False
model_path = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------------------------------------------
# Helper: Load model
# -------------------------------------------------------------------
def load_model_from_checkpoint(model_path, device='cpu'):
    global model, model_loaded

    try:
        checkpoint = torch.load(model_path, map_location=device)

        if 'model_config' in checkpoint:
            cfg = checkpoint['model_config']
            model_type = cfg.get('model_type', 'feedforward')
            input_dim = cfg.get('input_dim')
            hidden_dims = cfg.get('hidden_dims', [128, 256, 128, 64])
            dropout_rate = cfg.get('dropout_rate', 0.3)
        else:
            model_type = 'feedforward'
            input_dim = None
            hidden_dims = [128, 256, 128, 64]
            dropout_rate = 0.3

        if input_dim is None:
            raise ValueError("Model configuration missing input_dim. Please retrain the model.")

        if model_type == 'feedforward':
            model_instance = get_model(
                model_type='feedforward',
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
        else:
            model_instance = get_model(
                model_type='lstm',
                input_dim=input_dim,
                hidden_dim=hidden_dims[0] if hidden_dims else 128,
                num_layers=2,
                dropout_rate=dropout_rate
            )

        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.to(device)
        model_instance.eval()

        model = model_instance
        model_loaded = True
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

# -------------------------------------------------------------------
# Helper: Create prediction plot
# -------------------------------------------------------------------
def create_prediction_plot(predictions, actuals=None, save_path=None):
    plt.figure(figsize=(10, 6))

    if actuals is not None:
        plt.scatter(actuals, predictions, alpha=0.5, s=20)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()],
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual CLV', fontsize=12)
        plt.ylabel('Predicted CLV', fontsize=12)
        plt.title('Predicted vs Actual CLV', fontsize=14, fontweight='bold')

        from sklearn.metrics import r2_score
        r2 = r2_score(actuals, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        plt.hist(predictions, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Predicted CLV', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Predicted CLV', fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    if actuals is not None:
        plt.legend()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return img_base64

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route('/')
def index():
    """Health endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

# ------------------------ LOAD MODEL ------------------------
@app.route('/load_model', methods=['POST'])
def load_model_route():
    global model, processor, model_loaded, model_path

    try:
        model_file = request.files.get('model_file')
        if not model_file:
            default_paths = ['models/saved_model.pth', 'models/test_model.pth']
            current_model_path = next((p for p in default_paths if os.path.exists(p)), None)
            if not current_model_path:
                return jsonify({
                    'success': False,
                    'message': 'No model file provided and no default model found.'
                })
            model_path = current_model_path
        else:
            model_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                      f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            model_file.save(model_path)

        success, message = load_model_from_checkpoint(model_path, device)
        if success:
            processor = CLVDataProcessor()
            # Always fit processor on main training data
            training_data_paths = ['data/clv_features.csv', 'data/clv_features_sample.csv']
            for path in training_data_paths:
                if os.path.exists(path):
                    train_df = pd.read_csv(path)
                    processor.preprocess_data(train_df)
                    print(f"✓ Processor fitted on training data ({path}) with {len(processor.feature_names)} features")
                    break
            else:
                print("⚠ Warning: No training data found to fit processor.")

            return jsonify({'success': True, 'message': message, 'device': device})
        else:
            return jsonify({'success': False, 'message': message})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ------------------------ PREDICT ------------------------
@app.route('/predict', methods=['POST'])
def predict():
    global model, processor, model_loaded

    if not model_loaded:
        return jsonify({'success': False, 'message': 'Model not loaded'})

    try:
        file = request.files.get('data_file')
        if not file:
            return jsonify({'success': False, 'message': 'No file uploaded'})

        df = pd.read_csv(file)
        has_target = 'clv' in df.columns

        if processor is None:
            processor = CLVDataProcessor()

        # Ensure processor is fitted
        if not hasattr(processor, 'feature_names') or processor.feature_names is None:
            training_data_paths = ['data/clv_features.csv', 'data/clv_features_sample.csv']
            for path in training_data_paths:
                if os.path.exists(path):
                    train_df = pd.read_csv(path)
                    processor.preprocess_data(train_df)
                    print(f"✓ Processor fitted on training data ({path})")
                    break
            else:
                return jsonify({'success': False, 'message': 'Training data not found'})

        # Transform uploaded data
        features, targets, feature_names = processor.transform_data(df, require_target=False)

        # Make predictions
        features_tensor = torch.FloatTensor(features).to(device)
        with torch.no_grad():
            predictions = model(features_tensor).cpu().numpy().flatten()

        results_df = df.copy()
        results_df['predicted_clv'] = predictions

        # ✅ Initialize metrics before conditional
        metrics = None

        # If actual CLV values exist, compute metrics correctly
        if has_target:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # Ensure consistent scaling for targets
            if hasattr(processor, 'target_scaler'):
                try:
                    y_true = processor.inverse_transform_target(df['clv'])
                except Exception:
                    y_true = df['clv'].values
            else:
                y_true = df['clv'].values

            mse = mean_squared_error(y_true, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100

            metrics = {
                'MSE': round(mse, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2)
            }

            plot_base64 = create_prediction_plot(predictions, y_true)
        else:
            plot_base64 = create_prediction_plot(predictions)

        # Save results file
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        results_df.to_csv(save_path, index=False)

        # Summary statistics
        summary = {
            'total_customers': len(results_df),
            'mean_predicted_clv': float(np.mean(predictions)),
            'median_predicted_clv': float(np.median(predictions)),
            'min_predicted_clv': float(np.min(predictions)),
            'max_predicted_clv': float(np.max(predictions)),
            'std_predicted_clv': float(np.std(predictions)),
        }

        return jsonify({
            'success': True,
            'predictions': results_df.head(100).to_dict('records'),
            'total_rows': len(results_df),
            'metrics': metrics,
            'summary': summary,
            'plot': plot_base64,
            'download_path': filename
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'message': str(e),
            'traceback': traceback.format_exc()
        })

# ------------------------ DOWNLOAD ------------------------
@app.route('/download/<filename>')
def download(filename):
    if '/' in filename:
        filename = filename.split('/')[-1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    return jsonify({'error': 'File not found'}), 404


# ------------------------ DOWNLOAD SAMPLE FILE ------------------------
@app.route('/download_sample')
def download_sample():
    """Download sample input CSV for prediction."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_paths = [
        os.path.join(base_dir, 'data', 'clv_features_sample.csv'),
        os.path.join(base_dir, 'data', 'clv_features.csv')
    ]
    for path in sample_paths:
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name=os.path.basename(path))
    return jsonify({'error': 'No sample file found'}), 404


# ------------------------ DOWNLOAD MODEL FILE ------------------------
@app.route('/download_model')
def download_model():
    """Download current or default model."""
    global model_path
    if model_path and os.path.exists(model_path):
        return send_file(model_path, as_attachment=True, download_name=os.path.basename(model_path))
    # fallback
    default_paths = ['models/saved_model.pth', 'models/test_model.pth']
    for path in default_paths:
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name=os.path.basename(path))
    return jsonify({'error': 'No model file found'}), 404


# ------------------------ MODEL INFO ------------------------
@app.route('/model_info')
def model_info():
    global model, model_loaded
    if not model_loaded:
        return jsonify({'success': False, 'message': 'No model loaded'})
    
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info = {
            'model_type': 'feedforward',
            'input_dim': getattr(model, 'input_dim', 'N/A'),
            'hidden_dims': getattr(model, 'hidden_dims', None),
            'dropout_rate': getattr(model, 'dropout_rate', None),
            'total_parameters': f"{total:,}",
            'trainable_parameters': f"{trainable:,}",
            'device': device
        }
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# -------------------------------------------------------------------
# APP ENTRY POINT
# -------------------------------------------------------------------
if __name__ == '__main__':
    default_model = 'models/saved_model.pth'
    if not os.path.exists(default_model):
        default_model = 'models/test_model.pth'

    if os.path.exists(default_model):
        print(f"Attempting to load default model from {default_model}...")
        success, msg = load_model_from_checkpoint(default_model, device)
        print("✓" if success else "✗", msg)

        if success:
            processor = CLVDataProcessor()
            for path in ['data/clv_features.csv', 'data/clv_features_sample.csv']:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    processor.preprocess_data(df)
                    print(f"✓ Processor fitted on {path}")
                    break
    else:
        print("No default model found. Please train or upload one.")

    print(f"\nStarting Flask app at http://127.0.0.1:5000 (Device: {device})")
    app.run(debug=True, host='0.0.0.0', port=5000)
