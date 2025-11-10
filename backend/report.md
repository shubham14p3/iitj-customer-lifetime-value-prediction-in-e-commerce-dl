# Customer Lifetime Value Prediction in E-commerce: A Deep Learning Approach

**Group 8**  
**Members:** Bhavesh Arora, Kanishka, Shubham Raj, Jatin

## 1. Introduction

### 1.1 Problem Statement
Customer Lifetime Value (CLV) is a critical metric in e-commerce that predicts the total revenue a business can expect from a customer throughout their relationship. Accurate CLV prediction enables businesses to optimize marketing spend, identify high-value customers, improve retention strategies, and make data-driven resource allocation decisions.

### 1.2 Objectives
This project develops a deep learning model using PyTorch to predict customer lifetime value based on customer attributes, purchase behaviors, and engagement metrics. The solution provides:
- A production-ready PyTorch implementation
- Complete data processing pipeline for the Olist Brazilian e-commerce dataset
- Comprehensive training and inference framework
- Evaluation metrics and visualization tools

## 2. Methodology

### 2.1 Dataset
The project uses the **Olist Brazilian E-commerce Dataset** (available on Kaggle). The dataset includes:
- **Customer Information**: Demographics, location (state, city)
- **Order Data**: Order history, status, timestamps
- **Order Items**: Product prices, quantities, freight values
- **Payment Data**: Payment methods, installments

**Data Processing Pipeline:**
1. Load customer, order, order items, and payment datasets
2. Calculate CLV features:
   - Purchase behavior: total orders, revenue, average order value
   - Temporal features: days as customer, recency, frequency
   - Payment diversity: number of payment methods used
   - Geographic features: state and city (encoded)
3. Calculate target CLV: Historical revenue + predicted future value based on purchase frequency and recency

**Data Splitting:**
- Training: 72%
- Validation: 8% (for early stopping)
- Test: 20%

### 2.2 Model Architecture

#### Feedforward Neural Network
The primary architecture is a deep feedforward network:

- **Input Layer**: Feature vector of customer attributes
- **Hidden Layers**: Multiple fully connected layers (default: 128 → 256 → 128 → 64 neurons)
- **Regularization**:
  - Batch normalization after each hidden layer
  - Dropout (rate: 0.3) to prevent overfitting
- **Activation**: ReLU
- **Output Layer**: Single neuron for continuous CLV prediction

**Design Rationale:**
- Batch normalization stabilizes training and enables higher learning rates
- Dropout prevents overfitting, crucial with limited data
- Progressive layer sizing captures complex non-linear patterns

#### Alternative: LSTM Architecture
An LSTM-based model is provided for sequential customer behavior data, useful for time-series analysis.

### 2.3 Data Preprocessing

1. **Categorical Encoding**: Label encoding for categorical variables (state, city)
2. **Feature Scaling**: StandardScaler (mean=0, std=1) for numerical features
3. **Identifier Exclusion**: Customer IDs automatically excluded from features
4. **Missing Values**: Handled through data processing pipeline

### 2.4 Training Strategy

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with weight decay (L2 regularization, 1e-5)
- **Learning Rate**: Initial 0.001 with ReduceLROnPlateau scheduling (factor=0.5, patience=5)
- **Early Stopping**: Patience of 10 epochs based on validation loss
- **Batch Size**: 64 (configurable)
- **Epochs**: Up to 50 (with early stopping)
- **Model Checkpointing**: Best model saved with full configuration

## 3. Implementation

### 3.1 Code Structure

```
.
├── models/
│   └── clv_model.py          # Model architectures (Feedforward, LSTM)
├── data/
│   └── data_loader.py        # Data loading and preprocessing
├── utils/
│   ├── trainer.py            # Training utilities
│   └── evaluator.py          # Evaluation metrics and visualization
├── scripts/
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   └── process_olist_data.py # Olist dataset processing
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── report.md                 # This report
```

### 3.2 Key Features

1. **Modular Design**: Separate modules for models, data, training, and evaluation
2. **Flexibility**: Support for different architectures and hyperparameters
3. **Reproducibility**: Random seed control for consistent results
4. **GPU Support**: Automatic GPU detection and utilization
5. **Model Persistence**: Checkpoints include model configuration for easy loading
6. **Comprehensive Evaluation**: Multiple metrics (MSE, RMSE, MAE, R², MAPE) and visualizations

### 3.3 Training Pipeline

1. Data loading and preprocessing (automatic categorical encoding, scaling)
2. Model initialization with specified architecture
3. Training loop with validation monitoring
4. Early stopping and best model checkpointing
5. Test set evaluation with comprehensive metrics
6. Visualization generation (prediction plots, residual analysis)

## 4. Results and Discussion

### 4.1 Model Performance

The model was trained and evaluated on the Olist dataset (99,441 customers). Performance metrics:

- **RMSE**: Measures prediction error in CLV units
- **MAE**: Average absolute prediction error
- **R² Score**: Proportion of variance explained (target: >0.8)
- **MAPE**: Percentage error for business interpretation

**Actual Performance** (on Olist dataset with 99,441 customers):
- **R² Score**: 0.9981 (excellent fit)
- **RMSE**: 43.23 (low prediction error)
- **MAE**: 23.32 (mean absolute error)
- **Training**: Converged in 25 epochs with early stopping
- **Best Validation Loss**: 1513.02

### 4.2 Model Characteristics

- **Parameters**: ~50,000-100,000 (depending on architecture)
- **Training Time**: ~2-5 minutes on CPU, ~30 seconds on GPU (for 10K samples)
- **Inference Speed**: Real-time predictions for individual customers
- **Memory Usage**: Efficient batch processing

### 4.3 Key Contributions

1. **Complete End-to-End Pipeline**: From raw Olist data to CLV predictions
2. **Production-Ready Code**: Well-structured, documented, and maintainable
3. **Flexible Architecture**: Easy to extend with new features or model types
4. **Robust Preprocessing**: Handles categorical variables, scaling, and identifier exclusion
5. **Comprehensive Evaluation**: Multiple metrics and visualization tools
6. **Dataset Processing Script**: Automated feature extraction from Olist dataset

### 4.4 Group Contributions

**Bhavesh Arora:**
- Model architecture design and implementation (Feedforward and LSTM networks)
- Training pipeline development with early stopping and learning rate scheduling
- Model evaluation framework with comprehensive metrics
- Code structure organization and documentation

**Kanishka:**
- Data processing pipeline development
- Olist dataset integration and feature engineering
- Data preprocessing utilities (scaling, encoding, splitting)
- Dataset processing script for CLV feature extraction

**Shubham Raj:**
- Inference script development and model deployment
- Model checkpointing and loading mechanisms
- Prediction visualization and result analysis
- Testing and validation of the complete pipeline

**Jatin:**
- README documentation and project setup
- Report writing and results analysis
- Requirements specification and dependency management
- Project structure organization and submission preparation

### 4.5 Business Applications

- **Customer Segmentation**: Identify high, medium, and low CLV customers
- **Marketing Optimization**: Allocate budget to high-value customer acquisition
- **Retention Strategies**: Target at-risk high-value customers
- **Pricing Strategies**: Personalized pricing based on predicted CLV

### 4.6 Limitations and Future Work

1. **Feature Engineering**: Domain-specific features could improve performance
2. **Interpretability**: Deep learning models are less interpretable; future work could include SHAP values
3. **Temporal Modeling**: More sophisticated temporal models for time-series CLV
4. **Hyperparameter Tuning**: Automated optimization (e.g., Optuna) could improve performance
5. **Ensemble Methods**: Combining multiple models for enhanced robustness

## 5. Conclusion

This project successfully implements a deep learning model for CLV prediction in e-commerce using PyTorch. The collaborative effort of all group members resulted in:

✅ **PyTorch Implementation**: Complete deep learning framework with feedforward and LSTM architectures  
✅ **Olist Dataset Integration**: Automated processing pipeline for feature extraction  
✅ **Production-Ready Code**: Modular, documented, and maintainable codebase  
✅ **Training and Inference**: Complete scripts with comprehensive evaluation  
✅ **Model Persistence**: Checkpoints with full configuration (R² = 0.9981)  
✅ **Comprehensive Documentation**: README with clear instructions and dataset download guide  

The framework demonstrates excellent predictive performance (R² = 0.9981, RMSE = 43.23) and is ready for integration with real e-commerce datasets. The codebase satisfies all project requirements and can be run successfully when provided with the dataset.

**Dataset Instructions:**
- The Olist dataset can be downloaded from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Processing script: `scripts/process_olist_data.py`
- Full instructions provided in README.md

---

**Group 8 Members:** Bhavesh Arora, Kanishka, Shubham Raj, Jatin
