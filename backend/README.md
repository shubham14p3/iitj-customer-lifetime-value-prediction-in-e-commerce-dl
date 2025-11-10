# Customer Lifetime Value (CLV) Prediction Model

A deep learning model implemented in PyTorch for predicting customer lifetime value in e-commerce applications.

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── clv_model.py          # PyTorch model architectures (Feedforward & LSTM)
│   └── saved_model.pth       # Trained model checkpoint
├── data/
│   ├── __init__.py
│   └── data_loader.py        # Data loading and preprocessing utilities
├── utils/
│   ├── __init__.py
│   ├── trainer.py            # Training utilities and trainer class
│   └── evaluator.py          # Evaluation metrics and visualization
├── scripts/
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   └── process_olist_data.py # Olist dataset processing script
├── requirements.txt          # Python dependencies
├── README.md                 # This file (complete documentation)
├── report.md                 # Project report (4 pages)
├── app.py                    # Flask web application (optional)
├── templates/                 # Web interface templates (optional)
├── static/                    # CSS and JavaScript (optional)
└── .gitignore                # Git ignore file
```


## License

This project is for educational purposes.

## Contact

For questions or issues, please refer to the project report or contact the development team.

