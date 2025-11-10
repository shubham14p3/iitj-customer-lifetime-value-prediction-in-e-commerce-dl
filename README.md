#Customer Lifetime Value (CLV) Modeling
EV Charging Station Demand Forecasting

[![Contributors][contributors-shield]][contributors-url]  
[![Forks][forks-shield]][forks-url]  
[![Stargazers][stars-shield]][stars-url]  
![Issues][issues-shield]

---

## Overview

This project focuses on Forecast electricity consumption and session demand for Electric Vehicle (EV) charging stations using advanced time series modeling. This assists operators in optimizing load distribution and future planning.

---

## Requirements
- **Python 3.10+**
- **Node.js 18+**
- **Flask** for backend
- **React.js** for frontend
- **Redux Toolkit** for state management
- **Material-UI** for UI components

---

## Live Project Links
- **UI:** [http://51.20.36.32:5173/login/](http://51.20.36.32:5173/)
- **Backend:** [http://51.20.36.32:8000/raw](http://51.20.36.32:8000/)

User Name : **admin** || Password: **admin**  (Default)
---

## Setup Instructions

### Backend Setup

#### Step 1: Create and Activate Python Virtual Environment

1. **Create a Virtual Environment**:
    ```bash
    python3.10 -m venv veenv
    ```

2. **Activate Virtual Environment**:
   - **Command Prompt**:
     ```bash
     venv\Scripts\activate
     ```
   - **PowerShell**:
     ```bash
     .\venv\Scripts\Activate
     ```
   - **Git Bash**:
     ```bash
     source venv/Scripts/activate
     ```

# Install requirements
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

#### Step 2: Install Python Dependencies
Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

#### Step 3: Run the uvicorn Backend
Start the Uvicorn app:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

### Frontend Setup

#### Step 1: Install Node.js Dependencies
Navigate to the project root directory and install the required packages:
```bash
npm install
```

#### Step 2: Run the Frontend Application
Start the development server:
```bash
npm run dev
```

---

## Folder Structure
```
project-root/
├── backend/                # Backend code
│   ├── app.py              # App
│   ├── data/               # Dataset folder
│   ├── requirements.txt    # Python dependencies
├── src/                    # Frontend React code
│   ├── libs/               # React components
│   ├── pages/              # Layout components
│   ├── App.jsx             # Main application file
│   ├── main.jsx            # Entry point for React
├── assets/                 # Static assets
├── README.md               # Project documentation
├── package.json            # Node.js dependencies
├── vite.config.js          # Vite configuration
```
---

## Features

### Backend
- **uvicorn API** for data processing and machine learning predictions.
- **Endpoints** for:
  • /ingest-json: Flatten and save ACN JSON.
  • /clean: Aggregate and clean data.
  • /diagnostics: Generate ACF/PACF for parameter tuning.
  • /forecast: Run SARIMAX forecast with auto grid.
  • /download/forecast.csv: Download forecast results


### Frontend
- **React.js** for building the user interface.
- **Material-UI** for responsive and modern design.
- **Redux Toolkit** for state management.
- **Features**:
  - Login and authentication.
  - Data analysis, cleaning, and visualization.
  - Machine learning model selection and prediction.

---

## Screenshots

### Login Screen
![Login Screen](assets/login.JPG)

---

# Model Evaluation Summary


## Conclusion
- **Random Forest** and **Gradient Boosting** show excellent results.
- **Decision Tree** achieves perfect training accuracy but might overfit.
- **Linear Regression** and **XGBoost** underperform and may need feature engineering or model tuning.
- **Logistic Regression**, **Naive Bayes**, and **SVM** provide moderate classification performance.

---


## Technologies Used

### Backend
- **Flask**
- **Flask-CORS**
- **Pandas** for data processing
- **Scikit-Learn** for machine learning
- **XGBoost** for advanced modeling
- **Logistic Regression**
- **Svm**
- **Naive Bayes**

### Frontend
- **React.js**
- **Redux Toolkit**
- **Material-UI**
- **Chart.js** for visualizations
- **JSPDF** for PDF Download

---

## Authors

👤 **Shubham Raj**  
- GitHub: [@ShubhamRaj](https://github.com/shubham14p3)  
- LinkedIn: [Shubham Raj](https://www.linkedin.com/in/shubham14p3/)

---

## Future Upgrades

- Add more advanced machine learning models.
- Enhance data visualization with interactive charts.
- Integrate user-specific data upload and analysis.
- Adding more data for more better accuracy.
- Adding new disease dataset for more all round prediction.

---

## Contributions

Feel free to contribute by creating pull requests or submitting issues. Suggestions for improving data processing methods, adding more visualizations, or optimizing the application are welcome.

---

## Show Your Support

Give a ⭐ if you like this project!

---

## Acknowledgments

- Supported by [IIT Jodhpur](https://www.iitj.ac.in/).

---

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl.svg?style=flat-square
[contributors-url]: https://github.com/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl.svg?style=flat-square
[forks-url]: https://github.com/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl/network/members
[stars-shield]: https://img.shields.io/github/stars/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl.svg?style=flat-square
[stars-url]: https://github.com/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl/stargazers
[issues-shield]: https://img.shields.io/github/issues/shubham14p3/iitj-customer-lifetime-value-prediction-in-e-commerce-dl.svg?style=flat-square
