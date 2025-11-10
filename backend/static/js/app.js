// Global variables
let downloadPath = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // File input handlers
    const dataFileInput = document.getElementById('data-file');
    const modelFileInput = document.getElementById('model-file');
    
    dataFileInput.addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name || 'Choose CSV file...';
        document.getElementById('file-name').textContent = fileName;
        updatePredictButton();
    });
    
    modelFileInput.addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name || 'Choose model file (.pth)...';
        document.getElementById('model-file-name').textContent = fileName;
    });
    
    // Check if model is loaded
    checkModelStatus();
});

// Check model status
function checkModelStatus() {
    fetch('/model_info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateModelStatus(true);
                displayModelInfo(data.info);
            } else {
                updateModelStatus(false);
            }
        })
        .catch(() => {
            updateModelStatus(false);
        });
}

// Update model status UI
function updateModelStatus(loaded) {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const predictBtn = document.getElementById('predict-btn');
    
    if (loaded) {
        statusDot.classList.add('loaded');
        statusText.textContent = 'Model loaded';
        predictBtn.disabled = false;
    } else {
        statusDot.classList.remove('loaded');
        statusText.textContent = 'Model not loaded';
        predictBtn.disabled = true;
    }
}

// Display model information
function displayModelInfo(info) {
    const modelInfoDiv = document.getElementById('model-info');
    const modelDetailsDiv = document.getElementById('model-details');
    
    modelInfoDiv.style.display = 'block';
    
    const grid = document.createElement('div');
    grid.className = 'model-info-grid';
    
    for (const [key, value] of Object.entries(info)) {
        if (key === 'model_loaded') continue;
        
        const item = document.createElement('div');
        item.className = 'model-info-item';
        
        const label = document.createElement('div');
        label.className = 'model-info-label';
        label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        const val = document.createElement('div');
        val.className = 'model-info-value';
        val.textContent = value;
        
        item.appendChild(label);
        item.appendChild(val);
        grid.appendChild(item);
    }
    
    modelDetailsDiv.innerHTML = '';
    modelDetailsDiv.appendChild(grid);
}

// Load model modal
function loadModelModal() {
    document.getElementById('model-modal').style.display = 'block';
}

function closeModelModal() {
    document.getElementById('model-modal').style.display = 'none';
}

// Load model form submission
document.getElementById('load-model-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const modelFile = document.getElementById('model-file').files[0];
    
    if (modelFile) {
        formData.append('model_file', modelFile);
    }
    
    showLoading(true);
    
    fetch('/load_model', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        closeModelModal();
        
        if (data.success) {
            showToast('Model loaded successfully!', 'success');
            checkModelStatus();
        } else {
            showToast(data.message || 'Error loading model', 'error');
        }
    })
    .catch(error => {
        showLoading(false);
        showToast('Error: ' + error.message, 'error');
    });
})

// Predict form submission
document.getElementById('predict-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const dataFile = document.getElementById('data-file').files[0];
    
    if (!dataFile) {
        showToast('Please select a CSV file', 'error');
        return;
    }
    
    formData.append('data_file', dataFile);
    
    showLoading(true);
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = true;
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        predictBtn.disabled = false;
        
        if (data.success) {
            displayResults(data);
            showToast('Predictions completed successfully!', 'success');
        } else {
            showToast(data.message || 'Error making predictions', 'error');
        }
    })
    .catch(error => {
        showLoading(false);
        predictBtn.disabled = false;
        showToast('Error: ' + error.message, 'error');
    });
});

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Display summary statistics
    displaySummaryStats(data.summary);
    
    // Display metrics if available
    if (data.metrics) {
        displayMetrics(data.metrics);
    }
    
    // Display plot
    if (data.plot) {
        const plotContainer = document.getElementById('plot-image');
        plotContainer.innerHTML = `<img src="data:image/png;base64,${data.plot}" alt="Prediction Plot">`;
    }
    
    // Display predictions table
    displayPredictionsTable(data.predictions, data.total_rows);
    
    // Save download path
    downloadPath = data.download_path;
}

// Display summary statistics
function displaySummaryStats(summary) {
    const statsGrid = document.getElementById('summary-stats');
    statsGrid.innerHTML = '';
    
    const stats = [
        { label: 'Total Customers', value: summary.total_customers.toLocaleString() },
        { label: 'Mean Predicted CLV', value: 'R$ ' + summary.mean_predicted_clv.toLocaleString('pt-BR', {minimumFractionDigits: 2}) },
        { label: 'Median Predicted CLV', value: 'R$ ' + summary.median_predicted_clv.toLocaleString('pt-BR', {minimumFractionDigits: 2}) },
        { label: 'Min Predicted CLV', value: 'R$ ' + summary.min_predicted_clv.toLocaleString('pt-BR', {minimumFractionDigits: 2}) },
        { label: 'Max Predicted CLV', value: 'R$ ' + summary.max_predicted_clv.toLocaleString('pt-BR', {minimumFractionDigits: 2}) },
        { label: 'Std Dev', value: 'R$ ' + summary.std_predicted_clv.toLocaleString('pt-BR', {minimumFractionDigits: 2}) }
    ];
    
    stats.forEach(stat => {
        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <h4>${stat.label}</h4>
            <div class="stat-value">${stat.value}</div>
        `;
        statsGrid.appendChild(card);
    });
}

// Display metrics
function displayMetrics(metrics) {
    const metricsContainer = document.getElementById('metrics-container');
    const metricsGrid = document.getElementById('metrics-grid');
    
    metricsContainer.style.display = 'block';
    metricsGrid.innerHTML = '';
    
    const metricLabels = {
        'MSE': 'Mean Squared Error',
        'RMSE': 'Root Mean Squared Error',
        'MAE': 'Mean Absolute Error',
        'R2': 'R² Score',
        'MAPE': 'Mean Absolute Percentage Error (%)'
    };
    
    for (const [key, value] of Object.entries(metrics)) {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-label">${metricLabels[key] || key}</div>
            <div class="metric-value">${value}</div>
        `;
        metricsGrid.appendChild(card);
    }
}

// Display predictions table
function displayPredictionsTable(predictions, totalRows) {
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    const tableInfo = document.getElementById('table-info');
    
    if (predictions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="100%">No predictions available</td></tr>';
        return;
    }
    
    // Get column names
    const columns = Object.keys(predictions[0]);
    
    // Create header
    tableHead.innerHTML = '';
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);
    
    // Create body
    tableBody.innerHTML = '';
    predictions.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            let value = row[col];
            
            // Format numeric values
            if (typeof value === 'number') {
                if (col.includes('clv') || col.includes('revenue') || col.includes('value')) {
                    value = 'R$ ' + value.toLocaleString('pt-BR', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                } else {
                    value = value.toLocaleString('pt-BR', {maximumFractionDigits: 2});
                }
            }
            
            td.textContent = value;
            tr.appendChild(td);
        });
        tableBody.appendChild(tr);
    });
    
    // Update table info
    tableInfo.textContent = `Showing ${predictions.length} of ${totalRows.toLocaleString()} predictions`;
}

// Download results
function downloadResults() {
    if (!downloadPath) {
        showToast('No results to download', 'error');
        return;
    }
    
    // Extract filename (handle both full paths and just filenames)
    const filename = downloadPath.includes('/') ? downloadPath.split('/').pop() : downloadPath;
    window.location.href = `/download/${filename}`;
}

// Update predict button state
function updatePredictButton() {
    const dataFile = document.getElementById('data-file').files[0];
    const predictBtn = document.getElementById('predict-btn');
    const statusText = document.getElementById('status-text').textContent;
    
    predictBtn.disabled = !dataFile || statusText !== 'Model loaded';
}

// Show/hide loading overlay
function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('model-modal');
    if (event.target === modal) {
        closeModelModal();
    }
}

