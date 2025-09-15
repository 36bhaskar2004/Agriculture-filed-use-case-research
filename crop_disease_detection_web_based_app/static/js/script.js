document.addEventListener('DOMContentLoaded', function() {

    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsSection = document.getElementById('resultsSection');
    
  
    const diseaseModal = document.getElementById('diseaseModal');
    const modalContent = document.getElementById('modalContent');
    const closeModal = document.querySelector('.close-modal');
    
    
    const infoButtons = document.querySelectorAll('.info-btn');
    
   
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
            }
            reader.readAsDataURL(this.files[0]);
        } else {
            fileName.textContent = 'No file selected';
            previewContainer.style.display = 'none';
        }
    });
    
   
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files[0]) {
            alert('Please select an image file first');
            return;
        }
        
       
        loadingIndicator.style.display = 'flex';
        resultsContainer.style.display = 'none';
        resultsSection.style.display = 'block';
        
       
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('location', document.getElementById('location').value);
    
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            resultsContainer.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Error Processing Request</h3>
                    <p>${error.message}</p>
                </div>
            `;
            resultsContainer.style.display = 'block';
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';
            
           
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        });
    });
    
 
    function displayResults(data) {
        const prediction = data.prediction;
        const diseaseInfo = data.disease_info;
        const weather = data.weather;
        
        let weatherHTML = '';
        if (weather) {
            weatherHTML = `
                <div class="weather-card">
                    <div class="weather-header">
                        <i class="fas fa-cloud-sun"></i>
                        <h3>Current Weather in ${weather.location}</h3>
                    </div>
                    <div class="weather-info">
                        <div class="weather-item">
                            <div class="label">Temperature</div>
                            <div class="value">${weather.temperature}°C</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Humidity</div>
                            <div class="value">${weather.humidity}%</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Conditions</div>
                            <div class="value">${weather.conditions}</div>
                        </div>
                        <div class="weather-item">
                            <div class="label">Wind Speed</div>
                            <div class="value">${weather.wind_speed} m/s</div>
                        </div>
                    </div>
                    ${diseaseInfo.weather_recommendations ? `
                    <div class="weather-recommendations">
                        <h4><i class="fas fa-umbrella"></i> Weather-based Recommendations</h4>
                        <ul>
                            ${diseaseInfo.weather_recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>
            `;
        }
        
        let otherPredictionsHTML = '';
        if (prediction.top_predictions.length > 1) {
            otherPredictionsHTML = `
                <div class="detail-card">
                    <h4><i class="fas fa-list-ol"></i> Other Possible Diagnoses</h4>
                    <ul>
                        ${prediction.top_predictions.slice(1).map(pred => 
                            `<li>${pred[0].replace(/_/g, ' ')} (${(pred[1] * 100).toFixed(1)}% confidence)</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        }
        
        resultsContainer.innerHTML = `
            <div class="prediction-result">
                <div class="result-header">
                    <div class="result-title">
                        <i class="fas fa-diagnoses"></i>
                        Diagnosis Results
                    </div>
                    <div class="confidence">${(prediction.confidence * 100).toFixed(1)}% Confidence</div>
                </div>
                
                <div class="result-content">
                    <div class="result-image">
                        <img src="${prediction.image_path}" alt="Uploaded plant image">
                    </div>
                    
                    <div class="result-details">
                        <h3>${prediction.predicted_class.replace(/_/g, ' ')}</h3>
                        
                        <div class="detail-card">
                            <h4><i class="fas fa-info-circle"></i> Description</h4>
                            <p>${diseaseInfo.description}</p>
                            <p><strong>Severity:</strong> ${diseaseInfo.severity}</p>
                        </div>
                        
                        <div class="detail-card">
                            <h4><i class="fas fa-clipboard-check"></i> Recommended Actions</h4>
                            <ul>
                                ${diseaseInfo.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                            </ul>
                        </div>
                        
                        ${otherPredictionsHTML}
                        
                        <div class="detail-card">
                            <h4><i class="fas fa-spa"></i> Fertilizer Recommendations</h4>
                            <ul>
                                ${diseaseInfo.fertilizers.map(fert => `<li>${fert}</li>`).join('')}
                            </ul>
                        </div>
                        
                        ${diseaseInfo.organic_control ? `
                        <div class="detail-card">
                            <h4><i class="fas fa-leaf"></i> Organic Control</h4>
                            <p>${diseaseInfo.organic_control}</p>
                        </div>
                        ` : ''}
                    </div>
                </div>
                
                ${weatherHTML}
            </div>
        `;
        
        resultsContainer.style.display = 'block';
    }
    
   
    infoButtons.forEach(button => {
        button.addEventListener('click', function() {
            const disease = this.getAttribute('data-disease');
            fetchDiseaseInfo(disease);
        });
    });
    
    function fetchDiseaseInfo(disease) {
        
        const diseaseInfo = {
            "description": "Detailed information about " + disease.replace(/_/g, ' '),
            "recommendations": [
                "Recommendation 1 for " + disease,
                "Recommendation 2 for " + disease,
                "Recommendation 3 for " + disease
            ],
            "severity": "Moderate",
            "fertilizers": [
                "Fertilizer option 1",
                "Fertilizer option 2"
            ],
            "prevention": [
                "Prevention tip 1",
                "Prevention tip 2"
            ]
        };
        
        displayDiseaseModal(disease, diseaseInfo);
    }
    
    function displayDiseaseModal(disease, info) {
        modalContent.innerHTML = `
            <h2>${disease.replace(/_/g, ' ')}</h2>
            
            <div class="modal-section">
                <h3><i class="fas fa-info-circle"></i> Description</h3>
                <p>${info.description}</p>
                <p><strong>Severity:</strong> ${info.severity}</p>
            </div>
            
            <div class="modal-section">
                <h3><i class="fas fa-clipboard-list"></i> Symptoms</h3>
                <p>Common symptoms of this disease include...</p>
                <ul>
                    <li>Symptom 1</li>
                    <li>Symptom 2</li>
                    <li>Symptom 3</li>
                </ul>
            </div>
            
            <div class="modal-section">
                <h3><i class="fas fa-clipboard-check"></i> Treatment Recommendations</h3>
                <ul>
                    ${info.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
            
            <div class="modal-section">
                <h3><i class="fas fa-spa"></i> Fertilizer Recommendations</h3>
                <ul>
                    ${info.fertilizers.map(fert => `<li>${fert}</li>`).join('')}
                </ul>
            </div>
            
            <div class="modal-section">
                <h3><i class="fas fa-shield-alt"></i> Prevention</h3>
                <ul>
                    ${info.prevention.map(prev => `<li>${prev}</li>`).join('')}
                </ul>
            </div>
        `;
        
        diseaseModal.style.display = 'block';
    }
    
   
    closeModal.addEventListener('click', function() {
        diseaseModal.style.display = 'none';
    });
    
    
    window.addEventListener('click', function(event) {
        if (event.target === diseaseModal) {
            diseaseModal.style.display = 'none';
        }
    });
});

try {
    const response = await fetch('/predict_video', {
        method: 'POST',
        body: formData
    });
    
    //new
    const text = await response.text(); // get raw response
    let data;
    try {
        data = JSON.parse(text); // parse only if JSON
    } catch {
        throw new Error("Server returned non-JSON response: " + text);
    }

    if (data.error) throw new Error(data.error);

    document.getElementById('videoPrediction').innerHTML = `
        <div class="alert alert-success">
            Final Predicted Disease: <strong>${data.final_prediction}</strong>
        </div>
    `;
    document.getElementById('videoResultsContainer').style.display = 'block';
} catch (error) {
    alert('Video prediction error: ' + error.message);
}
