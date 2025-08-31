document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const reviewInput = document.getElementById('review-input');
    const resultContainer = document.getElementById('result-container');
    const predictionText = document.getElementById('prediction-text');
    const confidenceScore = document.getElementById('confidence-score');
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');

    analyzeBtn.addEventListener('click', async () => {
        const reviewText = reviewInput.value;

        // Hide previous results and errors
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');

        if (!reviewText.trim()) {
            showError('Please enter a review to analyze.');
            return;
        }

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    review_text: reviewText
                })
            });

            const result = await response.json();

            if (!response.ok) {
                showError(result.error || 'An unknown error occurred.');
            } else {
                displayResult(result);
            }

        } catch (error) {
            showError('Failed to connect to the server. Please try again.');
        }
    });

    function displayResult(result) {
        predictionText.textContent = result.prediction;
        confidenceScore.textContent = result.confidence;
        
        // Change color based on result
        if (result.prediction.includes('Genuine')) {
            predictionText.style.color = '#27ae60'; // Green
        } else {
            predictionText.style.color = '#e74c3c'; // Red
        }
        
        resultContainer.classList.remove('hidden');
    }

    function showError(message) {
        errorText.textContent = message;
        errorContainer.classList.remove('hidden');
    }
});