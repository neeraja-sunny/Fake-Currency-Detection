document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadContent = document.querySelector('.upload-content');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const fileName = document.getElementById('file-name');
    const clearBtn = document.getElementById('clear-btn');
    
    const detectBtn = document.getElementById('detect-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    
    const resultContainer = document.getElementById('result-container');
    const resultCard = document.getElementById('result-card');
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');
    const progressFill = document.getElementById('progress-fill');
    
    const errorMsg = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    let selectedFile = null;

    // Trigger file input dialog when clicking browse button or drop zone
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    // Also allow clicking anywhere in the dropzone if no file is selected
    dropZone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            if (!selectedFile) dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle the selected file
    function handleFile(file) {
        hideError();
        resultContainer.classList.add('hidden'); // Hide previous results
        
        // Validate file type
        if (!file.type.match('image.*')) {
            showError("Please upload an image file (JPG, PNG, WEBP).");
            return;
        }

        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            showError("File size is too large. Maximum 16MB allowed.");
            return;
        }

        selectedFile = file;
        
        // Display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            fileName.textContent = file.name;
            
            // UI transitions
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            detectBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Clear the selected image
    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        
        // UI transitions
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        detectBtn.disabled = true;
        resultContainer.classList.add('hidden');
        hideError();
    });

    // Detect Currency Action
    detectBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Loading state
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        detectBtn.disabled = true;
        resultContainer.classList.add('hidden');
        hideError();

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server processing failed');
            }

            // Display results
            displayResult(data);

        } catch (error) {
            showError(error.message || "Failed to connect to the server.");
        } finally {
            // Restore UI state
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            detectBtn.disabled = false;
        }
    });

    function displayResult(data) {
        // Remove previous status classes
        resultCard.classList.remove('success', 'danger');
        progressFill.style.width = '0%';
        
        // Set new values
        predictionText.textContent = data.result;
        confidenceText.textContent = data.confidence;
        
        // Set appropriate styling
        if (data.is_real) {
            resultCard.classList.add('success');
        } else {
            resultCard.classList.add('danger');
        }
        
        // Show container
        resultContainer.classList.remove('hidden');
        
        // Animate progress bar
        setTimeout(() => {
            progressFill.style.width = data.confidence;
        }, 100);
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorMsg.classList.remove('hidden');
    }

    function hideError() {
        errorMsg.classList.add('hidden');
    }
});
