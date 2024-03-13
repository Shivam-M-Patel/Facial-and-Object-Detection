function updateFileName(input) {
  const selectedFileElement = document.getElementById('selected-file');

  if (input.files.length > 0) {
    const fileName = input.files[0].name;
    selectedFileElement.textContent = 'Selected File: ' + fileName;
  } else {
    selectedFileElement.textContent = 'No file selected';
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const loadingSpinner = document.getElementById("ring");
  const loadingContainer = document.getElementById("loading-container");

  form.addEventListener("submit", function (event) {
    // Show the loading spinner when the form is submitted
    loadingSpinner.style.display = "block";
    loadingContainer.style.display = "block";
  });
});