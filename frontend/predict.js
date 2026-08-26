const form = document.getElementById('prediction_form');
const buggy_input = document.getElementById('buggy_code');
const fixed_input = document.getElementById('fixed_code');
const message = document.getElementById('status_message');
const button = document.getElementById('predict_button');
const v1_result = document.getElementById('v1_prediction');
const v2_result = document.getElementById('v2_prediction');
const v3_result = document.getElementById('v3_prediction');
const generated_diff = document.getElementById('generated_diff');

buggy_input.addEventListener("input", handleInputChange);
fixed_input.addEventListener("input", handleInputChange);

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearResults();

    try {
        const buggy_val = buggy_input.value;
        const fixed_val = fixed_input.value;
        message.textContent ='Status: Generating predictions...';
        button.disabled = true;
        const requestBody = {
            buggy_code: buggy_val,
            fixed_code: fixed_val
        }

        const jsonBody = JSON.stringify(requestBody)
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'},
            body: jsonBody
        });

        const responseText = await response.text();

        let data;

        try {
            data = JSON.parse(responseText);
        } catch {
            throw new Error(
                "The server could not process this prediction. Please try again."
            );
        }
        if (!response.ok) {
            throw new Error(data.detail || 'Prediction failed.');
        }

        v1_result.textContent = data.all_predictions.v1_prediction.replaceAll("_", " ");
        v2_result.textContent = data.all_predictions.v2_prediction.replaceAll("_", " ");
        v3_result.textContent = data.all_predictions.v3_prediction.replaceAll("_", " ");
        displayDiff(data.diff);
        message.textContent = 'Status: ' + 'Prediction generated.';
    } catch (error) {
        message.textContent = 'Status: ' + error.message;
    } finally {
        button.disabled = false;
    }
})

function displayDiff(diff) {
    generated_diff.replaceChildren();

    const lines = diff.split("\n");

    for (const line of lines) {
        const diffLine = document.createElement("span");
        diffLine.classList.add("diff-line");

        if (line.startsWith("+")) {
            diffLine.classList.add("diff-added");
        } else if (line.startsWith("-")) {
            diffLine.classList.add("diff-removed");
        } else {
            diffLine.classList.add("diff-context");
        }

        diffLine.textContent = line || " ";
        generated_diff.append(diffLine, document.createTextNode("\n"));
    }
}

function clearResults() {
    generated_diff.textContent = "No diff generated.";
    v1_result.textContent = "No prediction yet.";
    v2_result.textContent = "No prediction yet.";
    v3_result.textContent = "No prediction yet.";
}

function handleInputChange() {
    clearResults();
    message.textContent = "Status: Ready for a prediction.";
}