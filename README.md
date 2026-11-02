![Email Inbox](email_inbox.jpg)

# Intelligent Email Classifier with Llama

An AI-powered email classification system built with **Python, Llama, llama-cpp-python, and Google Colab**. The project automatically categorizes emails into three business-focused categories:

* **Priority**: Urgent or security-sensitive messages requiring attention
* **Updates**: Notifications, events, and informational messages
* **Promotions**: Marketing messages, offers, and sales content

The classifier runs a **quantized TinyLlama 1.1B model locally through `llama-cpp-python`**, avoiding the need for an external LLM API. 

## Project Overview

Professionals often receive hundreds of emails every day. Manually identifying important messages can be time-consuming and inefficient.

This project uses a lightweight LLM to automatically analyze email content and assign each message to the appropriate category. The workflow combines **data loading, local LLM inference, few-shot prompting, classification, evaluation, and result export** into a simple end-to-end pipeline.

```text
Email CSV
    │
    ▼
Load Dataset
    │
    ▼
TinyLlama 1.1B
    │
    │ Few-shot Prompt
    ▼
Email Classification
    │
    ├── Priority
    ├── Updates
    └── Promotions
    │
    ▼
Compare with Expected Labels
    │
    ▼
Accuracy & Misclassified Emails
    │
    ▼
CSV Results
```

## Dataset

The notebook expects an `email_categories_data.csv` file containing three columns: `email_id`, `email_content`, and `expected_category`. The expected category contains the ground-truth label used for evaluation. 

Example:

| email_id | email_content                      | expected_category |
| -------- | ---------------------------------- | ----------------- |
| 001      | Urgent: Password Reset Required... | Priority          |
| 002      | Special Offer: 50% Off...          | Promotions        |
| 003      | Team meeting has been cancelled... | Updates           |

## Technology Stack

* **Python**
* **Pandas**
* **TinyLlama 1.1B**
* **llama-cpp-python**
* **Hugging Face Hub**
* **GGUF quantized model**
* **Google Colab**

The notebook installs `llama-cpp-python==0.2.82` and `huggingface_hub`, then uses Hugging Face Hub to download the quantized TinyLlama model. 

## How It Works

### 1. Load the Dataset

The user uploads the CSV directly into Google Colab. Pandas then loads the file into a DataFrame for processing. 

### 2. Download the LLM

The project downloads:

```text
TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF
```

using the `hf_hub_download` function. It specifically uses the `Q4_K_M` GGUF quantized model. 

### 3. Few-Shot Classification

The classifier uses examples in the prompt to teach the model the expected classification behavior.

For example:

```text
Urgent: Password Reset Required
→ Priority

Special Offer - 50% Off Everything!
→ Promotions

Canceled Event - Team Meeting
→ Updates
```

The model then receives a new email and is instructed to return the appropriate category. 

### 4. Run Classification

Each email is processed individually through the `classify_message()` function.

The model uses:

```text
temperature = 0
max_tokens = 5
```

This keeps the output short and deterministic because the expected response is only a category label. 

The pipeline processes the **entire dataset**, storing the email ID, original content, expected category, and model prediction. 

### 5. Evaluate the Model

The predicted category is compared against the known `expected_category`.

The notebook calculates:

```text
Accuracy = Correct Predictions / Total Emails
```

It also displays the emails that were incorrectly classified, making it easier to investigate model failures. 

### 6. Export Results

The final predictions can be exported to:

```text
email_classification_results.csv
```

This provides a simple way to use the classification output outside the notebook. 

## Key Features

* **Local LLM inference** without requiring an external LLM API
* **Few-shot prompting** for classification guidance
* **Three-class email categorization**
* **Full dataset processing**
* **Automated accuracy evaluation**
* **Misclassification analysis**
* **CSV result export**
* **Google Colab compatible**
* **Quantized Llama model** for lightweight inference

## Project Structure

```text
.
├── email_classifier_colab.ipynb
├── email_categories_data.csv
└── email_classification_results.csv
```

`email_categories_data.csv` is the input dataset, while `email_classification_results.csv` is generated after classification. The notebook itself contains the complete processing workflow. 

## Running the Project

### 1. Open in Google Colab

Upload/open:

```text
email_classifier_colab.ipynb
```

### 2. Install Dependencies

Run the dependency installation cell.

### 3. Upload the Dataset

Upload:

```text
email_categories_data.csv
```

### 4. Run All Cells

The notebook will:

1. Load the dataset
2. Download TinyLlama
3. Initialize the LLM
4. Build the few-shot prompt
5. Classify every email
6. Calculate accuracy
7. Display misclassified emails
8. Export the results

## Output

The final results contain:

```text
email_id
email_content
expected_category
model_output
correct
```

The `correct` field indicates whether the model prediction matches the expected category. 

## Why This Project Matters

This project demonstrates how a **small, locally deployed LLM can be integrated into a practical data-processing workflow** rather than relying solely on large cloud-based models.

From a Data Engineering perspective, it combines **data ingestion, transformation, automated inference, validation, evaluation, and output generation** into one reproducible pipeline.

It also demonstrates a useful pattern for production systems: **LLM inference + structured output + automated evaluation**, which can be extended to more advanced email routing, document classification, customer support triage, fraud detection, and other classification workloads.

## Future Improvements

Potential next steps include:

* Add more email categories
* Use a larger or more capable instruction-tuned model
* Add precision, recall, and F1-score
* Generate a confusion matrix
* Batch LLM inference for better performance
* Add structured JSON output validation
* Add confidence scoring
* Build a FastAPI inference service
* Containerize the application with Docker
* Add automated tests and CI/CD
* Connect the classifier to Gmail or another email provider
* Add human-in-the-loop review for uncertain classifications

## Important Note

The notebook uses **TinyLlama 1.1B**, so the project is best viewed as a lightweight demonstration of LLM-powered classification rather than a production-ready enterprise email filtering system. The attached notebook does not provide latency, throughput, model-comparison, or production deployment benchmarks, so those should not be claimed without additional testing.
