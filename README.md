# Data Science Examples Project

This repository provides a collection of data science examples covering classical regression and classification models. It is organized for clarity and reusability, with Jupyter/Databricks notebooks for hands-on experimentation and a `src/` folder for reusable code components, including a general-purpose model pipeline.

## Features

- **End-to-End Data Science Workflows:**
  - Example notebooks for regression and classification tasks.
  - Step-by-step demonstrations from data loading to evaluation.

- **Reusable Code in `src/`:**
  - Utility functions, data preprocessing, and model pipelines (see `src/model_pipeline.py`).
  - Designed for import and reuse across multiple notebooks.

- **Project Structure for Collaboration:**
  - Clear separation of examples/<problem>/notebooks, source code, tests, and configuration.
  - Ready for extension with new models or datasets.

## Folder Structure

```
datascience_examples/
│
├── src/                        # Reusable Python modules (pipelines, preprocessing, etc.)
│   └── model_pipeline.py       # General-purpose ML pipeline utilities
├── examples/                   # Example problems and their notebooks
│   ├── regression/             # Regression problem notebooks and assets
│   │   └── regression_example.ipynb
│   ├── classification/         # Classification problem notebooks and assets
│   │   └── classification_example.ipynb
│   └── ...                     # Additional problem types or topics
├── test/                       # Unit and integration tests for src/
│   └── ...
├── workflows/                  # (Optional) Workflow/job definitions (e.g., for Databricks)
│   └── ...
├── pyproject.toml              # Project dependencies and configuration
├── uv.lock                     # Locked dependency versions
├── .github/workflows/          # CI/CD workflows (e.g., linting, testing)
├── .pre-commit-config.yaml     # Pre-commit hooks for code quality
└── README.md                   # Project overview and instructions
```

## Example Notebooks

- **Regression Example:** Linear regression, evaluation metrics, and visualization.
- **Classification Example:** Logistic regression, decision trees, accuracy, and confusion matrix.
- Each notebook is self-contained and demonstrates best practices for data science workflows.

## Getting Started

1. **Clone the Repository:**
   ```sh
   git clone <your-repo-url>
   cd datascience_examples
   ```

2. **Install Dependencies:**
   ```sh
   uv sync --project pyproject.toml
   ```

3. **Run Notebooks:**
   - Open notebooks in VS Code or Databricks.
   - Experiment with different models and datasets.

4. **Reuse Code:**
   - Import functions and pipelines from `src/model_pipeline.py` in your own notebooks or scripts.

5. **Testing and Code Quality:**
   - Run tests in `test/` and use pre-commit hooks for linting and formatting.

## Notes

- The repository is designed for flexibility and easy extension.
- Add new notebooks or modules to cover additional data science topics.
- Use the provided folder structure to keep code organized and maintainable.

---
