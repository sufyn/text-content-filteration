
# Text Analysis App

Text Analysis App is a versatile application built with [Streamlit](https://streamlit.io/) for real-time toxicity prediction and category suggestion based on user input text. This app leverages machine learning models to classify abusive content and suggest relevant categories for input keywords.

## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Demo
![Text Analysis App Demo](demo.gif)

## Features
- **Toxicity Prediction**: Predicts the abusiveness of input text and provides a probability score.
- **Category Suggestion**: Suggests categories based on keywords using cosine similarity on a set of predefined keywords for each category.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/text-analysis-app.git
    cd text-analysis-app
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Model and Vectorizer**: Ensure `logistic_regression_model.pkl` and `vectorizer.pkl` are in the project directory.

## Usage

To run the application, use the following command:

```bash
streamlit run app.py
```

After running this command, a local URL (e.g., `http://localhost:8501`) will be displayed. Open it in your web browser to start using the Text Analysis App.

## App Structure
### Toxicity Prediction
1. Navigate to **Toxicity Prediction** from the sidebar.
2. Enter the text to analyze for abusiveness.
3. Click **Predict Toxicity** to get a prediction score indicating the likelihood of the text being abusive.

### Category Suggestion
1. Select **Category Suggestion** from the sidebar.
2. Input a keyword to receive suggestions based on cosine similarity with predefined categories.
3. Click **Get Suggestions** to see the top 3 relevant categories.

## Dependencies

- **Streamlit** - for building the interactive web app.
- **scikit-learn** - for the machine learning models.
- **pandas** - for handling data operations.
- **joblib** - for loading pre-trained models.
- **re** - for text preprocessing.
- **TfidfVectorizer** - for text vectorization.

Install all required libraries:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Enjoy exploring text analysis with **Text Analysis App**! Contributions, issues, and feature suggestions are welcome.
