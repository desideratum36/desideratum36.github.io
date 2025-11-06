# CS 4641: Polymer Property Detection 🧪

![Python](https://img.shields.io/badge/python-3.14.0-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project for **CS 4641** to predict the properties of polymers based on a given dataset. This project explores various regression and classification models to determine material characteristics.

## 🌟 Overview

The goal of this project is to leverage machine learning techniques to predict key properties of polymers. By analyzing a dataset of polymers and their measured attributes, we aim to build a model that can accurately forecast these properties for new, untested materials. This can significantly accelerate material science research and development.

## 📊 Dataset

The dataset for this project is located in `data/TgSS_enriched_cleaned.csv`. It contains various features of polymers, which are used as inputs for our machine learning models. The target variables to be predicted are also included in this dataset.

**Data Source:** [Extra dataset with SMILES,Tg,PID,Polimers Class](https://doi.org/10.34740/kaggle/dsv/12871401), originally taken with permission from PoLyInfo. [\[doi: 10.1109/eidwt.2011.13\]](https://doi.org/10.1109/eidwt.2011.13)

## 🤖 Methodology

This project follows a standard machine learning workflow:

1. [**Data Preprocessing:**](/code/preprocessing.ipynb) Cleaning and preparing the data from `TgSS_enriched_cleaned.csv`. This includes handling missing values, feature scaling, categorical variables, and generating additional features using [RDKit](https://www.rdkit.org/).
2. [**Dimensionality Reduction:**](/code/dimensionality_reduction.ipynb) Removing highly correlated features and then choosing the top 40 features based on principle component analysis (PCA).
3. **Exploratory Data Analysis (EDA):** Analyzing the dataset to understand the relationships between different features and the target properties.
4. **Model Training:** Implementing and training several machine learning models, such as:
   * Linear Regression
   * [Random Forest](/code/random_forest.ipynb)
   * Support Vector Machines (SVM)
   * [Gradient Boosting](/code/gradient_boost.ipynb)
5. **Model Evaluation:** Assessing the performance of the trained models using metrics like Mean Squared Error (MSE), R-squared, and accuracy.
6. **Hyperparameter Tuning:** Optimizing the models to achieve the best possible performance.

The core logic for the machine learning pipeline can be found in `code/`.

## 📂 Project Structure
```
cs4641-polymer-property-detection/
│
├── .gitignore
├── README.md
├── environment.yml
├── favicon.ico
├── index.html
├── midterm.html
├── final.html
│
├── code/
│   ├── dimensionality_reduction.ipynb
│   ├── gradient_boost.ipynb
│   ├── preprocessing.ipynb
│   ├── random_forest.ipynb
│   └── plotting_help.json
│
├── data/
│   ├── TgSS_enriched_cleaned.csv
│   ├── featurized_TgSS.csv
│   └── reduced_TgSS.csv
│
├── fonts/
│   ├── FiraSans-Light.ttf
│   ├── FiraSans-Medium.ttf
│   └── FiraSans-Regular.ttf
│
├── hyperparameters/
│   ├── GradientBoostingMultiOutputClassifier.json
│   ├── GradientBoostingRegressor.json
│   ├── RandomForestClassifier.json
│   ├── RandomForestMultiOutputClassifier.json
│   └── RandomForestRegressor.json
│
├── images/
│   ├── gantt-chart.png
│   └── putImageHere.png
│
└── plots/
    ├── PCA_PC1_vs_PC2.png
    ├── effects_of_pairwise_feature_removal.png
    ├── gradient_boosting_multioutput_classifier.png
    ├── gradient_boosting_regressor.png
    ├── high_correlation_feature_network.html
    ├── polymer_class_breakdown.png
    ├── polymer_class_examples.png
    ├── random_forest_classifier.png
    ├── random_forest_multioutput_classifier.png
    ├── random_forest_regressor.png
    ├── scree_plot.png
    ├── top_40_features_weighted_variance.png
    └── transition_temperature_histogram.png
```

## 📁 Repository Contents

### Root Directory
`/`: Project root containing the main HTML reports, environment file, and README.

- `README.md`: This file — project overview, how to run, and a map of the repository.
- `environment.yml`: Conda environment specification for reproducible installs.
- `index.html`: Project landing page (static HTML presentation).
- `midterm.html`: Midterm presentation/report.
- `final.html`: Final project report / results presentation.
- `favicon.ico`: Site favicon used by the HTML pages.

### Code Directory
`/code/`: Jupyter notebooks and helper artifacts used for preprocessing, analyses, and model training.

- `preprocessing.ipynb`: Notebook that performs data cleaning, featurization checks, and preprocessing steps.
- `dimensionality_reduction.ipynb`: Notebook for PCA, correlation analysis, and feature selection.
- `random_forest.ipynb`: Notebook training and evaluating random forest models.
- `gradient_boost.ipynb`: Notebook training and evaluating gradient boosting models.
- `plotting_help.json`: JSON containing plotting configuration helpers (used by notebooks/plots).

### Data Directory
`/data/`: All primary CSV datasets used by the analysis.

- `TgSS_enriched_cleaned.csv`: Cleaned and enriched dataset used as the canonical source for experiments.
- `featurized_TgSS.csv`: Featurized dataset (raw features generated from SMILES / chemistry inputs).
- `reduced_TgSS.csv`: Dataset reduced by feature selection / PCA (used for downstream model training and plotting).

### Hyperparameters Directory
`/hyperparameters/`: JSON files with hyperparameter configurations (saved best configurations or tuning grids).

- `RandomForestRegressor.json`: RF regressor params.
- `RandomForestClassifier.json`: RF classifier params.
- `RandomForestMultiOutputClassifier.json`: RF multi-output classifier params.
- `GradientBoostingRegressor.json`: GB regressor params.
- `GradientBoostingMultiOutputClassifier.json`: GB multi-output classifier params.

### Fonts Directory
`/fonts/`: Font assets used by the HTML reports and visualizations.

- `FiraSans-Light.ttf`: Fira Sans Light font file.
- `FiraSans-Medium.ttf`: Fira Sans Medium font file.
- `FiraSans-Regular.ttf`: Fira Sans Regular font file.

### Images Directory
`/images/`: Static images for the report and documentation.

- `gantt-chart.png`: Project timeline / Gantt chart image.
- `putImageHere.png`: Placeholder image used in the site template.

### Plots Directory
`/plots/`: Generated plots and interactive plot files from analyses and model evaluation.

- `transition_temperature_histogram.png`: Histogram of transition temperatures (target distribution).
- `polymer_class_breakdown.png`: Plot showing class distribution among polymers.
- `polymer_class_examples.png`: Example polymers per class.
- `high_correlation_feature_network.html`: Interactive network showing highly correlated features.
- `effects_of_pairwise_feature_removal.png`: Plot showing effect of removing feature pairs.
- `scree_plot.png`: PCA scree plot (variance explained by components).
- `PCA_PC1_vs_PC2.png`: PCA scatter plot (PC1 vs PC2).
- `top_40_features_weighted_variance.png`: Ranked top 40 features by weighted variance.
- `random_forest_regressor.png`: RF regressor performance.
- `random_forest_classifier.png`: RF classifier performance.
- `random_forest_multioutput_classifier.png`: RF multi-output classifier performance.
- `gradient_boosting_regressor.png`: GB regressor performance.
- `gradient_boosting_multioutput_classifier.png`: GB multi-output classifier performance.

## 🚀 Getting Started

### Prerequisites

- Python 3.14.0
- Jupyter Notebook
- Required Python packages are listed in `environment.yml` for easier dependency installation.

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/hshah339/cs4641-polymer-property-detection.git
```
2. Navigate to the project directory:
```bash
   cd cs4641-polymer-property-detection
```
3. Install the required packages:
```bash
   conda env create -f environment.yml
   conda activate <environment-name>
```

### Usage

To run the analysis and train the models, open and execute the Jupyter notebooks in the `code/` directory in the following order:

1. `preprocessing.ipynb` - Clean and featurize the data
2. `dimensionality_reduction.ipynb` - Perform PCA and feature selection
3. `random_forest.ipynb` - Train and evaluate Random Forest models
4. `gradient_boost.ipynb` - Train and evaluate Gradient Boosting models

## 📈 Results and Visualization

The final results, including model performance metrics and visualizations, can be found in `final.html`. The `midterm.html` file contains the preliminary findings from the midterm report.

All generated plots from the analysis are stored in the `plots/` directory, and the project timeline is visualized in the Gantt chart located at `images/gantt-chart.png`.

## 🤝 Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to fork the repository and create a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- This project was completed as part of the **CS 4641: Machine Learning** course at Georgia Institute of Technology.
- Special thanks to the instructors and teaching assistants for their guidance and support.
- Dataset originally sourced from PoLyInfo with permission.
