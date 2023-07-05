---
editor_options: 
  markdown: 
    wrap: 72
---

# Analyse air quality data

Project **Analyse air quality data** based on air measurements in New
Zealand

## Project Description

This project focuses on analyzing air quality data obtained from
measurement stations across New Zealand, specifically from the [Ministry
for the Environment (MfE)](https://data.mfe.govt.nz/). The dataset
encompasses various pollutants, including PM2.5, PM10, and others.

The project begins with a comprehensive Exploratory Data Analysis (EDA)
to gain a deeper understanding of the data and its characteristics. This
EDA phase enables insights into trends, patterns, and potential
correlations within the air quality dataset.

To handle missing values within the dataset, advanced techniques
utilizing neural networks are employed. Specifically, the project
utilizes the Keras/TensorFlow framework to implement neural networks
that effectively interpolate missing values. This approach enables the
generation of more complete and reliable air quality measurements.

Furthermore, the project incorporates spatial interpolation techniques
to enhance the dataset's spatial coverage. This spatial interpolation
process allows for the estimation of air quality values at locations
without specific measurements, thus providing a more comprehensive
perspective on the overall air quality landscape in New Zealand.

As a final step, the project presents an interactive map that visualizes
the spatially interpolated air quality data. This map serves as a
user-friendly tool for exploring and understanding air quality patterns
across different regions of New Zealand.

Overall, this project combines thorough exploratory analysis, advanced
neural network interpolation methods, spatial interpolation techniques,
and interactive visualization to provide valuable insights into the air
quality landscape of New Zealand.

The workflow consists of three steps (each represented in one Jupyter
Notebook):

-   Exploratory data analysis (EDA)
-   Interpolation of missing data with neural networks
-   Spatial interpolation and mapping of data

## Files and data description

Overview of the most important files and data present in the root
directory:

-   Folders:
    -   `data`: Main folder.
        -   `zip`: Contains the input data in `dbf` format, downloaded
            from the
            [MfE](https://data.mfe.govt.nz/tables/category/environmental-reporting/air/?q=concentrations&updated_at.after=2021-02-05T02%3A21%3A54.805Z)
    -   `img`: Contains figures that show the average concentration of
        pollutants over time.
    -   `results`: Contains results such as interpolated data sets.
-   Files:
    -   `datasheet.md` : Description of the input data from MfE.
    -   `01_Air_Quality_EDA.ipynb`: Main workflow for the exploratory
        data analysis.
    -   `02_Air_Quality_Interpolation.ipynb`: Main workflow for
        interpolating data gaps, based on neural networks.
    -   `utils.py`: A library of functions used in the Jupyter
        Notebooks.


## Running Files


## **Run unit tests** 

Not implemented.

## Dependencies

-   Python 3.7
-   Pandas 1.2.4
-   Scikit-learn 0.24.2
-   Matplotlib 3.4.2
-   Seaborn 0.11.1
