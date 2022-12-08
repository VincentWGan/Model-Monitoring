# A Simple Live Model Monitoring Dashboard
A Project by Penn State UP DS440 Fall 2022 Section 002 Group 6 

## Introduction
Model monitoring is the practice of measuring model performance. As a field, model monitoring
is broadly defined. A model monitoring system can monitor features including
but not limited to predictive correctness, hardware usage, the integrity of input data,
and data drifts. Monitoring these features allows model users to spot model malfunctions
as early as possible and prevent loss.

The team crafted a model monitoring application that can provide users with real-time
updates on their model performance and possible drifts in data distribution and concept. The
users will only need to provide the data source and trainedmodel, as well as some customization
settings on the user interface. The data source can either be static data or live data from
an URL, data streaming API, or a local file. Currently, supported model types include binary
classification, regression, and time series for tabular data, and binary classification for
images.

## Installation
The project is built on Python version 3.9.13. To install, first clone the repo and install the Python dependencies by:

```bash
pip install -r requirements.txt
```

Some computer vision packages are also required for the image classification demo. They are specified in "packages.txt". 

Finally, to run, set your current directory to "Model-Monitoring/" in the command line, then:

```bash
streamlit run model_monitoring_app.py
```

