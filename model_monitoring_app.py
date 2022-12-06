import time  # to simulate a real time data, time loop
import joblib
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # data web app development
from Multivar_TS_demo import multiTSDataStream as multits
from Regression_demo import regressionDataStream as reg
from Classification_demo import classificationDataStream as cla
from ImageClassification_demo import imageDataStream as im
from sklearn.metrics import mean_absolute_percentage_error as mape
from copy import deepcopy
from math import ceil
from sklearn.metrics import accuracy_score as acu
from scipy.stats import ks_2samp


st.set_page_config(
    page_title="Real-Time Model Monitoring Dashboard",
    page_icon="✅",
    layout="wide",
)

supported_models = ['Classification', 'Regression', 'Time Series Forecasting', 'Image Classification']
with st.sidebar:
    st.title("Model Monitoring Dashboard")
    model_filter = st.selectbox("Select the model type", supported_models)
    if model_filter == 'Classification':
        df_path = st.text_input(
            'Relative path of data source', 'Classification_demo/df_test.csv')
        model_path = st.text_input(
            'Relative path of model', 'Classification_demo/Classification.sav')
        acu_alert_thres = st.number_input(
            "Enter the accuracy threshold below which alert will be triggered", min_value=0, max_value=1)

    if model_filter == 'Regression':
        df_path = st.text_input(
            'Relative path of data source', 'Regression_demo/df_test.csv')
        model_path = st.text_input(
            'Relative path of model', 'Regression_demo/regression.sav')
        mape_alert_thres = st.number_input(
            "Enter the MAPE threshold above which alert will be triggered", min_value=0, max_value=1)

    if model_filter == 'Time Series Forecasting':
        df_path = st.text_input(
            'Relative path of data source', 'Multivar_TS_demo/multi_test_df.csv')
        model_path = st.text_input(
            'Relative path of model', 'Multivar_TS_demo/multi_ts_forcast.sav')
        mape_alert_thres = st.number_input(
            "Enter the MAPE threshold above which alert will be triggered", min_value=0, max_value=1)

    if model_filter == 'Image Classification':
        df_path = st.text_input(
            'Relative path of image directory', 'ImageClassification_demo/test_copy')

    if (df_path != '') and (model_filter != 'Image Classification'):
        df_og = pd.read_csv(df_path).iloc[:, 1:]
        drift_detect_features = st.multiselect(
            'Enter a list of column names for drift detection', df_og.columns, list(df_og.columns))
        target_feature = st.selectbox(
            "Select the target feature", df_og.columns)

    batch_size = st.number_input(
        "Enter the batch size of data to evaluate at each iteration", min_value=2)

    start_button = st.button('Start')

# creating a single-element container
placeholder = st.empty()

# near real-time / live feed simulation
if start_button:
    if model_filter in ['Classification', 'Regression', 'Time Series Forecasting']:
        df_test = deepcopy(df_og)
        model = joblib.load(model_path)
        # create a dataframe for all tested data
        df_all = pd.DataFrame(columns=df_test.columns)
        # create a dataframe for all y_pred and y_test
        df_pred_test = pd.DataFrame(columns=['y_pred', 'y_test'])
        len_of_batches = 0

    if model_filter == 'Classification':
        # create a dataframe for all accuracy scores
        df_acu = pd.DataFrame(columns=['accuracy'])
        # create a dataframe for all ks test p values
        df_ks_test = pd.DataFrame(columns=drift_detect_features)
        acu_alert = ''
        ks_alert = {f: '' for f in drift_detect_features}
        num_dist_drifts = 0
        while True:
            df_all = pd.concat([df_all, df_test]).reset_index().iloc[:, 1:]
            y_test = df_test[target_feature]
            x_test = df_test.drop(columns=[target_feature])
            y_pred = model.predict(x_test)
            df_pred_test_batch = pd.DataFrame(
                {'y_pred': y_pred, 'y_test': y_test})
            df_pred_test = pd.concat(
                [df_pred_test, df_pred_test_batch]).reset_index().iloc[:, 1:]
            batch_size = min(batch_size, len(df_test))
            num_of_batch = ceil(len(df_test) / batch_size)
            acu_scores = np.array([])
            pred_test_batches = np.array_split(
                df_pred_test_batch, num_of_batch)
            df_test_batches = np.array_split(df_test, num_of_batch)

            # do some calculations on each batch of data
            for i in range(num_of_batch):
                # calculate accuracy score
                score = acu(
                    pred_test_batches[i]['y_test'].values, pred_test_batches[i]['y_pred'].values)
                if score < acu_alert_thres:
                    acu_alert = f'Batch {len_of_batches+i} scored below threshold.\n' + acu_alert
                acu_scores = np.append(acu_scores, score)

                # perform ks test on each column with respect to that column in df_og
                column_scores = {}
                for column in df_ks_test.columns:
                    p_val = ks_2samp(
                        df_og[column], df_test_batches[i][column]).pvalue
                    if p_val < 0.05:
                        alert = f'distribution drift detected at batch {len_of_batches+i}.\n'
                        ks_alert[column] = alert + ks_alert[column]
                        num_dist_drifts += 1
                    column_scores[column] = [p_val]

                df_column_scores = pd.DataFrame(column_scores)
                df_ks_test = pd.concat(
                    [df_ks_test, df_column_scores]).reset_index().iloc[:, 1:]

            df_acu_batch = pd.DataFrame({'accuracy': acu_scores})
            df_acu = pd.concat([df_acu, df_acu_batch]
                               ).reset_index().iloc[:, 1:]
            df_acu['batch_number'] = list(df_acu.index)
            acu_mean = df_acu['accuracy'].mean()
            len_of_batches = len(df_acu)

            with placeholder.container():
                kpi1, kpi2 = st.columns(2)
                kpi1.metric('avg accuracy', acu_mean)
                kpi2.metric('num of distribution drift', num_dist_drifts)

                plot2 = st.expander("Accuracy", expanded=True)
                with plot2:
                    acu_plot, acu_log = st.tabs(['Plot', 'Log'])

                    with acu_plot:
                        st.line_chart(data=df_acu, x='batch_number', y='accuracy')

                    with acu_log:
                        st.text(acu_alert)

                plot3 = st.expander('KS test for distribution drift detection')
                with plot3:
                    column_tabs = st.tabs(drift_detect_features)
                    for i in range(len(drift_detect_features)):
                        column_tabs[i].text(ks_alert[drift_detect_features[i]])

            df_test = cla.classificationDataStream(len(df_test))
            time.sleep(2)

    if model_filter == 'Regression':
        # create a dataframe for all mape scores
        df_mape = pd.DataFrame(columns=['mape'])
        # create a dataframe for all ks test p values
        df_ks_test = pd.DataFrame(columns=drift_detect_features)
        mape_alert = ''
        ks_alert = {f: '' for f in drift_detect_features}
        num_dist_drifts = 0
        while True:
            df_all = pd.concat([df_all, df_test]).reset_index().iloc[:, 1:]
            y_test = df_test[target_feature]
            x_test = df_test.drop(columns=[target_feature])
            y_pred = model.predict(x_test)
            df_pred_test_batch = pd.DataFrame(
                {'y_pred': y_pred, 'y_test': y_test})
            df_pred_test = pd.concat(
                [df_pred_test, df_pred_test_batch]).reset_index().iloc[:, 1:]
            batch_size = min(batch_size, len(df_test))
            num_of_batch = ceil(len(df_test) / batch_size)
            mape_scores = np.array([])
            pred_test_batches = np.array_split(
                df_pred_test_batch, num_of_batch)
            df_test_batches = np.array_split(df_test, num_of_batch)

            # do some calculations on each batch of data
            for i in range(num_of_batch):
                # calculate mape score
                score = mape(
                    pred_test_batches[i]['y_test'].values, pred_test_batches[i]['y_pred'].values)
                if score > mape_alert_thres:
                    mape_alert = f'Batch {len_of_batches+i} scored above threshold.\n' + mape_alert
                mape_scores = np.append(mape_scores, score)

                # perform ks test on each column with respect to that column in df_og
                column_scores = {}
                for column in df_ks_test.columns:
                    p_val = ks_2samp(
                        df_og[column], df_test_batches[i][column]).pvalue
                    if p_val < 0.05:
                        alert = f'distribution drift detected at batch {len_of_batches+i}.\n'
                        ks_alert[column] = alert + ks_alert[column]
                        num_dist_drifts += 1
                    column_scores[column] = [p_val]

                df_column_scores = pd.DataFrame(column_scores)
                df_ks_test = pd.concat(
                    [df_ks_test, df_column_scores]).reset_index().iloc[:, 1:]

            df_mape_batch = pd.DataFrame({'mape': mape_scores})
            df_mape = pd.concat([df_mape, df_mape_batch]
                                ).reset_index().iloc[:, 1:]
            df_mape['batch_number'] = list(df_mape.index)
            mape_mean = df_mape['mape'].mean()
            len_of_batches = len(df_mape)

            with placeholder.container():
                kpi1, kpi2 = st.columns(2)
                kpi1.metric('avg MAPE', mape_mean)
                kpi2.metric('num of distribution drift', num_dist_drifts)

                plot2 = st.expander(
                    "Mean Absolute Percentage Error", expanded=True)
                with plot2:
                    mape_plot, mape_log = st.tabs(['Plot', 'Log'])

                    with mape_plot:
                        st.line_chart(data=df_mape, x='batch_number', y='mape')

                    with mape_log:
                        st.text(mape_alert)

                plot3 = st.expander('KS test for distribution drift detection')
                with plot3:
                    column_tabs = st.tabs(drift_detect_features)
                    for i in range(len(drift_detect_features)):
                        column_tabs[i].text(ks_alert[drift_detect_features[i]])

            df_test = reg.regressionDataStream(len(df_test))
            time.sleep(2)

    if model_filter == 'Time Series Forecasting':
        # create a dataframe for all mse scores
        df_mape = pd.DataFrame(columns=['mape'])
        # create a dataframe for all ks test p values
        df_ks_test = pd.DataFrame(columns=drift_detect_features)
        mape_alert = ''
        ks_alert = {f: '' for f in drift_detect_features}
        num_dist_drifts = 0
        while True:
            df_all = pd.concat([df_all, df_test]).reset_index().iloc[:, 1:]
            y_test = df_test[target_feature]
            x_test = df_test.drop(columns=[target_feature])
            y_pred = model.predict(x_test)
            df_pred_test_batch = pd.DataFrame(
                {'y_pred': y_pred, 'y_test': y_test})
            df_pred_test = pd.concat(
                [df_pred_test, df_pred_test_batch]).reset_index().iloc[:, 1:]
            num_of_batch = ceil(len(df_test) / batch_size)
            mape_scores = np.array([])
            pred_test_batches = np.array_split(
                df_pred_test_batch, num_of_batch)
            df_test_batches = np.array_split(df_test, num_of_batch)

            # do some calculations on each batch of data
            for i in range(num_of_batch):
                # calculate mape score
                score = mape(
                    pred_test_batches[i]['y_test'].values, pred_test_batches[i]['y_pred'].values)
                if score > mape_alert_thres:
                    mape_alert = f'Batch {len_of_batches+i} scored above threshold.\n' + mape_alert
                mape_scores = np.append(mape_scores, score)

                # perform ks test on each column with respect to that column in df_og
                column_scores = {}
                for column in df_ks_test.columns:
                    p_val = ks_2samp(
                        df_og[column], df_test_batches[i][column]).pvalue
                    if p_val < 0.05:
                        alert = f'distribution drift detected at batch {len_of_batches+i}.\n'
                        ks_alert[column] = alert + ks_alert[column]
                        num_dist_drifts += 1
                    column_scores[column] = [p_val]

                df_column_scores = pd.DataFrame(column_scores)
                df_ks_test = pd.concat(
                    [df_ks_test, df_column_scores]).reset_index().iloc[:, 1:]

            df_mape_batch = pd.DataFrame({'mape': mape_scores})
            df_mape = pd.concat([df_mape, df_mape_batch]
                                ).reset_index().iloc[:, 1:]
            df_mape['batch_number'] = list(df_mape.index)
            mape_mean = df_mape['mape'].mean()
            len_of_batches = len(df_mape)

            with placeholder.container():
                kpi1, kpi2 = st.columns(2)
                kpi1.metric('avg MAPE', mape_mean)
                kpi2.metric('num of distribution drift', num_dist_drifts)

                plot2 = st.expander(
                    "Mean Absolute Percentage Error", expanded=True)
                with plot2:
                    mape_plot, mape_log = st.tabs(['Plot', 'Log'])

                    with mape_plot:
                        st.line_chart(data=df_mape, x='batch_number', y='mape')

                    with mape_log:
                        st.text(mape_alert)

                plot3 = st.expander('KS test for distribution drift detection')
                with plot3:
                    column_tabs = st.tabs(drift_detect_features)
                    for i in range(len(drift_detect_features)):
                        column_tabs[i].text(ks_alert[drift_detect_features[i]])

            df_test = multits.multiTSDataStream()
            time.sleep(2)

    if model_filter == 'Image Classification':
        df_metrics = pd.DataFrame(columns=[
                                  'percentage_valid', 'percentage_invalid', 'percentage_weak', 'percentage_wrong'])
        while True:
            batch_metrics = im.imageDataStream(df_path, batch_size)
            df_metrics = pd.concat(
                [df_metrics, batch_metrics]).reset_index().iloc[:, 1:]
            df_metrics['batch_number'] = list(df_metrics.index)
            percent_valid = df_metrics['percentage_valid'].mean()
            percent_invalid = df_metrics['percentage_invalid'].mean()
            percent_weak = df_metrics['percentage_weak'].mean()
            percent_wrong = df_metrics['percentage_wrong'].mean()

            with placeholder.container():
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric('Percentage Valid', round(percent_valid, 2))
                kpi2.metric('Percentage Invalid', round(percent_invalid, 2))
                kpi3.metric('Percentage Weak', round(percent_weak, 2))
                kpi4.metric('Percentage Wrong', round(percent_wrong, 2))

                plot2 = st.expander('Custom Metrics', expanded=True)
                with plot2:
                    st.line_chart(df_metrics, x='batch_number', y=list(df_metrics.columns))