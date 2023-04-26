# SDS384 Project

<img src="https://github.com/jeiloh/SDS384-Project/blob/main/e-scooter_austin.png" width=50% height=50%>
Source: Ride Report Micromobility Dashboard | Austin, Texas

## E-scooter Demand Prediction in Austin

Team 1: Ziyu Fan, Zizhe Jiang, Jeil Oh

## Goals
The primary objectives of this work are the following:
* We construct a regression problem for hourly E-scooter demand prediction.
* We evaluate the predictive performance of five frequently used models (Linear Regression, Ridge regression, Gaussian Process Regression, Random Forest, XGBoost (eXtreme Gradient Boosting), Neural Network) based on the real-world dataset.
* We examine the connections between demographic data features and scooter demand prediction through model interpretation techniques to improve our understating of electric scooter demand.

## Data set
Shared Micromobility Vehicle Trips
https://data.austintexas.gov/Transportation-and-Mobility/Shared-Micromobility-Vehicle-Trips/7d8e-dm7r

This dataset contains shared micromobility vehicle trip data reported to the City of Austin Transportation Department as part of the Shared Small Vehicle Mobility Systems operating rules. The dataset has time-related information and the the GEOID of the 2010 US Census Tract in which the trip originated and ended. 

## Contents
`data`: shared micromobility vehicle trip data

`ct data`: United States Census Bureau’s American Community Survey

`scripts`: Exploratory Analysis & ML regression models

`report`: Final project report

## References
[1] Ride Report | Micromobility Dashboard. [public.ridereport.com/austin?x=-97.7335137.](https://public.ridereport.com/)
