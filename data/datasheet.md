# Datasheet: Air pollutant concentrations in New Zealand

Author: Fabian Löw
Date: June 2023

Files (input data):
	mfe-particulate-matter-25-concentrations-2006-2021-DBF.zip
	mfe-particulate-matter-10-concentrations-2004-2021-DBF.zip


## Motivation

The datasets are a collection of measurements of various pollutants at several measurement stations across New Zealand. The dataset was downloaded from the data repository of the Ministry for the Environment (MfE)[https://data.mfe.govt.nz/]

The datasets were analyzed but are not provided here in this repository. Please login to the MfE repository, download the dataset(s) and unzipp them manually. The workflow of this projects reads and processed the dbf-files, which are contained in the downloaded zip-files from the MfE repository.


## Composition

There are two datasets in total.

1. mfe-particulate-matter-25-concentrations-2006-2021-DBF.zip
	This is the main dataset that contains the measurements for various stations in New Zealand from 2006-2021. The dataset includes the following columns: site, date, council, method, pm2_5, complete_for_trend, complete_for_mean, complete_year. Some of the columns are missing some values, including the measurements of the pollutant.
	
2. mfe-particulate-matter-10-concentrations-2004-2021-DBF.zip
	This is the main dataset that contains the measurements for various stations in New Zealand from 2004-2021. The dataset includes the following columns: site, date, council, method, pm10, complete_for_trend, complete_for_mean, complete_year. Some of the columns are missing some values, including the measurements of the pollutant.