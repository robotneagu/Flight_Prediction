# Flight_Prediction

## Dataset Overview

The Flight Prices dataset consists of flight tickets advertised from April 16, 2022 through October 5, 2022. These tickets are associated with airports only located in the United States. Each ticket is vastly distinct from each other given the diversity of options of which airline to fly with, time of departure, price of ticket, and more which can found in the Dataset Source. The dataset contains 82,138,753 with flight prices ranging from $20 to $8260.61.

### Dataset Source:

Flight Prices: https://www.kaggle.com/datasets/dilwong/flightprices/data

## Goal of Project and Approach

The project focus is to develop a machine learning model that predicts the price of a flight ticket given its set of features. The price of the ticket is the label column, encouraging linear regressions to be utillized. Given this dataset is big data, PySpark was used as the main code when working with the data. PySpark was ran on Google Cloud Platform's DataProc cluster API. Models deployed include Linear Regression, Ridge Regression, Lasso Regression, and Lasso Ridge Regression.

## Data Acquisition

1. **Obtain Kaggle API Token:** Create a Kaggle API token file through account settings. Download the JSON file to a local machine.
2. **Revisit the Dataset in Kaggle:** Revisit the flight dataset on Kaggle and store the Copy API Command.
3. **Create a Virtual Machine on Google Cloud Platform**: The Compute Engine API was used to create a VM with the e2-standard-4 machine type, 60GB of storage, hosted in Iowa, and ran the Debian 12 OS.
4. **Transfer the Kaggle API file:** In the VM, a Kaggle directory was made with the Kaggle JSON file uploaded to that folder.
5. **Install Additional Packages:** Software such as ZIP, Python, and Kaggle cli tools were installed to use commands that can transfer the dataset directly to the VM.
6. **Download the Dataset:** Kaggle's API was used to download the dataset from Kaggle's server straight to the VM.
7. **Unzip the Compressed File:** The unzip command was executed to obtain the raw dataset as a .csv file
8. **Transfer the Dataset to the Bucket** After authentication, the raw CSV file was copied to the landing folder under the project bucket. Additional folders such as code, cleaned, models, and trusted were created for later files.
9. **Verify the Transfer was Successful** The ls command was used to list all items in the landing folder which showed the CSV, indicating a successful transfer.
10. **Delete the VM** The VM is no longer needed, deleting it saves costs for later operations.

## EDA and Data Cleaning

The following observations were made when performing EDA and Data Cleaning:
- About 10% of data contained NULL values. It is best to remove these records as filling in the values can bring inaccuracies and be a difficult process with many string values present. Having over 70 million records can still be considered sufficient to analyze as big data.
- Extreme values were present for totalFare, having prices as low as $20 and over $8000.
- Basefare and Totalfare have a perfect correlation. Basefare is not needed given that Totalfare provides the final cost of the ticket.
- 'isRefundable' had nearly all values as 0, making the column insignificant.
- The schema was pre-defined, requiring no adjustments on datatypes for columns
- legId', 'fareBasisCode' and 'segmentsAirlineName' have redundant information for either having unique values for each row or repeating information from another column.
- 'segmentsDepartureTimeRaw', 'segmentsArrivalTimeRaw', 'segmentsDepartureTimeEpochSeconds', 'segmentsArrivalTimeEpochSeconds'
have their time summarized in the ‘travelDuration’ column, giving no significant purpose to keep.

## Feature Engineering

New columns were created to compress columns and make them numeric:
- 'daysBetweenFlight' is the difference between the 'flightDate' and 'searchDate'. The latter two columns are then not used.
- 'flightDate_Weekday' finds the day of the week of the flight. This column is then used for 'flightDate_OnWeekend' which binarizes the data. 'flightDate_Weekday' then becomes irrelevant to use.
- From the segments column, only 'segmentsArrivalAirportCode' and 'segmentsCabinCode' were used as these provided useful information while being simple to encode.
- 'totalTravelDistance' used the MinMax Scalar to make its values more like a normal distribution.
- String columns used the StringIndexer, categorical columns underwent One-Hot enconding. The resulting columns, including those not adjusted, were put together in one Vector through the VectorAssembler.
- For each machine learning model, its regression estimator and evaluator were implemented in the Feature Engineering pipeline, resulting in 4 unique transformed dataframes.

## Machine Learning and Model Analysis

- Models were split in 70% training and 30% testing. A cross validator with 3 folds was used to validate consistent results.
- All models performed near identically with a RMSE of about 126, R-squared of 0.58, and intercept of about 213. Lasso Ridge Regression technically performed the best with an R-squared of 0.59 yet this is not a signficant improvement. This model was used for all remaining analysis.
- The most important features were 'totalTravelDistance' and 'flightDate_OnWeekend' yet their negative coefficients do not make logical sense.
- Visualizations of the model demonstrate extreme values still exist in the cleaned dataset with the data being slightly right skewed.
- Despite a high kurtosis, the model shows the potential to be normally distributed is fine-tuned.

## Conclusion

The results indicate that it cannot be used reliably as a reference for checking fraudulent fare prices. Having a weak R-squared, tight distribution of data, and coefficients that appear off makes the results not statistically significant for real world use. The model should be fine-tuned and reevaluated to remove extreme values, increase the R-squared to explain more data with confidence, and have the coefficients make more sense with its respective feature. Doing so could result in a better model that may be more useful in predicting flight prices.
