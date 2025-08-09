# DS 5220 Summer Project

## Setup

* This repo is based on Linux environment. If you are using Windows, you might have to setup dependencies on your own instead of using the environment.yml file in here. Other Linux commands should be able to work if you are using Anaconda Prompt, which shares most of the regular Linux commands. If not, you simply have to download and/or install some of the tools and convert dataset separately on Windows version instead of using the Make commands.

* Download and unzip [CMU MoCap](https://mocap.cs.cmu.edu/faqs.php) data by following command:
```
make data_download
```


* CMU MoCap data is kept in AMC format. We will convert this into BVH format using [amc2bvh](https://github.com/thcopeland/amc2bvh). Follow the installation guide from [amc2bvh](https://github.com/thcopeland/amc2bvh). The below command can be used on linux terminal to download the .tar file and unzip it to a binary file as in the installation guide:

```
install_amc2bvh
```

* Make sure [bvh-toolbox](https://github.com/OlafHaag/bvh-toolbox) is properly installed from importing the dependencies of this repo.

* Make sure to check if there is any update on [CMU MoCap](https://mocap.cs.cmu.edu/), [amc2bvh](https://github.com/thcopeland/amc2bvh) and/or [bvh-toolbox](https://github.com/OlafHaag/bvh-toolbox) after this repo has been updated.

## Generate Smaller Test Dataset
* Following command can be used to convert amc/asf files into bvh files in the first folder '01.' Use it to make sure everything is working fine:
```
make data_convert_bvh_01
```

* Now .bvh files can be converted to .csv files using [bvh-toolbox](https://github.com/OlafHaag/bvh-toolbox). This process may take a long time. This can be done by following command:
```
make data_convert_csv_01
```

* Proper [scaling](https://mocap.cs.cmu.edu/faqs.php) of the raw data is done during this process. Following command will merge previous csv files into a one single csv file:
```
make data_generate_3D_01
```

* Following command will generate 2D view of the previous 3D positional data, using a simple pinhole camera model with focal length of 5cm and 10m distance away from the object.
```
make data_generate_2D_01
```

* You can test if the generated dataset is working normal by plotting animation with following command:
```
make plot
```

## Regression on Small Dataset and Analysis

* You can set different range for 'param_grid' variable from the [source file](src/regression.py) to apply different grid search for each model. If you are having issue with parallel computing, change 'n_jobs' value of 'param_grid' variable. You can also comment out certain lines on the main script to perform only functions you want to run.
* Hyperparameter tuning on Linear regression, Ridge and Lasso of sklearn can be ran by following command:
```
make regression
```
* The result is as following:
```
Hyperparameter Tuning:

        Linear Regression
Best Parameters of Linear Regression: {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'positive': False}
Best mean cross-validation R^2 Score: 0.8626126009656108
Test R^2 score with best parameters: 0.8615260935923504
Time taken: 67.3483 s

        Ridge
Best Alpha of Ridge: {'alpha': 1e-07}
Best mean cross-validation R^2 Score: 0.8626126491394246
Test R^2 score with best parameters: 0.8615251359049926
Time taken: 1.3393 s

        Lasso
Best Alpha of Lasso: {'alpha': 0.005}
Best mean cross-validation R^2 Score: 0.7556416553476296
Test R^2 score with best parameters: 0.7520458868097597
Time taken: 19.8404 s

        KNN
Best Parameters of KNN: {'metric': 'manhattan', 'n_neighbors': 2, 'weights': 'distance'}
Best mean cross-validation MSE Score: 2.22690937059902e-05
Test MSE score with best parameters: 1.1356126601709736e-05
Time taken: 10.6285 s

        Random Forest
Best Parameters of Random Forest: {'ccp_alpha': 1e-07, 'max_depth': 10, 'max_features': 'sqrt', 'min_impurity_decrease': 1e-07, 'min_samples_split': 10, 'n_estimators': 400}
Best mean cross-validation MSE Score: 0.007381764532006477
Test MSE score with best parameters: 0.007185994783336368
Time taken: 399.9078 s

        XGBoosting
Best Parameters of XGBoosting: {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.15, 'max_depth': 7, 'n_estimators': 300, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 0.7}
Best mean cross-validation MSE Score: 0.0005681937094777822
Test MSE score with best parameters: 0.00047149998135864735
Time taken: 374.0592 s

        MLP
Best Parameters of MLP: {'alpha': 1e-06, 'hidden_layer_sizes': (512, 256, 128)}
Best mean cross-validation MSE Score: 0.0050587206035472496
Test MSE score with best parameters: 0.00462931338131516
Time taken: 265.3242 s
```
* As it can be seen, as alpha reduces on Ridge and Lasso, their R^2 score approaches to that of Linear regression, which means regularization is not really helping us in this case. In addition, Lasso starts to not converge as alpha gets lower than 1e-3. Thus, only linear regression would be considered in further analysis.
* The optimized parameters of Linear Regression is the same as that of the default option.
* KNN model provides the best R^2 score among above models.
* XGB model had the second best MSE score.
* Increasing size and depth of the MLP model seems to enhancing the result, but not by significant amount. They may be adjusted based on desicred accuracy and computational time.
* Increasing max_depth and n_estimators in Random Forest model also seems to enhancing the result, but not by significant amount. They may be adjusted based on desicred accuracy and computational time.
```
Save model trained with hyperparameters found:

 Training Linear Regression Model...
Training Finished! Time taken: 0.0924 s

 Training KNN Model...
Training Finished! Time taken: 0.1114 s

 Training Random Forest Model...
Training Finished! Time taken: 279.9517 s

 Training XGB Model...
Training Finished! Time taken: 79.2334 s

 Training MLP Model...
Training Finished! Time taken: 96.0963 s

 Training MLP Model 2...
Training Finished! Time taken: 36.3382 s

Test the time it takes to run the trained models:

 Testing Linear Regression Model...
R-squared: 0.8615260935923504
Testing Finished! Time taken: 0.0095 s

 Testing KNN Model...
MSE: 1.1356126601709736e-05
Testing Finished! Time taken: 1.8219 s

 Testing Random Forest Model...
MSE: 0.00018879443550074627
Testing Finished! Time taken: 4.6577 s

 Testing XGB Model...
MSE: 0.00047149998135864735
Testing Finished! Time taken: 1.9377 s

 Testing MLP Model...
MSE: 0.00462931338131516
Testing Finished! Time taken: 0.0609 s

 Testing MLP Model...
MSE: 0.009532699207365028
Testing Finished! Time taken: 0.0206 s
```
* Time it takes to train the model was significantly high on random forest model, XGB model and MPL model.
* Linear regression model had the least performance. Linear regression model may not be considered in further analysis.
* KNN model showed the best performance. However, it won't scale well as we increase the number of data points.
* Random Forest model showed the second best performance. However, it took the longest time for the prediction.
* XGB model showed the third best performance, but the time it takes for prediction is still in the scale of seconds.
* MLP model showed the forth best performance, which were still very good. It's best advantage was that the time it takes for prediction is in the scale of 0.01s, which can be used for real-time application.
* Overall, it seems to be the models are the fastest in the order of MLP, XGB and Random Forest, and the models are the most accurate in the order of Random Forest, XGB and MLP.
- The time shown here is based on the models applied to dataset in the scale of ~10,000 data points. In Real-Time situation, assuming 100FPS (very good FPS), the time should be divided by roughly 100. Then, the time for prediction for Real-Time become roughly:
  - Random Forest: ~0.05s
  - XGB: ~0.02s
  - MLP: ~0.0005s

* Last two MLP models are different by hidden layer sizes of (512, 256, 126) and (256, 128, 64). The performance was worsen by twice, but the speed of prediction got faster by three times. The structure of hidden layers may be adjusted depending on how much latency and accuracy are required.

## Regression on Full Dataset and Analysis
```
 Training MLP Model-All by chunks...
Time taken: 208.6011 s
Testing model...
MSE: 0.030062122270464897
Time taken: 0.8196 s

 Testing Model-All on small dataset...
MSE: 0.08077887807904548
Testing Finished! Time taken: 0.0476 s

 Training XGB Model-All...
Time taken: 2645.4371 s
Testing model...
MSE: 0.013023355975747108
Time taken: 51.2431 s

 Testing Model-All on small dataset...
MSE: 0.03426402807235718
Testing Finished! Time taken: 1.7026 s

 Training Random Forest Model-All...
Time taken: 15405.8409 s
Testing model...
MSE: 0.05542476930066533
Time taken: 80.3925 s

 Testing Model-All on small dataset...
MSE: 0.1836885355328568
Testing Finished! Time taken: 0.8513 s

 Training KNN Model-All...
Time taken: 31.7444 s
Testing model...
MSE: 1.0926668584648443e-05
Time taken: 11058.5858 s

 Testing Model-All on small dataset...
MSE: 1.9627092363855956e-06
Testing Finished! Time taken: 927.5214 s
```
- MLP model
  - MSE for smaller dataset: ~0.08
  - Prediction time for 100FPS: ~0.0005s
  - Less accurate, but the fastest for real-time application.
- XGB
  - MSE for smaller dataset: ~0.03
  - Prediction time for 100FPS: ~0.02s
  - The most accurate. Suitable for real-time application, but the highest latency.
- Random Forest
  - MSE for smaller dataset: ~0.05
  - Prediction time for 100FPS: ~0.01s
  - The second most accurated. Suitable for real-time application, but in the same scale of latency with that of XGB model.
- KNN
  - MSE for smaller dataset: ~2e-6
  - Prediction time for 100FPS: ~10s
  - Very accurate, but not suitable for real-time application.
- Conclusion
  - Use MLP model for real-time application where low latency is cruicial, where interaction with multiple people in sink is cruicial, such as multi-player gaming.
  - Use XGB for real-time application where accurate motion capture is cruicial, such as virtual performances.
## Motion Capture Using MediaPipe

## UDP Data Streaming

## Unity

## References