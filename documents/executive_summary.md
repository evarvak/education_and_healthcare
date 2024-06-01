<!-- ## Executive summary
Healthcare access plays a major role in a child’s wellbeing and development. There are many factors contributing to children’s access to healthcare from financial barriers, such as health insurance coverage and family income, to nonfinancial barriers, such as family structure, cultural factors, health literacy of the parents, and the availability of healthcare providers. Healthcare access is shown to impact several aspects of child development, including physical, emotional, and mental health, as well as growth, and academic performance. 

Health care access is the ability to obtain healthcare services such as prevention, diagnosis, treatment, and management of diseases, illness, disorders, and other health-impacting conditions. It is one of the social determinants of health, and can affect a wide range of functioning, quality of life, and health outcomes. We are interested in exploring in particular the effect of healthcare access on academic performance. “Children with Medicaid access, for instance, have been found to have better grades, fewer missed days, better graduation rates, and higher long-term earnings.”1 

Our dataset consists of two parts: health dataset and education dataset. For the health data, we use The National Survey of Children’s Health (NSCH) dataset to generate our “access to healthcare” features. These will include features like children’s current health care coverage, how much money was paid for health related care, insurance type, government assistance, how often the child is allowed to see providers, whether there was a time the child needed healthcare but did not receive it, etc. For the education data, we plan to use a combination of two datasets: we will use the The National Survey of Children’s Health (NSCH) again to extract features like grades received in school across all subjects, days missed in school, and whether the children do all required homework. We also plan to use data from theEDFacts centralized data governed by the U.S. Department of Education.  In the academic achievement data (Mathematics FS175/ DG583 and Reading/Language Arts FS178/ DG584) posted by Office of Elementary and Secondary Education (OESE)states provide the count of students taking each type of assessment and scoring in each performance level by subject, and grade. 

The primary hypothesis of the project is that the academic performance features (APF), measured by test scores and number of missed school days, is negatively impacted by student access to health care, measured by the previously defined “access to healthcare” features (AHF). 

Despite the great effort made in healthcare access since the passage of the Affordable Care Act, access to quality health care is still affected by socioeconomic status. “Children from economically disadvantaged communities often lag behind their peers in more affluent communities in access to quality health care.”3  This is one of the reasons why we predict that several confounding factors will interact with each other to create a complex system that influences the relation between our variables. -->

# Healthy Bodies, Bright Minds: "The impact of Healthcare Access on Children's Academic Performance"
**May - Summer 2024, Erdos Institute Data Science Boot Camp project** 

# Executive Summary

## Problem and data selection
We are interested in exploring the effect of healthcare access on children's academic performance. Research shows that school absences have negative impact on grades and student's academic achievement. For that reason, we will use absenteeism as a metric of student educational outcome. We use the dataset from the National Survey of Children's Health (NSCH) dataset, and we extract from this dataset two sets of variables:
- Predicting variables:
     - "access to healthcare" features (e.g. children's current healthcare coverage, how often the child is allowed to see providers, etc),
     - health-related features (e.g. depression, children's general health),
- Target variable:
     - "days missed in school". We convert days_missed into a categorical variable: 0 means 0-6 days missed, 1 means 7+ days missed.


## Feature selection 1.0

Our dataset is a high-dimensional dataset with 29433 rows and 448 columns (447 features, 1 target). Feature selection is a [crucial](https://hex.tech/use-cases/feature-selection/#:~:text=Feature%20selection%20simplifies%20models%20by,to%20stakeholders%20or%20regulatory%20bodies.) step for our model as it reduces overfitting, improves accuracy, reduces computational costs, and aids interpretability. We use **three** different methods for feature selection: 
- **Handpick**: we parse through the 447 features in the NSCH dataset, picking any related to health and healthcare access,
- **Correlation analysis with the target variable** (supervised filter method): we compute the linear correlation between each feature and the number of days missed, keeping features with high correlation,
- **Histogram analysis** (supervised filter method): for each feature, we measured the change in histogram shape among children with low and with high absenteeism, keeping features with sufficiently different histograms. 

Through these methods, we refined our dataset to 88 features.

## Initial model selection
One of our goals was to create a simple model to classify students between "low absenteeism," which we define as missing 0-6 school days in a year, and "high absenteeism," which we define as missing 7 or more school days in a year. We start by splitting students into training and testing tests.  We train and evaluate a logistic regression classifier, a random forest classifier, a support vector classifier, and a KNN classifier to predict whether children will miss 7 or more school days. For each classifier (including a stratified dummy model as a baseline classifier), we calculate accuracy, precision, recall, average precision score (i.e., area under the precision-recall curve), and f1 score (harmonic mean of precision and recall) on the training and test data. For brevity, we only show the precision-recall curves of each classifier below, while the other performance metrics can be found by running the notebook [model_selection.ipynb)(https://github.dev/evarvak/education_and_healthcare/blob/master/notebook/model_selection.ipynb).

![](figures/pr1.png)<!-- -->
<!--![](figures/dummy_performance.png) -->
![](figures/pr3.png)<!-- -->
<!--![](figures/lr_performance.png) -->
![](figures/pr4.png)<!-- -->
<!--![](figures/rf_performance.png) -->
![](figures/svc_prec_rec.png)<!-- -->
<!--![](figures/svc_performance.png) -->
![](figures/knn_prec_rec.png)<!-- -->
<!--![](figures/knn_performance.png) -->

From the above, it seems logistic regression and random forest perform similarly and both outperform the other three classifiers. We use logistic regression since it's the more interpretable model of the two.

## Feature selection 2.0

### Correlation testing between features

It is important to understand the correlation between different variables and features in our model  for [several reasons:](https://medium.com/@abdallahashraf90x/all-you-need-to-know-about-correlation-for-machine-learning-e249fec292e9#:~:text=By%20analyzing%20correlations%2C%20researchers%20can,a%20model's%20ability%20to%20generalize.)
- Feature selection: by analyzing correlations, we can identify redundant features, and select a minimal set of important features that best represent our target varaibles. This prevents overfitting and improves our model's ability to generalize.
- Reducing bias: by identifying correlation between input features and sensitive attributes, we can evaluate our model for potential biases, monitor feature importance, and apply techniques like fair representation learning to mitigate bias.
- Detecting multicollinearity: highly linearly correlated features can negatively impact our models by increasing invariance and making it difficult to determine the significance and effect of individual predictors. 

The goal of this feature selection method is to automatically identify which features in our data set are highly correlated with each other, and systematically remove them. We start by using the clean version of our data set (that contains 84 features), and we drop some features that don't make sense to investigate correlation with, like the state and their FIPS code, as well as our target variable "days missed". We then create a series with values equal to the correlation of the multiindex of a pair of features, and we look at pair of features whose correlation is higher than a certain threshold, that we set to be 0.8. By defining the edges to be the indices of the different correlated features, and the weight to be the correlation between two pairs, we are able to create a graph from our edges and weights. In addition to clusters of two, the graph above shows two large clusters in our features.

![](figures/corr_clusters_with_labels.png)<!-- -->
                                                   
We eliminate all but one feature from each highly collinear “cluster” found. We compare the percent of missing entries for each our feature, and we decide to drop: 
- 'num_checkups' (this has more missing data than 'doctor_visit')
- 'birth_year' (this has more missing data than 'age')
- 'saw_nonmental_specialist' ('difficulty_with_specialist' is more related to healthcare access)

We should **keep** the following features in their respective cluster:
- 'currently_insured' (most directly related to healthcare access and is connected to all other features in cluster)

### Recursive feature elimination (RFE)
Next, we fit a logistic regression model to our refined dataset and visualize feature importance by plotting the fitted model coefficients. 
![](figures/feature_importance_all.png)<!-- -->

Many of these features do not seem to influence the model much. Next, we iteratively remove the feature with the smallest coefficient (in magnitude) until model performance starts to suffer. We do this using sklearn's recursive feature elimination (RFE) function while tracking the average precision score after each feature is dropped. 
![](figures/rf2.PNG)<!-- -->

For the sake of comparison with the results below, we compute the average precision score of the model with all features. 
- The average precision score on the test set with all 61 features is 0.42352262496485665.
- The average precision score on the test set while keeping 25 features is 0.42185069455482616.
- The average precision score on the test set while keeping 20 features is 0.41692541715139575.
- The average precision score on the test set while keeping 15 features is 0.41426164940946525.
- The average precision score on the test set while keeping 10 features is 0.4079551492111875.

The average precision with 25 features is nearly identical to the average precision of the model will all 61 features. Performace dips from there, though there is not much difference between the model with 20 and the model with 15 features. Let's look at what features we keep in each case.

The 25 most important features are 
['alternative_healthcare', 'anxiety', 'avoided_changing_jobs', 'breathing_problems', 'cut_hours', 'depression', 'doctor_visit', 'does_homework', 'emotional_problem', 'financial_problems', 'general_health', 'has_sick_place', 'headaches', 'health_affects_things', 'hostpital_er', 'hostpital_stay', 'memory_condition', 'needed_decisions', 'needed_referral', 'num_without_special_healthcare', 'physical_pain', 'recieved_food_stamps', 'recieved_welfare', 'reported_school_problems', 'stomach_problems']


The 20 most important features are 
['alternative_healthcare', 'anxiety', 'avoided_changing_jobs', 'breathing_problems', 'cut_hours', 'depression', 'doctor_visit', 'emotional_problem', 'financial_problems', 'general_health', 'health_affects_things', 'hostpital_er', 'hostpital_stay', 'memory_condition', 'needed_decisions', 'needed_referral', 'physical_pain', 'recieved_welfare', 'reported_school_problems', 'stomach_problems']


The 15 most important features are 
['alternative_healthcare', 'breathing_problems', 'cut_hours', 'depression', 'doctor_visit', 'financial_problems', 'general_health', 'hostpital_er', 'hostpital_stay', 'needed_decisions', 'needed_referral', 'physical_pain', 'recieved_welfare', 'reported_school_problems', 'stomach_problems']


The 10 most important features are 
['cut_hours', 'depression', 'doctor_visit', 'general_health', 'hostpital_er', 'hostpital_stay', 'needed_decisions', 'physical_pain', 'reported_school_problems', 'stomach_problems']

## Final model selection

We focus on the model with 10 features.  We can visualize the importance of each feature in each model by plotting the size of each coefficient in a bar graph. 
![](figures/feature_importance_10.png)<!-- -->

We can similarly visualize odds ratios. Odds ratios tell us the relative increase in the odds that a student will have high absenteeism due to a unit increase in the given feature. 
![](figures/odds_raios_10.png)<!-- -->

Overall, we found that poor health was strongly related to absenteeism.  
- Specifically, we found that a higher number of missed days was predicted by poorer general health and more time spent in the hospital, as well as the presence of depression, chronic physical pain, and digestive problems.
- Additionally, children who reported having problems at school, needed healthcare-related decisions made on their behalf, or experienced health problems for which their family needed to cut work hours were also found to be more likely to miss school

## Conclusion and future directions
- It looks like many features related to health highly affect absenteeism. The preliminary results suggest that access to health care is not the strongest predictor of child absenteeism.
- It is possible that the relationship between access to health care and absenteeism was drowned out by the more potent predictors of missed days, such as the general health of the child. A future study could control for predictors which are more related to access to healthcare
- Likewise, it is possible that absenteeism is a poor metric for education outcomes; future work could try other metrics, such as grades or scores on standardized tests
- Our data comes from 2019, so it is pre-COVID. It would be interesting to see if there was a more clear relationship between health care access and absenteeism in more modern, post-COVID data

