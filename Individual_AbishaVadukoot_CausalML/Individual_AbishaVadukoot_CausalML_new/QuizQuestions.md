**Part 4: Quiz Questions - Causal Inference in Machine Learning**



---



\*\*Question 1:\*\* People who carry umbrellas tend to get sick more often. A researcher concludes that carrying umbrellas causes illness. What is the flaw in this reasoning?



A) The sample size is too small

B) A confounder (rainy weather) causes both umbrella-carrying and illness

C) Illness is a collider between umbrellas and weather

D) The researcher should have used a larger dataset



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* Rainy weather is a confounder — it makes people carry umbrellas \*and\* increases the likelihood of getting sick (due to cold, damp conditions). The researcher mistook a confounded association for a causal effect. Option A and D are about data quantity, not causal structure. Option C misidentifies the causal role: illness is the outcome, not a collider.



---



\*\*Question 2:\*\* In the potential outcomes framework, the fundamental problem of causal inference is that:



A) We cannot collect enough data to estimate treatment effects

B) Randomized experiments are impossible to conduct

C) We can never observe both potential outcomes for the same individual

D) Confounders always exist in observational data



\*\*Correct Answer:\*\* C



\*\*Explanation:\*\* For any individual, we observe either the outcome under treatment or the outcome under control, never both. The counterfactual (what would have happened) is always missing. This is the core problem that all causal methods try to work around. Option A is a practical concern, not the fundamental problem. Option B is false: experiments are possible, just not always feasible. Option D is common but not the \*fundamental\* problem, as confounding can be addressed if identified.



---



\*\*Question 3:\*\* In a DAG, a collider is a variable that:



A) Causes both the treatment and the outcome

B) Is caused by both the treatment and the outcome

C) Lies on the causal path between treatment and outcome

D) Is unrelated to both treatment and outcome



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* A collider receives arrows from both the treatment and the outcome (or their causes). Option A describes a confounder, not a collider. Option C describes a mediator. Option D describes an irrelevant variable. The critical property of a collider is that adjusting for it \*introduces\* bias rather than removing it, which is the opposite of what happens with confounders.



---



\*\*Question 4:\*\* A researcher adjusts for a collider in their causal analysis. What happens?



A) Confounding bias is removed

B) The estimate becomes doubly robust

C) A spurious association between treatment and outcome is created

D) The estimate moves closer to the true causal effect



\*\*Correct Answer:\*\* C



\*\*Explanation:\*\* Adjusting for a collider opens a non-causal path between treatment and outcome, creating a false association that didn't exist before. This is called collider bias or "explaining away." Options A and D describe what happens when you correctly adjust for a confounder, not a collider. Option B refers to a specific estimation method, not a consequence of collider adjustment.



---



\*\*Question 5:\*\* What does the backdoor criterion help us determine?



A) Whether the treatment has a causal effect on the outcome

B) Which variables to adjust for to block non-causal paths between treatment and outcome

C) The sample size needed for a causal study

D) Whether the data is missing completely at random



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* The backdoor criterion, introduced by Judea Pearl, identifies which set of variables we need to condition on to block all "backdoor" (non-causal) paths from treatment to outcome, giving us an unbiased estimate. It doesn't tell us whether an effect exists (A), that requires estimation. It has nothing to do with sample size (C) or missing data mechanisms (D).



---



\*\*Question 6:\*\* In predictive ML, dropping a feature with low predictive power is common practice. In causal ML, this can be problematic because:



A) It reduces the R-squared of the model

B) The dropped feature might be a confounder whose removal biases the treatment effect

C) It makes the model harder to interpret

D) Regularization methods won't work without all features



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* A variable with weak predictive power for the outcome might still be a confounder that strongly influences treatment assignment. Dropping it removes our ability to adjust for that confounding, biasing the treatment effect estimate. This is a key difference between causal and predictive data preparation. Options A, C, and D are concerns about model performance or interpretability, not about causal bias.



---



\*\*Question 7:\*\* A propensity score represents:



A) The probability that an individual has a positive treatment effect

B) The probability that an individual receives the treatment, given their observed covariates

C) The expected outcome under treatment for an individual

D) The difference between treated and control group means



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* The propensity score is defined as e(X) = P(T=1|X), the probability of receiving treatment given observed characteristics. It compresses all observed confounders into a single number for balancing purposes. Option A confuses propensity with treatment effect. Option C describes a predicted potential outcome, not a propensity score. Option D describes a naive estimator.



---



\*\*Question 8:\*\* Why is high accuracy in a propensity score model a potential warning sign?



A) It means the model is overfitting to noise

B) It indicates that treated and control groups are very different, suggesting poor overlap

C) It proves that confounders have been eliminated

D) It means the treatment was randomly assigned



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* If we can nearly perfectly predict who received treatment from covariates, it means the groups are so different that treatment assignment is almost deterministic. This implies poor overlap (positivity violations), which makes causal estimation unreliable. In a randomized experiment, treatment would be unpredictable from covariates (accuracy near 50%). Option A is possible but not the primary concern. Option C is incorrect, as high accuracy doesn't eliminate confounding. Option D is the opposite of what high accuracy implies.



---



\*\*Question 9:\*\* In Inverse Probability Weighting (IPW), extreme propensity scores (close to 0 or 1) are problematic because:



A) They indicate that the outcome model is misspecified

B) They produce very large weights that make the estimate unstable

C) They prove that the treatment has no effect

D) They mean the backdoor criterion is not satisfied



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* When a propensity score is near 0, the weight 1/e(X) becomes very large. A single observation with an extreme weight can dominate the entire estimate, making it unstable and unreliable. This is why practitioners clip propensity scores to a range like \[0.01, 0.99]. Option A confuses the propensity model with the outcome model. Options C and D are unrelated to extreme weights.



---



\*\*Question 10:\*\* A doubly robust estimator is called "doubly robust" because:



A) It uses two different datasets for estimation

B) It is consistent if either the propensity score model or the outcome model is correctly specified

C) It requires twice as many observations as other methods

D) It adjusts for both observed and unobserved confounders



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* Doubly robust estimation combines a propensity score model with an outcome model. The key property is that the estimate remains consistent if \*either\* model is correct, giving you two chances to get it right. Option A is incorrect, as it uses one dataset. Option C is incorrect, as sample size requirements aren't doubled. Option D is incorrect, as no observational method can adjust for unobserved confounders without additional assumptions.



---



\*\*Question 11:\*\* Double Machine Learning (DML) uses cross-fitting (sample splitting) primarily to:



A) Increase the sample size available for estimation

B) Prevent overfitting bias in the nuisance model estimates

C) Ensure that the treatment is randomly assigned

D) Reduce computation time



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* In DML, ML models predict both the outcome and treatment from covariates. If these predictions are made in-sample (using the same data used to fit the model), overfitting in the nuisance models can leak into the causal estimate. Cross-fitting prevents this by predicting each observation using a model trained on other observations. Option A is wrong, as splitting reduces the training data per fold. Option C is wrong, as cross-fitting doesn't change treatment assignment. Option D is wrong, as cross-fitting actually increases computation.



---



\*\*Question 12:\*\* What does a Causal Forest estimate that traditional propensity score methods do not?



A) The Average Treatment Effect (ATE)

B) The propensity score for each individual

C) Conditional Average Treatment Effects (CATEs), showing how effects vary across individuals

D) Whether confounders are present in the data



\*\*Correct Answer:\*\* C



\*\*Explanation:\*\* Causal Forests are specifically designed to estimate heterogeneous treatment effects, revealing how the treatment effect differs based on individual characteristics. Traditional propensity score methods (matching, IPW) estimate a single average effect (ATE or ATT) without revealing variation. Option A is what traditional methods already provide. Option B is an input to propensity methods, not an output of Causal Forests. Option D requires domain knowledge and DAGs, not a specific estimation method.



---



\*\*Question 13:\*\* When handling missing data in a causal study, why is MNAR (Missing Not at Random) the most dangerous scenario?



A) Because it reduces the dataset size significantly

B) Because the probability of missingness depends on the missing value itself, which no standard imputation can fully correct

C) Because it only occurs in randomized experiments

D) Because it means all variables have missing values



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* In MNAR, the reason data is missing is related to the unobserved value. For example, high earners not reporting income \*because\* it's high. Standard imputation methods assume missingness depends on observed data (MAR), so they cannot correct MNAR bias. Option A describes a consequence of any missing data. Option C is incorrect, as MNAR can occur in any study. Option D is incorrect, as MNAR doesn't require all variables to be affected.



---



\*\*Question 14:\*\* Using target encoding (replacing categories with the mean outcome for each category) is risky in causal inference because:



A) It increases the number of features in the dataset

B) It leaks outcome information into the covariates, potentially distorting treatment effect estimates

C) It only works with binary variables

D) It violates the backdoor criterion



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* Target encoding uses the outcome variable to create feature values. This means the encoded feature already contains information about the outcome, which can create artificial associations between covariates and the treatment effect. If a category has high outcomes because of the treatment (not the category itself), target encoding bakes that effect into the feature. Option A describes one-hot encoding, not target encoding. Option C is incorrect, as target encoding works with any categorical variable. Option D is incorrect, as encoding doesn't change the DAG structure, but it can bias the estimation.



---



\*\*Question 15:\*\* A researcher estimates an ATE of 5.0 for a new drug. A colleague argues that this single number is insufficient for policy decisions. The colleague's reasoning is most likely that:



A) The ATE should be reported as a confidence interval instead of a point estimate

B) The ATE might hide significant heterogeneity, where the drug may help some patients enormously while having no effect on others

C) The ATE can only be estimated using randomized experiments

D) The ATE does not account for the cost of the drug



\*\*Correct Answer:\*\* B



\*\*Explanation:\*\* An ATE of 5.0 could mean everyone benefits by exactly 5 units, or it could mean half the population benefits by 10 while the other half gets 0. These two scenarios have identical ATEs but very different policy implications. The colleague is arguing for estimating heterogeneous treatment effects (CATEs) rather than relying on a single average. Option A is about statistical uncertainty, which is important but not what the colleague is arguing. Option C is incorrect, as ATE can be estimated from observational data. Option D is about cost-effectiveness, not treatment effect heterogeneity.



---

