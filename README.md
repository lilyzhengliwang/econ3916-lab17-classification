# econ3916-lab17-classification

# Classification: When 90% Accurate Is Worthless

## Objective

I showed that accuracy is a poor measure when defaults are rare, and I chose a loan-decline threshold from the cost of each error instead of using the default 0.5.

## Methodology

- Simulated 6,000 loan applications with three features (log income, debt to income, number of prior defaults) and a known logistic formula for the chance of default.
- Split the data into training and test rows, keeping the default rate the same in both, and scored a rule that approves everyone as a baseline.
- Fitted a logistic regression with a scaler in a scikit-learn pipeline, and compared its accuracy and confusion matrix with the baseline at the usual 0.5 threshold.
- Refitted the model in statsmodels and read each coefficient as a log-odds change and as an odds ratio. I then applied one odds ratio to different starting probabilities to see that the same odds ratio gives different probability changes.
- Traced precision and recall across six thresholds, from 0.5 down to 0.04.
- Priced the two errors: approving a loan that defaults, and declining a borrower who would have repaid. I found the cheapest threshold by trying every value from 0.010 to 0.600 on the training rows (using 5-fold cross-validated probabilities), then priced that threshold once on the test rows. I compared it with the closed-form threshold, FP cost / (FP cost + FN cost).
- Wrote `precision_recall_at(model, X, y, t)`, a function that gives the precision and recall of any classifier on any rows at any threshold. It passed the check cell.

## Key Findings

- The default rate was 9.6%, so a rule that approves everyone is 90.4% accurate.
- The logistic regression's accuracy at 0.5 was 90.4%, the same as the baseline, with 0 predicted defaults. It declined nobody, so accuracy said nothing about whether the model had learned anything. The precision and recall at lower thresholds showed that it had.
- One more prior default has an odds ratio of 1.63. It is a multiplier on the odds, so the change in probability depends on where an applicant starts.
- The cost-minimising threshold, chosen on the training rows, was 0.121, against a closed-form 0.118.
- The expected cost was 944 per applicant on the test rows, against 1,147 at 0.5.
- At the closed-form threshold, precision was 0.181 and recall was 0.477.
