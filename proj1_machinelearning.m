close all; clear; clc
%%
train = readtable('dataclean1.csv');
test = readtable('test.csv');

train.Age = double(train.Age);
test.Age = double(test.Age);

% Set as float for precision
train.Fare = double(train.Fare);
test.Fare = double(test.Fare);

% Converted into a binary data - if 'S' = 1; 'C' | 'Q' = 0
train.Embarked = categorical(train.Embarked);
test.Embarked = categorical(test.Embarked);
train.Embarked = double(train.Embarked == 'S');
test.Embarked = double(test.Embarked == 'S');

% Get family same with dataclean1.csv
test.Family = test.SibSp + test.Parch + 1;

% Get the total number of family
train.TotalNumber = train.Family;
test.TotalNumber = test.Family;

% Fare per passenger
train.FarePerHead = train.Fare ./ train.TotalNumber;
test.FarePerHead = test.Fare ./ test.TotalNumber;

% Convert Sex - 1 if male, 0 if female
train.Sex = double(strcmp(train.Sex, 'male'));  
test.Sex = double(strcmp(test.Sex, 'male'));

% Dropped columns (categorical)
train.Name = [];
train.PrefixTicket1 = [];
train.PrefixTicket2 = [];
train.Cabin = [];

train
test
%% Random Forest
X_train = train;
X_train.Survived = []; % Dropped column as it is the target output, not an input feature
Y_train = train.Survived; % Target
X_test = test; % Copy dataset to X_test
X_train = normalize(X_train);

RandomForest = TreeBagger(30, X_train, Y_train, 'Method', 'classification');
[Y_predict, scores] = predict(RandomForest, X_test);
Y_predict = str2double(Y_predict);

Y_train_predict = predict(RandomForest, X_train);
Y_train_predict = str2double(Y_train_predict);
Accuracy1 = sum(Y_train_predict == Y_train) / length(Y_train);

figure(1);
Stats = confusionmatStats(Y_train, Y_train_predict);
fprintf('Classification Report:\n')
fprintf('\nAccuracy: %.20f\n', Accuracy1);
fprintf('Precision: %.20f\n', Stats.precision);
fprintf('Recall: %.20f\n', Stats.recall);
fprintf('F1-Score: %.20f\n', Stats.F1);

% Start of Hyperparameter Tuning for Random Forest
EstimatorValues = [10, 20, 30, 40, 50]; % Test values
MaxDepthValues = [10, 20, 30, 40, 50]; % Test values

BestAccuracy = 0;
BestEstimator = 0;
BestMaxDepth = 0;
BestPrecision = 0;
BestRecall = 0;
BestF1 = 0;

for Estimators = EstimatorValues
    for MaxDepth = MaxDepthValues
        RandomForest = TreeBagger(Estimators, X_train, Y_train, 'Method', 'classification', 'MaxNumSplits', MaxDepth);

        Y_train_predict = predict(RandomForest, X_train);
        Y_train_predict = str2double(Y_train_predict);

        Accuracy2 = sum(Y_train_predict == Y_train) / length(Y_train);

        ConfusionMatrix = confusionmat(Y_train, Y_train_predict);
        tp = ConfusionMatrix(2, 2);
        fp = ConfusionMatrix(1, 2);
        fn = ConfusionMatrix(2, 1);
        tn = ConfusionMatrix(1, 1);

        precision = tp / (tp + fp);
        recall = tp / (tp + fn);
        F1 = 2 * (precision * recall) / (precision + recall);

        if Accuracy2 > BestAccuracy
            BestAccuracy = Accuracy2;
            BestEstimator = Estimators;
            BestMaxDepth = MaxDepth;
            BestPrecision = precision;
            BestRecall = recall;
            BestF1 = F1;
        end
    end
end
fprintf('\nHyperparameter Tuning Results:\n')
fprintf('Best Accuracy: %.20f\n', BestAccuracy);
fprintf('Best Estimator: %d\n', BestEstimator);
fprintf('Best Max Depth: %d\n', BestMaxDepth);
fprintf('Best Precision: %.20f\n', BestPrecision);
fprintf('Best Recall: %.20f\n', BestRecall);
fprintf('Best F1-Score: %.20f\n', BestF1);
% End of Hyperparameter Tuning
%% Logistic Regression
% logreg = fitclinear(X_train, Y_train, 'Learner', 'logistic', 'Solver', 'lbfgs');
% Y_pred = predict(logreg, X_test);
% Y_train_predict = predict(logreg, X_train);
% Accuracy3 = sum(Y_train_predict == Y_train) / length(Y_train);
% 
% figure(2);
% Stats = confusionmatStats(Y_train, Y_train_predict);
% fprintf('\nClassification Report:\n')
% fprintf('Accuracy Score: %.10f\n', Accuracy3);
% fprintf('Precision: %.10f\n', Stats.precision);
% fprintf('Recall: %.10f\n', Stats.recall);
% fprintf('F1-Score: %.10f\n', Stats.F1);
% 
%%% Start of Hyperparameter Tuning for Logistic Regression
% SolverValues = {'sgd', 'asgd', 'dual', 'bfgs', 'lbfgs', 'sparsa'};
% LambdaValues = [0.01, 0.1, 1, 10];  % Regularization strength
% 
% BestAccuracyLogReg = 0;
% BestSolver = '';
% BestLambda = 0;
% BestPrecisionLogReg = 0;
% BestRecallLogReg = 0;
% BestF1LogReg = 0;
% 
% for Solver = SolverValues
%     for Lambda = LambdaValues
%         % Check if the 'dual' then 'LossFun' = 'hinge'
%         if strcmp(Solver{1}, 'dual')
%             logreg = fitclinear(X_train, Y_train, 'Learner', 'svm', 'Solver', Solver{1}, 'Lambda', Lambda, 'LossFun', 'hinge', 'Regularization', 'ridge');
%         % Check if 'sparsa' then 'Regularization' = 'lasso'
%         elseif strcmp(Solver{1}, 'sparsa')
%             logreg = fitclinear(X_train, Y_train, 'Learner', 'svm', 'Solver', Solver{1}, 'Lambda', Lambda, 'Regularization', 'lasso');
%         else
%             logreg = fitclinear(X_train, Y_train, 'Learner', 'logistic', 'Solver', Solver{1}, 'Lambda', Lambda);
%         end
% 
%         Y_train_predict = predict(logreg, X_train);
% 
%         AccuracyLogReg = sum(Y_train_predict == Y_train) / length(Y_train);
% 
%         ConfusionMatrix = confusionmat(Y_train, Y_train_predict);
%         tp = ConfusionMatrix(2, 2);
%         fp = ConfusionMatrix(1, 2);
%         fn = ConfusionMatrix(2, 1);
%         tn = ConfusionMatrix(1, 1);
% 
%         precision = tp / (tp + fp);
%         recall = tp / (tp + fn);
%         F1 = 2 * (precision * recall) / (precision + recall);
% 
%         if AccuracyLogReg > BestAccuracyLogReg
%             BestAccuracyLogReg = AccuracyLogReg;
%             BestSolver = Solver{1};
%             BestLambda = Lambda;
%             BestPrecisionLogReg = precision;
%             BestRecallLogReg = recall;
%             BestF1LogReg = F1;
%         end
%     end
% end
% 
% fprintf('\nHyperparameter Tuning Results for Logistic Regression:\n')
% fprintf('Best Accuracy: %.10f\n', BestAccuracyLogReg);
% fprintf('Best Solver: %s\n', BestSolver);
% fprintf('Best Lambda: %.2f\n', BestLambda);
% fprintf('Best Precision: %.10f\n', BestPrecisionLogReg);
% fprintf('Best Recall: %.10f\n', BestRecallLogReg);
% fprintf('Best F1-Score: %.10f\n', BestF1LogReg);
%%% End of Hyperparameter Tuning
%%
function stats = confusionmatStats(trueLabels, predictedLabels)
    ConfusionMatrix = confusionmat(trueLabels, predictedLabels);
    
    tp = ConfusionMatrix(2, 2);
    fp = ConfusionMatrix(1, 2);
    fn = ConfusionMatrix(2, 1);
    tn = ConfusionMatrix(1, 1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = 2 * (precision * recall) / (precision + recall);

    stats.precision = precision;
    stats.recall = recall;
    stats.F1 = F1;


    ConfusionMatrix
    confusionchart(ConfusionMatrix)
end