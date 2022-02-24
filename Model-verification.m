trainingdata = [ 5.86, 0.74;
      1.34, 1.18;
      3.65, 0.51;
      4.69, -0.48;
      4.13, -0.07;
      4.87, 0.37;
      7.91, 1.35;
      5.57, 0.30;
      7.30,1.64;
      7.89, 1.75
];
testdata = [
      5.80, 0.93;
      0.57, 1.87;
      4.30,-0.06;
      6.55, 1.60;
      0.82, 1.22;
      3.72, 0.90;
      5.80, 0.93;
      3.26, 1.53;
      6.75, 1.73;
      4.77,-0.51
            ];

Xtrain = trainingdata(:,1);
Ytrain = trainingdata(:,2);
Xtest = testdata(:,1);
Ytest = testdata(:,2);
X1 = [ones(length(Xtrain),1) Xtrain];
XT = transpose(X1);
YtrainT = transpose(Ytrain);
theta = ( XT * X1)^(-1) * (XT * Ytrain);
format long;
firstderiv = 2* XT * (X1 * theta - Ytrain );
secondderiv = 2 * XT * X1;
Y = poly2sym(flip(theta));
f1 = figure;
scatter(Xtrain,Ytrain);
hold on;
fplot(Y, [0 8]);

Y_prediction = polyval(flip(theta),Xtest);
%%
% 
% * ITEM1
% * ITEM2
% 
xlabel('X');
ylabel('Y');
title('1a) Linear Regression');
grid on

MSE1 = (sum((Y_prediction - Ytest).^2))/(length(Ytest))
RMSE1 = sqrt(MSE1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X2 = [ones(length(Xtrain),1) Xtrain Xtrain.^2];
XT2 = transpose(X2);
theta2 = ( XT2 * X2)^(-1) * (XT2 * Ytrain)
format long;
firstderiv2 = 2* XT2 * (X2 * theta2 - Ytrain );
secondderiv2 = 2 * XT2 * X2;
Y2 = poly2sym(flip(theta2)) ;
f2 = figure;
scatter(Xtrain,Ytrain);
hold on;
fplot(Y2, [0 8]);
Y_prediction2 = polyval(flip(theta2),Xtest);
xlabel('X');
ylabel('Y');
title('1b) Linear Regression Curve');
grid on
MSE2 = (sum((Y_prediction2 - Ytest).^2))/(length(Ytest))
RMSE2 = sqrt(MSE2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X3 = [ones(length(Xtrain),1) Xtrain Xtrain.^2 Xtrain.^3 Xtrain.^4];
XT3 = transpose(X3);
theta3 = ( XT3 * X3)^(-1) * (XT3 * Ytrain)
format long;
firstderiv3 = 2* XT3 * (X3 * theta3 - Ytrain );
secondderiv3 = 2 * XT3 * X3;
Y3 = poly2sym(flip(theta3)) ;
f3 = figure;
scatter(Xtrain,Ytrain);
hold on;

fplot(Y3, [0 8]);

xlabel('X');
ylabel('Y');
title('1c) Linear Regression polynomial');
grid on
%idx= find(Xtest)
Y_prediction3 = polyval(flip(theta3),Xtest);
MSE3 = (sum((Y_prediction3 - Ytest).^2))/(length(Ytest))
RMSE3 = sqrt(MSE3)
%%%%%%%%%%%%%%%%%%
% Create PNG files
%%%%%%%%%%%%%%%%%%
if 0                    
% Print PNG files
print -dpng -f1 fig1
print -dpng -f2 fig2
print -dpng -f3 fig3
end