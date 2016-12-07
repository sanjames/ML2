function sanPlotData()
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
%Find Indices of Positive and Negative Examples
%pos = find(y==1); neg = find(y == 0);
% Plot Examples
%plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
%'MarkerSize', 7);
%plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
%'MarkerSize', 7);

data = load('ex2data1.txt');
X1 = data(:, 1);
X2 = data(:, 2);
y = data(:, 3);

pos = find(y==1);
disp(pos);
neg = find(y==0);
scatter(X1(pos,1),X1(pos,2));
% =========================================================================

hold off;

end
