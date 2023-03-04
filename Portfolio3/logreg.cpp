/*
Name: Bushra Rahman
Class: CS 4375.004
Assignment: ML Algorithms from Scratch - Logistic Regression
Due date: 3/4/23
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <regex>
#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono; // used to measure time
string fileName = "titanic_project.csv";


// Sigmoid function
double sigmoid(double z)
{
    double sigmoidValue = 1.0 / (1+exp(-1*z));
    return sigmoidValue;
}


// Vector print function for debugging
void vectorPrint(vector<int> vect, int n)
{
    for(int i=0; i < n; i++) {
        cout << vect.at(i) << " ";
    }
} // End of vectorPrint()



// main()
int main(int argc, char** argv)
{
    // Begin timer
    auto start = high_resolution_clock::now();

    // Steps to read in csv file
    ifstream inFS;                 // Input file stream
    string heading;
    string Xnum_in, pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1050;
    vector<int> Xnum(MAX_LEN);
    vector<int> pclass(MAX_LEN);
    vector<int> survived(MAX_LEN);
    vector<int> sex(MAX_LEN);
    vector<double> age(MAX_LEN);   // Some age values are doubles

    
    cout << "\nOpening file " + fileName << endl;
    // Try to open file
    inFS.open(fileName);
    if(!inFS.is_open())
    {
        cout << "ERROR: Could not open file " + fileName << endl;
        return 1;   // 1 indicates error
    }

    // Can now use inFS stream like cin stream

    // first row of titanic_project.csv is the heading: "","pclass","survived","sex","age"
    getline(inFS, heading);

    regex re("\"");  // Regex object for double quotes, so that they can be removed from Xnum
    // titanic_project.csv should contain 1 numerical string, 3 ints, and 1 int or double
    // titanic_project.csv should contain 1046 observations
    int numObservations = 0;
    while (inFS.good())
    {
        getline(inFS, Xnum_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');
        
        // stoi() converts string to int
        Xnum.at(numObservations) = stoi(regex_replace(Xnum_in, re, ""));  // replace double quotes with ""
        pclass.at(numObservations) = stoi(pclass_in);
        survived.at(numObservations) = stoi(survived_in);
        sex.at(numObservations) = stoi(sex_in);
        age.at(numObservations) = stof(age_in);     // stof() converts string to float

        numObservations++;
    }

    Xnum.resize(numObservations);
    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    cout << "Closing file " + fileName << endl << endl;
    inFS.close();   // Done with file, so close it

    /*
    // Vector printing for debugging
    cout << "\nVector names: " << heading << endl;
    cout << "Number of records: " << numObservations << endl;
    int size = numObservations;  // can use size in vectorPrint() like args for head() and tail() in R
    cout << "\nElements of XNUM:    "; vectorPrint(Xnum,size);
    cout << "\nElements of PCLASS: "; vectorPrint(pclass,size);
    cout << "\nElements of SURVIVED: "; vectorPrint(survived,size);
    cout << "\nElements of SEX: "; vectorPrint(sex,size);
    cout << "\nElements of AGE: ";
    for(int i=0; i < size; i++) {
        cout << age.at(i) << " ";
    }
    */
    
    // Next step: logistic regression on target=survived, predictor=sex, train=first 800 observations

    int train = 800;  // Size of training dataset, can be adjusted to different sizes
    int test = numObservations - train;
    // cout << "numObservations = " << numObservations << "\ntrain = " << train << "\ntest = " << test << endl;

    // Initialize the coefficients w0 and w1 to 1
    double weights[] = {1.0, 1.0};

    // Initialize data matrix: # of rows = train, # of columns = 2 (bc weights has 2 elements)
    int dataMatrix[train][2];
    // Its transpose: # of rows = 2, # of columns = train
    int transpose_dataMatrix[2][train];

    for(int i = 0; i < train; i++)
    {
        // Populate dataMatrix: col1 = all 1s, col2 = predictor values
        dataMatrix[i][0] = 1;              // All 1's, to be multiplied by the intercept w0
        dataMatrix[i][1] = sex.at(i);      // Predictors, to be multiplied by the coefficient w1

        // Populate transpose_dataMatrix: row1 = all 1s, row2 = predictor values
        transpose_dataMatrix[0][i] = 1;
        transpose_dataMatrix[1][i] = sex.at(i);
        // Transpose is used later for Step 3 of Gradient Descent
    }

    /*
    // Array print for debugging
    for(int i = 0; i < train; i++)      // Outer loop for rows
    {
        for (int j = 0; j < 2; j++)     // Inner loop for columns
        {
            cout << dataMatrix[i][j] << " ";
        }
        cout << endl;
    }
    */
    
    // Get factor values (either 0 or 1) for target
    int labels[train];
    for(int i = 0; i < train; i++)
    {
        labels[i] = survived.at(i);
        // cout << labels[i] << endl;
    }

    // Set up variables for gradient descent
    int iterations = 500000;        // Number of iterations, can be adjusted for accuracy
    double learningRate = 0.001;    // Also knows as alpha
    double gradient[] = {0.0, 0.0}; // gradient vector used when updating weights
    double probVector[train];       // vector of probabilities, initialized to 0
    double errorVector[train];      // vector of errors, initialized to 0
    for(int i = 0; i < train; i++)
    {
        probVector[i] = 0;
        errorVector[i] = 0;
    }
    
    // Gradient descent from scratch: 3 steps per iteration
    for(int iter = 0; iter < iterations; iter++)
    {
        /* Step 1:
        Multiply the data by the weights to get the log likelihood,
        then run these values through sigmoid() to get a vector of probabilities.
        
        Columns of first matrix should be equal to row of second matrix:
        columns(dataWeights) = 2 and rows(weights) = 2.

        If dim(matrix1) = m × n and dim(matrix2) = n × p, then dim(matrix1 %*% matrix2) = m × p:
        dim(dataMatrix) = train x 2 and dim(weights) = 2 x 1, so dim(dataMatrix %*% weights) = train x 1.
        */

        // Loop to do sigmoid(dataMatrix %*% weights) = probVector
        for(int i = 0; i < train; i++)
        {
            probVector[i] = sigmoid((dataMatrix[i][0] * weights[0]) + (dataMatrix[i][1] * weights[1]));
            // This matrix multiplication doesn't require a more complicated loop
            // because weights[] is a simple 1D vector, easy to hard-code the multiplication for.
        }
        
        /* Step 2:
        Compute the error: the labels of the target (which are 0 or 1) minus the probabilities. */

        for(int i = 0; i < train; i++)
        {
            errorVector[i] = labels[i] - probVector[i];
        }
        
        /* Step 3:
        Update the [weights] by [weights + (learning rate)(gradient)].
        
        Gradient is given as transpose(dataMatrix) %*% errorVector:
        dim(dataMatrix) = train x 2, so dim(transpose_dataMatrix) = 2 x train.

        Columns of first matrix should be equal to row of second matrix:
        columns(transpose_dataMatrix) = train and rows(errorVector) = train.

        If dim(matrix1) = m × n and dim(matrix2) = n × p, then dim(matrix1 %*% matrix2) = m × p:
        dim(transpose_dataMatrix) = 2 x train and dim(errorVector) = train x 1,
        so dim(transpose_dataMatrix %*% errorVector) = 2 x 1 = dim(weights).
        */

        // Loop to do (transpose_dataMatrix %*% errorVector) = gradient
        for(int i = 0; i < train; i++)
        {
            gradient[0] += transpose_dataMatrix[0][i] * errorVector[i];
            gradient[1] += transpose_dataMatrix[1][i] * errorVector[i];
        }
        // Update weights
        weights[0] += learningRate * gradient[0];
        weights[1] += learningRate * gradient[1];
        // Reset gradients for next iteration
        gradient[0] = 0.0;
        gradient[1] = 0.0;
    } // End of iterations

    
    // Print weights
    cout << "First coefficient w0 is " << weights[0] << endl;
    cout << "Second coefficient w1 is " << weights[1] << endl;
    // According to R, the weights should be about w0 = 0.9999 and w1 = -2.4109 for titanic_project.csv
    // In this program the output should be w0 = 0.999877 and w1 = -2.41086
    

    // Convert the probabilities to predictions
    int predVector[train];
    for(int i = 0; i < train; i++)
    {
        if(probVector[i] < 0.5)
        {
            predVector[i] = 0;
        }
        else  // If probVector[i] = 0.5 or more
        {
            predVector[i] = 1;
        }
    }


    /*
    // Print some of the probabilities + predictions + actual values for debugging
    cout << "The first 10 probabilities, predictions, and actual values are:" << endl;
    for(int i = 0; i < 10; i++)
    {
        cout << probVector[i] << " ";
        cout << predVector[i] << " ";
        cout << survived.at(i) << " ";
        cout << endl;
        // Note: in the internal coding of sex in titanic_project.csv,
        // male=1 and female=0 & survived=1 and died=0
        // (determined by comparing to the csv file on GitHub)
    }
    */
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time for training data was " << duration.count() << " microseconds." << endl;





    // Next step: predict on test data
    double weights_TEST[] = {1.0, 1.0};
    int dataMatrix_TEST[test][2];
    int transpose_dataMatrix_TEST[2][test];

    for(int i = train; i < numObservations; i++)
    {
        dataMatrix_TEST[i][0] = 1;
        dataMatrix_TEST[i][1] = sex.at(i);
        transpose_dataMatrix_TEST[0][i] = 1;
        transpose_dataMatrix_TEST[1][i] = sex.at(i);
    }

    int labels_TEST[test];
    for(int i = test; i < numObservations; i++)
    {
        labels_TEST[i] = survived.at(i);
    }

    //int iterations = 500000;
    //double learningRate = 0.001;
    //double gradient[] = {0.0, 0.0};
    double probVector_TEST[test];
    double errorVector_TEST[test];
    for(int i = 0; i < test; i++)
    {
        probVector_TEST[i] = 0;
        errorVector_TEST[i] = 0;
    }

    for(int iter = 0; iter < iterations; iter++)
    {
        for(int i = 0; i < test; i++)
        {
            probVector_TEST[i] = sigmoid((dataMatrix_TEST[i][0] * weights_TEST[0]) + (dataMatrix_TEST[i][1] * weights_TEST[1]));
        }
        
        for(int i = 0; i < test; i++)
        {
            errorVector_TEST[i] = labels_TEST[i] - probVector_TEST[i];
        }

        for(int i = 0; i < test; i++)
        {
            gradient[0] += transpose_dataMatrix_TEST[0][i] * errorVector_TEST[i];
            gradient[1] += transpose_dataMatrix_TEST[1][i] * errorVector_TEST[i];
        }
        weights_TEST[0] += learningRate * gradient[0];
        weights_TEST[1] += learningRate * gradient[1];
        gradient[0] = 0.0;
        gradient[1] = 0.0;
    } // End of iterations

    // Convert the probabilities to predictions
    int predVector_TEST[test];
    for(int i = 0; i < test; i++)
    {
        if(probVector_TEST[i] < 0.5)
        {
            predVector_TEST[i] = 0;
        }
        else
        {
            predVector_TEST[i] = 1;
        }
    }

    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    // Calculate accuracy, sensitivity, & specificity
    for(int i = 0; i < test; i++)
    {
        if(survived.at(train+i) == 0 && predVector_TEST[i] == 0)
        {
            tn++;
        }
        else if(survived.at(train+i) == 1 && predVector_TEST[i] == 1)
        {
            tp++;
        }
        else if(survived.at(train+i) == 0 && predVector_TEST[i] == 1)
        {
            fp++;
        }
        else if(survived.at(train+i) == 1 && predVector_TEST[i] == 0)
        {
            fn++;
        }
    }

    double accuracy = (tp+tn)/(tp+tn+fp+fn);
    double sens = tp/(tp+fn);
    double spec = tn/(tn+fp);
    cout << "Accuracy is " << accuracy << endl;
    cout << "Sensitivity is " << sens << endl;
    cout << "Sensitivity is " << spec << endl;

    
    
    return 0;
}   // End of main()