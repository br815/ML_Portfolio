/*
Name: Bushra Rahman
Class: CS 4375.004
Assignment: ML Algorithms from Scratch - Naive Bayes
Due date: 3/4/23
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <regex>
#include <bits/stdc++.h>
#include <math.h>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono; // used to measure time
string fileName = "titanic_project.csv";

// Function to calculate age likelihood
double ageLikelihood_calc(double x, double mean, double var)
{
    double ageLikelihood = (1 / sqrt(2*M_PI*var)) * exp(-1*pow(x-mean,2)/(2 * var));
    return ageLikelihood;
}

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


    // Next step: NB on target=survived, predictors=pclass+sex+age, train=first 800 observations
    int train = 800;  // Size of training dataset, can be adjusted to different sizes
    int test = numObservations - train;

    // Get factor counts for target and all factor predictors
    int survivedCount_0 = 0;
    int survivedCount_1 = 0;

    int pclassCount_1 = 0;
    int pclassCount_2 = 0;
    int pclassCount_3 = 0;

    int sexCount_0 = 0;
    int sexCount_1 = 0;

    for(int i = 0; i < train; i++)
    {
        // Get survived count for survived=1
        survivedCount_1 += survived.at(i);

        // Get pclass counts
        if(pclass.at(i) == 1)
        {
            pclassCount_1 += 1;
        }
        else if(pclass.at(i) == 2)
        {
            pclassCount_2 += 1;
        }
        else
        {
            pclassCount_3 += 1;
        }

        // Get sex count for sex=1
        sexCount_1 += sex.at(i);
    }
    survivedCount_0 = train - survivedCount_1;
    sexCount_0 = train - sexCount_1;
    // cout << "Survived = " << survivedCount_1 << ", Perished = " << survivedCount_0 << endl;
    // cout << "pclass 1 = " << pclassCount_1 << ", pclass 2 = " << pclassCount_2 << ", pclass 3 = " << pclassCount_3 << endl;
    // cout << "sex 0 = " << sexCount_0 << ", sex 1 = " << sexCount_1 << endl;


    // Prior (apriori) probs = counts of survived or died / total number of observations
    double aprioriProbs[] = {0.0, 0.0};
    aprioriProbs[0] = (double) survivedCount_0 / train;
    aprioriProbs[1] = (double) survivedCount_1 / train;
    cout << "Apriori probabilities are " << aprioriProbs[0] << " and " << aprioriProbs[1] << endl;


    /* Conditional probs = likelihoods = P(X|Y) = P(data|target)
    Likelihood is the probability that an event which already happened would have yielded a specific outcome.
    Likelihoods do not necessitate that all possible values sum to 1.
    Formulas (for each factor level i in a class):
    likelihood (class=i|survived=yes) = count(factor = i and survived=yes) / count(survived=yes)
    likelihood (class=i|survived=no) = count(factor = i and survived=no) / count(survived=no)
    */

    // Likelihoods for factor predictors: p(survived|pclass), p(survived|sex)
    // 2 rows for survived=0,1 and 3 cols for pclass=1,2,3
    double pclassLikelihoods[2][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0}};
     // 2 rows for survived=0,1 and 2 cols for sex=0,1 
    double sexLikelihoods[2][2] = {{0.0,0.0},{0.0,0.0}};
    for(int i = 0; i < train; i++)
    {
        // pclassLikelihoods[0][0] = survived=0, pclass=1
        if(survived.at(i) == 0 && pclass.at(i) == 1)
            pclassLikelihoods[0][0]++;
        // pclassLikelihoods[0][1] = survived=0, pclass=2
        else if(survived.at(i) == 0 && pclass.at(i) == 2)
            pclassLikelihoods[0][1]++;
        // pclassLikelihoods[0][2] = survived=0, pclass=3
        else if(survived.at(i) == 0 && pclass.at(i) == 3)
            pclassLikelihoods[0][2]++;
        // pclassLikelihoods[1][0] = survived=1, pclass=1
        else if(survived.at(i) == 1 && pclass.at(i) == 1)
            pclassLikelihoods[1][0]++;
        // pclassLikelihoods[1][1] = survived=1, pclass=2
        else if(survived.at(i) == 1 && pclass.at(i) == 2)
            pclassLikelihoods[1][1]++;
        // pclassLikelihoods[1][2] = survived=1, pclass=3
        else if(survived.at(i) == 1 && pclass.at(i) == 3)
            pclassLikelihoods[1][2]++;
        
        // sexLikelihoods[0][0] = survived=0, sex=0
        if(survived.at(i) == 0 && sex.at(i) == 0)
            sexLikelihoods[0][0]++;
        // sexLikelihoods[0][1] = survived=0, sex=1
        else if(survived.at(i) == 0 && sex.at(i) == 1)
            sexLikelihoods[0][1]++;
        // sexLikelihoods[1][0] = survived=1, sex=0
        else if(survived.at(i) == 1 && sex.at(i) == 0)
            sexLikelihoods[1][0]++;
        // sexLikelihoods[1][1] = survived=1, sex=1
        else if(survived.at(i) == 1 && sex.at(i) == 1)
            sexLikelihoods[1][1]++;
    }
    // complete calculation of likelihood
    for(int i = 0; i < 3; i++)
    {
        pclassLikelihoods[0][i] /= survivedCount_0;
        pclassLikelihoods[1][i] /= survivedCount_1;
    }
    for(int i = 0; i < 2; i++)
    {
        sexLikelihoods[0][i] /= survivedCount_0;
        sexLikelihoods[1][i] /= survivedCount_1;
    }

    // print cond probs for pclass
    cout << "\nConditional probabilities for pclass are:" << endl;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            cout << pclassLikelihoods[i][j] << " ";
        }
        cout << endl;
    }
    // print cond probs for sex
    cout << "\nConditional probabilities for sex are:" << endl;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            cout << sexLikelihoods[i][j] << " ";
        }
        cout << endl;
    }

    // Likelihood for continuous predictor age: p(survived|age)
    // First step: calculate mean and variance of age for survived=0 or 1
    double ageMean[] = {0.0,0.0};   // 0th element for survived=0, 1st elem for survived=1
    double ageVar[] = {0.0,0.0};
    for(int i = 0; i < train; i++)
    {
        if(survived.at(i) == 0)
        {
            ageMean[0] += age.at(i);
        }
        else
        {
            ageMean[1] += age.at(i);
        }
    }
    // complete calculation of mean
    ageMean[0] /= train;
    ageMean[1] /= train;

    // formula for variance: numerator = sum((x-mean)^2), denominator = train
    for(int i = 0; i < train; i++)
    {
        if(survived.at(i) == 0)
        {
            ageVar[0] += pow((age.at(i)-ageMean[0]),2);
        }
        else
        {
            ageVar[1] += pow((age.at(i)-ageMean[1]),2);
        }
    }
    // complete calculation of variance
    ageVar[0] /= train;
    ageVar[1] /= train;

    cout << "\nMeans of age for not survived and survived are:" << endl;
    cout << ageMean[0] << " " << ageMean[1] << endl;
    cout << "\nVariances of age for not survived and survived are:" << endl;
    cout << ageVar[0] << " " << ageVar[1] << endl;

    


    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time was " << duration.count() << " microseconds." << endl;
    return 0;
}   // End of main()