/*
Name: Bushra Rahman
Class: CS 4375.004
Assignment: Data Exploration in C++
Due date: 2/4/23
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <bits/stdc++.h> //used for vector sort() function
#include <cmath>

using namespace std;
string fileName = "Boston.csv";


// Function to find the sum of a numeric vector
double vectorSum(vector<double> column)
{
    double sum = 0.0;
    for(int i = 0; i < column.size(); i++)
    {
        sum += column[i];
    }
    return sum;
}   // End function vectorSum()



// Function to find the mean of a numeric vector
double vectorMean(vector<double> column)
{
    double mean = vectorSum(column) / column.size();
    return mean;
}   // End function vectorMean()



// Function to find the median of a numeric vector
double vectorMedian(vector<double> column)
{
    double median = 0.0;
    int middle = column.size() / 2;

    if(column.size()%2 == 0)    // If # of elements is even
    {
        median = (column[middle] + column[middle - 1])/2;
        // Median is the avg of the 2 middle elements
    }
    else                        // If # of elements is odd
    {
        median = column[middle];
        // Median is the middle element
    }
    return median;
}   // End function vectorMedian()



// Function to find the range of a numeric vector
double vectorRange(vector<double> column)
{
    double range = 0.0;
    double min = column[0];
    double max = column[column.size()-1];
    range = max - min;
    return range;
}   // End function vectorRange()



// Function to compute covariance between 2 numeric vectors
double covar(vector<double> col1, vector<double> col2)
{
    double covariance = 0.0;
    double sum = 0.0;
    int n = col1.size();
    double mean1 = vectorMean(col1);
    double mean2 = vectorMean(col2);

    for(int i = 0; i < n; i++)
    {
        sum += (col1[i]-mean1) * (col2[i]-mean2);
    }
    covariance = sum / (n-1);

    return covariance;
}   // End function covar()



// Function to compute correlation between 2 numeric vectors
double cor(vector<double> col1, vector<double> col2)
{
    double correlation = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    int n = col1.size();
    double mean1 = vectorMean(col1);
    double mean2 = vectorMean(col2);

    double SD1 = 0.0;
    double SD2 = 0.0;

    // Formula for standard deviation SD = sqrt(variance)
    for(int i = 0; i < n; i++)
    {
        sum1 += (col1[i]-mean1) * (col1[i]-mean1);
        sum2 += (col2[i]-mean2) * (col2[i]-mean2);
    }
    SD1 = sqrt(sum1/(n-1));
    SD2 = sqrt(sum2/(n-1));

    correlation = covar(col1,col2) / (SD1 * SD2);

    return correlation;
}// End function cor()



// Function to print statistical results
void printStats(vector<double> column)
{   
    cout << "Sum: " << vectorSum(column) << endl;

    cout << "Mean: " << vectorMean(column) << endl;

    cout << "Median: " << vectorMedian(column) << endl;

    cout << "Range: " << vectorRange(column) << endl;

    return;
}   // End function printStats



// main()
int main(int argc, char** argv)
{
    // Steps to read in csv file
    ifstream inFS;                 // Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // Try to open file
    cout << "Opening file " + fileName << endl;

    inFS.open(fileName);
    if(!inFS.is_open())
    {
        cout << "ERROR: Could not open file " + fileName << endl;
        return 1;   // 1 indicates error
    }

    // Can now use inFS stream like cin stream
    // Boston.csv should contain 2 doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // Echo heading
    cout << "heading: " << line << endl;
    // Heading is first row of Boston.csv: rm,medv

    int numObservations = 0;
    while (inFS.good())
    {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);   // stof() converts string to float
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;

    cout << "Closing file " + fileName << endl;
    inFS.close();   // Done with file, so close it

    // Sort vectors
    sort(rm.begin(), rm.end());
    sort(medv.begin(), medv.end());

    cout << "Number of records: " << numObservations << endl;
    // numObservations should be 506 for Boston.csv

    cout << "\nStats for rm" << endl;
    printStats(rm);

    cout << "\nStats for medv" << endl;
    printStats(medv);

    cout << "\nCovariance = " << covar(rm, medv) << endl;

    cout << "\nCorrelation = " << cor(rm, medv) << endl;

    cout << "\nProgram terminated." << endl;
    
    return 0;
}   // end function main()