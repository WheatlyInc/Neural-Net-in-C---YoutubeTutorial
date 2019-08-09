/**************************************
https://www.youtube.com/watch?v=KkwX7FkLfug&list=LLR-Oo87u6NOclDwiwdSQsoQ&index=6&t=1s
Neural Net in C++ Tutorial on Vimeo

Before implementing TrainingData Class

David Miller, http://millermattson.com/dave
vinh nguyen
**************************************/
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert> //For assert function
#include <cmath> //For tanh functions
#include <fstream>
#include <sstream>

using namespace std;

//To complete our previous version, we will implement a TrainingData class
// Silly class to read TraningData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a field specified by a command line argument, etc.

class TrainingData
{
public:
    TrainingData(const string fielname);
    bool isEof(void) {return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0)
        abort();

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
            inputVals.push_back(oneValue);
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if ( label.compare("out:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }

    return targetOutputVals.size();
}

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ********************************  class Neuron ********************************
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val){ m_outputVal = val;}
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;      // [0.0..1.0] overall net training rate
    static double alpha;    // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) {return rand() / double(RAND_MAX);  }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;  // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 9; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate (eta):
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight
                + alpha             // -> Momentum
                * oldDeltaWeight;
    /*** eta -> overall net learning rate
        0.0 - slow learner
        0.2 - medium learner
        1.0 - reckless learner
    **** alpha -> momentum
        0.0 - no momentum
        0.5 - moderate momentum *******/

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }

}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0..1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}
// ********************************  class Neuron ********************************


class Net // Higher Level Class
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const ;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers{layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over


void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)       //Loop through all Neurons in output layer
        resultVals.push_back(m_layers.back()[n].getOutputVal());
}

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement (Not needed, but will print how well the neural net has been doing):

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
            hiddenLayer[n].calcHiddenGradients(nextLayer);
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n)
            layer[n].updateInputWeights(prevLayer);
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    // For prototyping/assuming you know what is true
    assert(inputVals.size() == m_layers[0].size() - 1);   // size - 1 excludes the bias neuron

    // Assign (Latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate (Tell each neuron to feed forward)
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1]; // Create a pointer to efficiently pass into feedForward right below
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer()); // Append on container m_layers

        //If the layer number in this loop is the output layer, then the number of outputs is 0.
        //Otherwise the number of outputs is whatever is in that element topology for the next layer over.
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        //We have made a new Layer, now fill it with neurons, AND
        //add a bias neuron to the layer:
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) //NOTICE: We are doing <= since we are also adding an additional neuron: a bias neuron
        {
            // Make a new neuron object and
            // append that to the layer we just got done creating
            m_layers.back().push_back(Neuron(numOutputs, neuronNum)); // Back() gives last element in container
            cout << "Made a Neuron!" << endl;
        }
        // Force the bias node's node's output to 1.0 (it was the last neuron pushed in this layer) :
        m_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
        cout << v[i] << " ";

    cout << endl;
}

int main()
{

    TrainingData trainData("trainingData.txt");

    // e.e., {3, 2, 1}
    vector<unsigned>topology;
    trainData.getTopology(topology);

    // Pass a structure into the constructor;
    // this line should be all that is required to construct a neural net.
    Net myNet(topology);


    // To train it we need a member function that feeds forward a bunch of input values
    vector<double> inputVals;

    // During training, after feeding forward a bunch of inputs, we need to tell the neural net what the outputs were supposed to have been
    // so that it can go back propogatation / learning.
    vector<double> targetVals;

    // After training, we are also interested in the net's results
    vector<double> resultVals;

    int trainingPass = 0;

    while(!trainData.isEof())
    {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0])
            break;

        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent ssamples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
    return 0;
}
