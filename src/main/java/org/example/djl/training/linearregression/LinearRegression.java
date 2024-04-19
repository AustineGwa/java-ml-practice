package org.example.djl.training.linearregression;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class LinearRegression {

    public static void main(String[] args) {
        try {
            manualImplementation();
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static void conciseDjlImplementation(){

    }

    static void manualImplementation() throws TranslateException, IOException {

        //generating the dataset
        NDManager manager = NDManager.newBaseManager();
        NDArray trueW = manager.create(new float[]{2, -3.4f});
        float trueB = 4.2f;

        DataPoints dp = syntheticData(manager, trueW, trueB, 1000);
        NDArray features = dp.getX();
        NDArray labels = dp.getY();

        System.out.printf("features: [%f, %f]\n", features.get(0).getFloat(0), features.get(0).getFloat(1));
        System.out.println("label: " + labels.getFloat(0));

//        // plot the dataset
//        float[] X = features.get(new NDIndex(":, 1")).toFloatArray();
//        float[] y = labels.toFloatArray();
//
//
//        Table data = Table.create("Data")
//                .addColumns(
//                        FloatColumn.create("X", X),
//                        FloatColumn.create("y", y)
//                );
//
//        ScatterPlot.create("Synthetic Data", data, "X", "y");


        //reading the dataset
        int batchSize = 10;
        ArrayDataset dataset = new ArrayDataset.Builder()
                .setData(features) // Set the Features
                .optLabels(labels) // Set the Labels
                .setSampling(batchSize, false) // set the batch size and random sampling to false
                .build();

//        Batch batch = dataset.getData(manager).iterator().next();
//        // Call head() to get the first NDArray
//        NDArray X = batch.getData().head();
//        NDArray y = batch.getLabels().head();
//        System.out.println(X);
//        System.out.println(y);
//        // Don't forget to close the batch!
//        batch.close();

        //initializing model parameters
        NDArray w = manager.randomNormal(0, 0.01f, new Shape(2, 1), DataType.FLOAT32);
        NDArray b = manager.zeros(new Shape(1));
        NDList params = new NDList(w, b);

        //training
        float lr = 0.03f;  // Learning Rate
        int numEpochs = 3;  // Number of Iterations

        // Attach Gradients
        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }

        Training training = new Training();
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Assuming the number of examples can be divided by the batch size, all
            // the examples in the training dataset are used once in one epoch
            // iteration. The features and tags of minibatch examples are given by X
            // and y respectively.
            for (Batch batch : dataset.getData(manager)) {
                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    // Minibatch loss in X and y
                    NDArray l = training.squaredLoss(training.linreg(X, params.get(0), params.get(1)), y);
                    gc.backward(l);  // Compute gradient on l with respect to w and b
                }
                training.sgd(params, lr, batchSize);  // Update parameters using their gradient

                batch.close();
            }
            NDArray trainL = training.squaredLoss(training.linreg(features, params.get(0), params.get(1)), labels);
            System.out.printf("epoch %d, loss %f\n", epoch + 1, trainL.mean().getFloat());
        }


        //validation of training
        float[] w_ = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
        System.out.println(String.format("Error in estimating w: [%f, %f]", w_[0], w_[1]));
        System.out.println(String.format("Error in estimating b: %f", trueB - params.get(1).getFloat()));


    }

    // Generate y = X w + b + noise
    public static  DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.dot(w).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }
}

class DataPoints {
    private NDArray X, y;
    public DataPoints(NDArray X, NDArray y) {
        this.X = X;
        this.y = y;
    }

    public NDArray getX() {
        return X;
    }

    public NDArray getY() {
        return y;
    }
}

class Training{
    //defining the model
    // Saved in Training.java for later use
    public NDArray linreg(NDArray X, NDArray w, NDArray b) {
        return X.dot(w).add(b);
    }

    //defining the loss function
    // Saved in Training.java for later use
    public NDArray squaredLoss(NDArray yHat, NDArray y) {
        return (yHat.sub(y.reshape(yHat.getShape()))).mul
                ((yHat.sub(y.reshape(yHat.getShape())))).div(2);
    }

    //defining the optimization algorithm
    // Saved in Training.java for later use
    public void sgd(NDList params, float lr, int batchSize) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }
}


