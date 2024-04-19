package org.example.djl.training.linearregression;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LinearRegression {

    public static void main(String[] args) {

        try {
//            manualImplementation();
            conciseDjlImplementation();
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static void conciseDjlImplementation() throws TranslateException, IOException {

        //generating the dataset
        NDManager manager = NDManager.newBaseManager();
        NDArray trueW = manager.create(new float[]{2, -3.4f});
        float trueB = 4.2f;

        DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
        NDArray features = dp.getX();
        NDArray labels = dp.getY();

//        //reading the dataset
        int batchSize = 10;
        ArrayDataset dataset = Utils.loadArray(features, labels, batchSize, false);
//
//        Batch batch = dataset.getData(manager).iterator().next();
//        NDArray X = batch.getData().head();
//        NDArray y = batch.getLabels().head();
//        System.out.println(X);
//        System.out.println(y);
//        batch.close();

//        defining the model
        Model model = Model.newInstance("lin-reg");
        SequentialBlock net = new SequentialBlock();
        Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
        net.add(linearBlock);
        model.setBlock(net);

        //Defining the loss function
        //L2 Loss or ‘Mean Squared Error’ is the sum of the squared difference between the true y value and the predicted y value.
        Loss l2loss = Loss.l2Loss();

        //defining the optimization algorithm
        Tracker lrt = Tracker.fixed(0.03f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        //instantiate configuration and trainer
        DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(manager.getEngine().getDevices(1)) // single GPU
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        //initialize model parameters
        // First axis is batch size - won't impact parameter initialization
        // Second axis is the input size
        trainer.initialize(new Shape(batchSize, 2));


        //metrics
        Metrics metrics = new Metrics();
        trainer.setMetrics(metrics);

        //training

        int numEpochs = 3;

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.printf("Epoch %d\n", epoch);
            // Iterate over dataset
            for (Batch batch : trainer.iterateDataset(dataset)) {
                // Update loss and evaulator
                EasyTrain.trainBatch(trainer, batch);

                // Update parameters
                trainer.step();

                batch.close();
            }
            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }

        Block layer = model.getBlock();
        ParameterList params = layer.getParameters();
        NDArray wParam = params.valueAt(0).getArray();
        NDArray bParam = params.valueAt(1).getArray();

        float[] w = trueW.sub(wParam.reshape(trueW.getShape())).toFloatArray();
        System.out.printf("Error in estimating w: [%f %f]\n", w[0], w[1]);
        System.out.printf("Error in estimating b: %f\n", trueB - bParam.getFloat());

        //saving the model
        Path modelDir = Paths.get("../models/lin-reg");
        Files.createDirectories(modelDir);

        model.setProperty("Epoch", Integer.toString(numEpochs)); // save epochs trained as metadata

        model.save(modelDir, "lin-reg");

        System.out.println(model);


    }

    class Utils{
        // Saved in the utils file for later use
        public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
            return new ArrayDataset.Builder()
                    .setData(features) // set the features
                    .optLabels(labels) // set the labels
                    .setSampling(batchSize, shuffle) // set the batch size and random sampling
                    .build();
        }
    }

    static void manualImplementation() throws TranslateException, IOException {

        //generating the dataset
        NDManager manager = NDManager.newBaseManager();
        NDArray trueW = manager.create(new float[]{2, -3.4f});
        float trueB = 4.2f;

        DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
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

    // Generate y = X w + b + noise
    public static  DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.dot(w).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
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


