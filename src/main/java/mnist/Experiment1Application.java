package mnist;

import ai.djl.*;
import ai.djl.basicdataset.cv.ImageDataset;
import ai.djl.basicdataset.cv.classification.ImageClassificationDataset;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.*;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.util.ProgressBar;

import ai.djl.translate.TranslateException;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;
import java.util.ArrayList;

@SpringBootApplication
public class Experiment1Application {

    public static final int numberOfModels = 3;
    public static final int epochs = 20;
    public static final int batchSize = 64;
    public static final int numberOfOutputNeurons = 10;
    public static final float trainingDatasetPercentage = 0.9f;

    // https://d2l.djl.ai/chapter_linear-networks/softmax-regression-scratch.html
    static public NDArray softmax(NDArray X) {
        NDArray Xexp = X.exp(); // Returns the exponential value of this NDArray element-wise.
        NDArray partition = Xexp.sum(new int[]{1}, true);
        return Xexp.div(partition); // The broadcast mechanism is applied here
    }
    static public NDList softmax(NDList list){
        for(int i=0; i<list.size();i++){
            list.set(i, softmax(list.get(i)));
        }
        return list;
    }

    static public Block createConv2dLayer(int kernelSize, int features, int stride, int padding) {
        return Conv2d.builder()
                .setKernelShape(new Shape(kernelSize,kernelSize))
                .setFilters(features)
                .optStride(new Shape(stride,stride))
                .optPadding(new Shape(padding,padding))
                .build();
    }

    static public Block maxPoolValid() {
        return Pool.maxPool2dBlock(new Shape(2,2));
    }

    static public Block maxPoolSame() {
        // kernel Size, stride, padding
        return Pool.maxPool2dBlock(new Shape(2,2), new Shape(1,1), new Shape(1,1));
    }

    static public ArrayList<Model> createModels() {

        ArrayList<Model> models = new ArrayList<>();

        // TODO: replace
        for(int i=1; i < numberOfModels;i++){

            Model model = Model.newInstance("Model("+i+")");
            SequentialBlock sequentialBlock = new SequentialBlock();

            boolean isSecondOrThirdModel = i > 0;
            boolean isThirdModel = i > 1;

            sequentialBlock
                    .add(createConv2dLayer(5,24,1,1))
                    .add(Activation.reluBlock())
                    .add(maxPoolValid());

            if(isSecondOrThirdModel){
                sequentialBlock
                        .add(createConv2dLayer(5,48,1,1))
                        .add(Activation.reluBlock())
                        .add(maxPoolValid());
            }

            if(isThirdModel){
                sequentialBlock
                        .add(createConv2dLayer(5,64,1,1))
                        .add(Activation.reluBlock())
                        .add(maxPoolSame());
            }

            // Output Layer
            sequentialBlock
                    // Flatten
                    .add(Blocks.batchFlattenBlock())
                    // Dense Layer 256 RELU
                    .add(Linear.builder().setUnits(256).build())
                    .add(Activation.reluBlock())
                    // Dense Layer 10 softmax
                    .add(Linear.builder().setUnits(numberOfOutputNeurons).build())
                    .add(new LambdaBlock(Experiment1Application::softmax));

            model.setBlock(sequentialBlock);
            models.add(model);
        }

        return models;
    }


    public static TrainingConfig createTrainingConfig(){

        // 60.000 := mnist Dataset size
        Tracker annealer = new Annealer(batchSize,(int)(60000 * trainingDatasetPercentage),1E-3f,0.8f);
        Optimizer adam = Optimizer
                .adam()
                .optWeightDecays(0.001f) // in default adam von keras aus beispiel nicht verwendet
                .optClipGrad(0.001f) // https://github.com/fizyr/keras-retinanet/issues/942 // in default adam von keras aus beispiel nicht verwendet
                .optBeta1(0.9f)
                .optBeta2(0.999f)
                .optEpsilon(1e-7f)
                .optLearningRateTracker(annealer)
                .build();

        // softmaxCrossEntropyLoss anstelle von  categorical_crossentropy
        DefaultTrainingConfig trainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(adam)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        return trainingConfig;
    }


    // 1. Download data from: https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download
    // 2. in assets Folder legen (trainingSet,testSet) // Achtung doppelt verschachtelt: trainingSet/trainingSet
    public static Dataset[] createMnistDatasetCustom() throws TranslateException, IOException {
        Dataset[] datasets = new Dataset[2];

        // Alternativ: ein Dataset aus allen daten -> dann split wie unten für mnist // siehe createMnistCustomSimple

        CustomImageClassificationDataset trainingDataset = new CustomImageClassificationDataset(batchSize,28,28,"trainingSet","src/main/resources/assets/data",false);
        CustomImageClassificationDataset validationDataset = new CustomImageClassificationDataset(batchSize,28,28,"testSet","src/main/resources/assets/data",false);
        trainingDataset.prepare(new ProgressBar());
        validationDataset.prepare(new ProgressBar());

        return datasets;
    }


    // 1. Download data from: https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download
    // 2. in assets Folder legen nur trainingSet // Achtung doppelt verschachtelt: trainingSet/trainingSet
    public static Dataset[] createMnistCustomSimple() throws TranslateException, IOException {
        Dataset[] datasets = new Dataset[2];

        CustomImageClassificationDataset dataset = new CustomImageClassificationDataset(batchSize,28,28,"trainingSet","src/main/resources/assets/data",false);
        dataset.prepare(new ProgressBar());

        int divider = (int)(dataset.size() * trainingDatasetPercentage);
        Dataset trainingDataset = dataset.subDataset(0,divider);
        Dataset validationDataset = dataset.subDataset(divider,(int)(dataset.size()));
        datasets[0] = trainingDataset;
        datasets[1] = validationDataset;
        return datasets;
    }

    // Get Training and validiation Dataset
    public static Dataset[] splitMnist() throws TranslateException, IOException {

        Dataset[] datasets = new Dataset[2];

        Mnist mnist = Mnist.builder().setSampling(batchSize,false).build();
        mnist.prepare(new ProgressBar());

        int divider = (int)(mnist.size() * trainingDatasetPercentage);
        Dataset trainingDataset = mnist.subDataset(0,divider);
        Dataset validationDataset = mnist.subDataset(divider,(int)(mnist.size()));


        trainingDataset.prepare(new ProgressBar());
        validationDataset.prepare(new ProgressBar());

        datasets[0] = trainingDataset;
        datasets[1] = validationDataset;

        System.out.println(
                "Datesets: mnist: " + mnist.size() +
                        " training: " + divider +
                        " validationDatasetSize: " + (mnist.size() - divider)
        );

        return datasets;
    }

    public static void main(String[] args) {
        SpringApplication.run(Experiment1Application.class, args);

        ArrayList<Model> models = createModels();

        for(int i = 0; i < models.size(); i++) {
            System.out.println("\n \n Training Model " + models.get(i).getName() + "\n");

            TrainingConfig trainingConfig = createTrainingConfig();
            Trainer trainer = models.get(i).newTrainer(trainingConfig);
            trainer.initialize(new Shape(batchSize, 1, 28, 28));

            try {

                //Dataset[] datasets = splitMnist();
                Dataset[] datasets = createMnistCustomSimple();

                Dataset trainingDataset = datasets[0];
                Dataset validationDataset = datasets[1];

                EasyTrain.fit(trainer, epochs, trainingDataset, validationDataset);
                TrainingResult result = trainer.getTrainingResult();
                trainer.close();

                System.out.println(result);
            }catch (Exception e) {
                System.out.println(e.getMessage());
                System.exit(-1);
            }
        }


    }



}