package mnist;

import ai.djl.*;
import ai.djl.basicdataset.cv.ImageDataset;
import ai.djl.basicdataset.cv.classification.AbstractImageFolder;
import ai.djl.basicdataset.cv.classification.ImageClassificationDataset;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.repository.Repository;
import ai.djl.training.*;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.BatchSampler;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.util.ProgressBar;

import ai.djl.translate.TranslateException;
import ai.djl.util.cuda.CudaUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Logger;
import java.util.stream.Collectors;

@SpringBootApplication
public class Experiment1Application {

    static final Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
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

        for(int i = 0; i < numberOfModels; i++){

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


    public static TrainingConfig createTrainingConfig(int trainingSetSize){

        Tracker annealer = new Annealer(batchSize,trainingSetSize,1E-3f,0.7f);
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
        TrainingConfig trainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(adam)
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        return trainingConfig;
    }

    // 1. Download data from: https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download
    // 2. in assets Folder legen nur trainingSet // Achtung doppelt verschachtelt: trainingSet/trainingSet
    public static Dataset[] createMnistCustomSimple() throws IOException, TranslateException {

        Repository repository = Repository.newInstance("trainingSet", Paths.get("./src/main/resources/data/trainingSet/"));
        System.out.println("Data path: " + Paths.get("./src/main/resources/data/trainingSet/"));

        ImageFolder dataset = ImageFolder.builder()
                .setRepository(repository)
                .addTransform(new Resize(28, 28))
                .addTransform(new ToTensor())
                .setSampling(batchSize, true)
                .optFlag(Image.Flag.GRAYSCALE)
                .build();

        //Image.Flag.GRAYSCALE
        dataset.prepare(new ProgressBar());

        System.out.println("Loaded Dataset size "+ dataset.size());
        System.out.println("Dataset Classes: "+ dataset.getClasses());

        return dataset.randomSplit(10,1);
    }





    public static void main(String[] args) {
        SpringApplication.run(Experiment1Application.class, args);

        logger.info("Start");

        ArrayList<Model> models = createModels();

        for(int i = 0; i < models.size(); i++) {
            System.out.println("\n \n Training Model " + models.get(i).getName() + "\n");

            try {
                Dataset[] datasets = createMnistCustomSimple();
                Dataset trainingDataset = datasets[0];
                Dataset validationDataset = datasets[1];

                TrainingConfig trainingConfig = createTrainingConfig((int)(42000 * trainingDatasetPercentage));
                Trainer trainer = models.get(i).newTrainer(trainingConfig);
                trainer.initialize(new Shape(batchSize, 1, 28, 28));

                System.out.println("Device used for Training: " + trainer.getDevices()[0].toString() );

                System.out.println("GPU count: " + Engine.getInstance().getGpuCount());

                EasyTrain.fit(trainer, epochs, trainingDataset, validationDataset);
                TrainingResult result = trainer.getTrainingResult();
                trainer.close();
                System.out.println(result);

            }catch (Exception e) {
                System.out.println(e.getMessage());
                System.exit(-1);
            }
        }

        logger.info("End");


    }




    // Get Training and validiation Dataset
    // EXAMPLE WITH Ready to go mnist Dataset provided by DJL
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




}