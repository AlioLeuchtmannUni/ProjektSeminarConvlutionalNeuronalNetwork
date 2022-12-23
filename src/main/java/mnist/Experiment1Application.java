package mnist;

import ai.djl.*;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
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
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.cuda.CudaUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.logging.Logger;

@SpringBootApplication
public class Experiment1Application {

    static final Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    public static final int numberOfModels = 3;
    public static final int epochs = 20;
    public static final int batchSize = 64;
    public static final int numberOfOutputNeurons = 10;
    public static final int trainingDatasetRatio = 10;

    /**
     * // https://d2l.djl.ai/chapter_linear-networks/softmax-regression-scratch.html
     * Apply softmax on NDArray and return
     * @param X {@link NDArray} to apply softmax on
     * @return {@link NDArray}
     */
    static public NDArray softmax(NDArray X) {
        NDArray Xexp = X.exp(); // Returns the exponential value of this NDArray element-wise.
        NDArray partition = Xexp.sum(new int[]{1}, true);
        return Xexp.div(partition); // The broadcast mechanism is applied here
    }

    /**
     * applying {@link #softmax(NDArray) } to every {@link NDArray} contained in passed {@link NDList}
     * @param list {@link NDList}
     * @return {@link NDList}
     */
    static public NDList softmax(NDList list){
        for(int i = 0; i < list.size(); i++){
            list.set(i, softmax(list.get(i)));
        }
        return list;
    }


    /**
     * Creates conv2dLayer
     * @param kernelSize Size of Kernel -> Produces Shape(kernelSize,kernelSize)
     * @param features number of Filters to use
     * @param stride produces Shape(stride,stride) for stride
     * @param padding produces Shape(padding,padding) for padding
     * @return block Containing Conv2d Layer
     */
    static public Block createConv2dLayer(
            int kernelSize,
            int features,
            int stride,
            int padding)
    {
        return Conv2d.builder()
                .setKernelShape(new Shape(kernelSize,kernelSize))
                .setFilters(features)
                .optStride(new Shape(stride,stride))
                .optPadding(new Shape(padding,padding))
                .build();
    }

    /**
     * Creates maxPool2dBlock with valid padding
     * kernel: Shape(2,2)
     * @return {@link Block} created from {@link Pool}
     */
    static public Block maxPoolValid() {
        return Pool.maxPool2dBlock(new Shape(2,2));
    }

    /**
     * Creates maxPool2dBlock with same padding
     * kernel: Shape(2,2)
     * stride: Shape(1,1)
     * padding: Shape(1,1)
     * @return {@link Block} created from {@link Pool}
     */
    static public Block maxPoolSame() {
        // kernel Size, stride, padding
        return Pool.maxPool2dBlock(new Shape(2,2), new Shape(1,1), new Shape(1,1));
    }

    /** <pre>
     * Creates Models from https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
     * @return List of the 3 created Models {@link ArrayList<Model>}
     * </pre>
     * **/
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


    /**
     * Creates a {@link TrainingConfig} with adam {@link Optimizer} and {@link SoftmaxCrossEntropyLoss}
     * @param tracker {@link Tracker} LearningRateTracker to update LearningRate
     * @return {@link TrainingConfig}
     */
    public static TrainingConfig createTrainingConfig(Tracker tracker){

        Optimizer adam = Optimizer
                .adam()
                .optWeightDecays(0.01f) // 0.001f // in default adam von keras aus beispiel nicht verwendet, renne onnst aber in Loss NaN
                .optClipGrad(0.001f) // https://github.com/fizyr/keras-retinanet/issues/942 // in default adam von keras aus beispiel nicht verwendet
                .optBeta1(0.9f)
                .optBeta2(0.999f)
                .optEpsilon(1e-7f)
                .optLearningRateTracker(tracker)
                .build();

        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(adam)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }


    // Developer note to find data download of mnist
    // 1. Download data from: https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download
    // 2. in assets Folder legen nur trainingSet // Achtung doppelt verschachtelt: trainingSet/trainingSet

    /**<pre>
     * Requirements for mnist example:
     * 1. Download data from: https://www.kaggle.com/datasets/scolianni/mnistasjpg?resource=download
     * 2. Put in asset folder // Note archiv Structure of related mnist Dataset: trainingSet/trainingSet
     *
     * Can be used to create a Dataset for any kind od Image Classification
     * Images are seperated by Class in separate Folders, Foldernames are the class names
     *
     * Example Structure:
     * folderName: trainingDataset
     * path: path to Folder containing trainingDataset
     * trainingDatasets contains Folders with names: 0,1,2,3,4,5,6,7,8,9
     * Folders 0,1,2,3,4,5,6,7,8,9 contain images of that class
     *<pre>
     * @param folderName {@link String} folderName containing the Training Data
     * @param path {@link String} Path to Folder containing the Training Data
     * @param imageResize {@link Resize} Size Images should be Transformed to
     * @param batchSize {@link Integer} BatchSize
     * @param channelFlag {@link Image.Flag} flag to choose between Greyscale or Colored
     * @param random boolean set random Flag
     * @param trainingDatasetRatio {@link Integer} applies trainingDatasetRatio : 1 to Dataset split
     * @return TrainingSet at index 0 and ValidationSet at index 1
     * @throws IOException  {@link IOException}
     * @throws TranslateException  {@link TranslateException}
     */
    public static RandomAccessDataset[] createMnistCustomSimple(
            String folderName,
            String path,
            Resize imageResize,
            Integer batchSize,
            Image.Flag channelFlag,
            boolean random,
            Integer trainingDatasetRatio
    ) throws IOException, TranslateException {

        Repository repository = Repository.newInstance(folderName, Paths.get(path));
        logger.fine("Dataset Path: "+ repository.getBaseUri().getPath());

        ImageFolder dataset = ImageFolder.builder()
                .setRepository(repository)
                .addTransform(imageResize)
                .addTransform(new ToTensor())
                .setSampling(batchSize, random)
                .optFlag(channelFlag)
                .build();

        dataset.prepare(new ProgressBar());

        logger.info("Loaded Dataset size "+ dataset.size());
        logger.info("Dataset Classes: "+ dataset.getClasses());

        return dataset.randomSplit(trainingDatasetRatio,1);
    }


    public static void main(String[] args) {
        SpringApplication.run(Experiment1Application.class, args);

        logger.info("Start");

        int width = 28;
        int height = 28;
        ArrayList<Model> models = createModels();

        for (Model model : models) {
            logger.info("\n \n Training Model " + model.getName() + "\n");

            try {

                RandomAccessDataset[] datasets = createMnistCustomSimple(
                        "trainingSet",
                        "./data/trainingSet",
                        new Resize(width,height),
                        batchSize,
                        Image.Flag.GRAYSCALE,
                        true,
                        trainingDatasetRatio
                );
                RandomAccessDataset trainingDataset = datasets[0];
                RandomAccessDataset validationDataset = datasets[1];

                Tracker annealer = new Annealer(batchSize,(int) trainingDataset.size(),1E-3f,0.95f);
                TrainingConfig trainingConfig = createTrainingConfig(annealer);
                Trainer trainer = model.newTrainer(trainingConfig);
                trainer.initialize(new Shape(batchSize, 1, width, height));

                Device[] devices = Engine.getInstance().getDevices();
                logger.info(devices.length + " devices found on System "+ "containing " + Engine.getInstance().getGpuCount() + " Gpus");

                logger.info("Devices used for Training: " + trainer.getDevices().length);
                logger.info("Cuda utils found "+CudaUtils.getGpuCount()+ " Gpus");
                logger.info("Cuda detected: "+CudaUtils.hasCuda());

                EasyTrain.fit(trainer, epochs, trainingDataset, validationDataset);
                TrainingResult result = trainer.getTrainingResult();
                trainer.close();
                System.out.println(result);

            } catch (Exception e) {
                System.out.println(e.getMessage());
                System.exit(-1);
            }
        }

        logger.info("End");


    }


    /**
     * Create Mnist Training and Validation Dataset from {@link Mnist} provided by DJL
     * @return Array of {@link Dataset} TrainingDataset at index 0 and ValidationDataset at index 1
     * @throws TranslateException {@link TranslateException}
     * @throws IOException {@link IOException}
     */
    public static Dataset[] splitMnist() throws TranslateException, IOException {

        Dataset[] datasets = new Dataset[2];

        Mnist mnist = Mnist.builder().setSampling(batchSize,false).build();
        mnist.prepare(new ProgressBar());

        int divider = (int)(mnist.size() * (1f - (1.0f/(float) trainingDatasetRatio)));
        Dataset trainingDataset = mnist.subDataset(0,divider);
        Dataset validationDataset = mnist.subDataset(divider,(int)(mnist.size()));

        trainingDataset.prepare(new ProgressBar());
        validationDataset.prepare(new ProgressBar());

        datasets[0] = trainingDataset;
        datasets[1] = validationDataset;

        logger.info(
                "Datesets: mnist: " + mnist.size() +
                        " training: " + divider +
                        " validationDatasetSize: " + (mnist.size() - divider)
        );

        return datasets;
    }




}