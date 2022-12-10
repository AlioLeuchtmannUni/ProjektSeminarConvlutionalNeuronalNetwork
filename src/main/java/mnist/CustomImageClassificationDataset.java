package mnist;


import ai.djl.basicdataset.cv.classification.ImageClassificationDataset;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.repository.Repository;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;

/*
Creates Image Classification Dataset.

requires Structured Image Data.

Folder Structure   folderName/label1/imagesOfClassLabel1 , folderName/label2/imagesOfClassLabel2

and automatically creates Labeled Data
* */
public class CustomImageClassificationDataset extends ImageClassificationDataset {

    ImageFolder dataset; // https://javadoc.io/doc/ai.djl/basicdataset/latest/ai/djl/basicdataset/cv/classification/ImageFolder.html
    private final int imageWidth;
    private final int imageHeight;
    private final int batchSize;

    public CustomImageClassificationDataset(
            int batchSize,
            int imageWidth,
            int imageHeight,
            String folderName,
            String path,
            boolean random) {

        super(new BaseBuilder<BaseBuilder>() {
            @Override
            protected BaseBuilder self() {return this;}
        });

        this.batchSize = batchSize;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;

        Repository repository = Repository.newInstance(folderName, Paths.get(path));
        dataset = ImageFolder.builder()
                        .setRepository(repository)
                        .addTransform(new Resize(imageWidth, imageHeight))
                        .addTransform(new ToTensor())
                        .setSampling(batchSize, random)
                        .build();

    }


    public CustomImageClassificationDataset(
            BaseBuilder<?> builder,
            int batchSize,
            int imageWidth,
            int imageHeight,
            String folderName,
            String path,
            boolean random) {

        super(new BaseBuilder<BaseBuilder>() {
            @Override
            protected BaseBuilder self() {
                return this;
            }
        });

        this.batchSize = batchSize;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;

        Repository repository = Repository.newInstance(folderName, Paths.get(path));
        dataset = ImageFolder.builder()
                .setRepository(repository)
                .addTransform(new Resize(imageWidth, imageHeight))
                .addTransform(new ToTensor())
                .setSampling(batchSize, random)
                .build();

    }

    @Override
    protected long getClassNumber(long index) throws IOException {
        return 0;
    }

    @Override
    public List<String> getClasses() {
        return dataset.getClasses();
    }

    @Override
    protected Image getImage(long index) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public Optional<Integer> getImageWidth() {
        return Optional.of(imageWidth);
    }

    @Override
    public Optional<Integer> getImageHeight() {
        return Optional.of(imageHeight);
    }

    @Override
    protected long availableSize() {
        return 0;
    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {
        dataset.prepare(progress);
    }
}
