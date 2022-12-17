package mnist;

import ai.djl.training.tracker.Tracker;

public class Annealer implements Tracker {

    private final int batchSize;
    private final int trainingDatasetSize;
    private final float baseLr;
    private float lr;
    private final float factor;
    private int epoch = 1;
    private int lastUpdateNum = 0;

    /**
     * <pre>
     * Annealer implementing {@link Tracker}
     * changing Learnrate every epoch with lr = baseLr * (float) Math.pow(factor,epoch);
     * </pre>
     * @param batchSize batchSize
     * @param trainingDatasetSize TrainingDatasetSize
     * @param baseLr BaseLearnrate to start with
     * @param factor factor which gets applied like this: lr = baseLr * (float) Math.pow(factor,epoch);
     */
    Annealer(
            int batchSize,
            int trainingDatasetSize,
            float baseLr,
            float factor
    ){
        this.batchSize = batchSize;
        this.trainingDatasetSize = trainingDatasetSize;
        this.baseLr = baseLr;
        this.lr = baseLr;
        this.factor = factor;
    }

    /**
     * Updating Learnrate
     * @param numUpdate Index of current batch
     */
    private void updateLernrate(int numUpdate) {
        epoch++;
        lastUpdateNum = numUpdate;
        this.lr = baseLr * (float) Math.pow(factor,epoch);
        System.out.println("Current Learnrate: " + lr + " Epoch: " + epoch + " numUpdates(Batch iterations): " + numUpdate + " Batch size: "+ batchSize);
    }

    /**
     * Only providing a new Value if a new Epoch has started: numUpdate  % (trainingDatasetSize / batchSize) == 0
     * @param numUpdate  Index of current batch
     * @return new LearnRate
     */
    @Override
    public float getNewValue(int numUpdate) {

        boolean didNotUpdate = lastUpdateNum != numUpdate;
        boolean isNewEpoch = numUpdate  % (trainingDatasetSize / batchSize) == 0;

        if(isNewEpoch && didNotUpdate){
            updateLernrate(numUpdate);
        }

        return lr;
    }
}
