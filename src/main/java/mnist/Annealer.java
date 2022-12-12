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

    void updateLernrate(int numUpdate) {
        epoch++;
        lastUpdateNum = numUpdate;
        lr = baseLr * (float) Math.pow(factor,epoch);
        System.out.println("Current Learnrate: " + lr + " Epoch: " + epoch + " numUpdates(Batch iterations): " + numUpdate + " Batch size: "+ batchSize);
    }

    @Override
    public float getNewValue(int numUpdate) {

        boolean didNotUpdate = lastUpdateNum != numUpdate;
        boolean isNewEpoch = numUpdate  % (trainingDatasetSize / batchSize) == 0; // test ob -1 wie gewünscht

        if(isNewEpoch && didNotUpdate){
            updateLernrate(numUpdate);
        }

        return lr;
    }
}
