package mnist;

import ai.djl.training.tracker.Tracker;

public class Annealer implements Tracker {

    private final int batchsize;
    private final int trainingDatasetSize;
    private final float baseLr;
    private float lr;
    private final float factor;
    private int epoch = 1;
    private int lastUpdateNum = 0;

    Annealer(
            int batchsize,
            int trainingDatasetSize,
            float baseLr,
            float factor
    ){
        this.batchsize = batchsize;
        this.trainingDatasetSize = trainingDatasetSize;
        this.baseLr = baseLr;
        this.lr = baseLr;
        this.factor = factor;
    }

    void updateLernrate(int numUpdate) {
        epoch++;
        lastUpdateNum = numUpdate;
        lr = baseLr * (float) Math.pow(factor,epoch);
        System.out.println("Current Learnrate: " + lr + " Epoch: " + epoch + " numUpdates(Batch iterations): " + numUpdate + " Batch size: "+batchsize);
    }

    @Override
    public float getNewValue(int numUpdate) {

        boolean didNotUpdate = lastUpdateNum != numUpdate;
        boolean isNewEpoch = numUpdate  % (trainingDatasetSize / batchsize) == 0; // test ob -1 wie gewünscht

        if(isNewEpoch && didNotUpdate){
            updateLernrate(numUpdate);
        }

        return lr;
    }
}
