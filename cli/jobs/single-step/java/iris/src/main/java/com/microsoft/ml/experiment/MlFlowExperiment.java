/**
 * 
 * Author(s):  Mufy, Abe
 * 
 * Date: 7/1/2022
 */

package com.microsoft.ml.experiment;

import java.io.File;
import java.io.IOException;
import java.util.Optional;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.mlflow.api.proto.Service.Experiment;
import org.mlflow.api.proto.Service.RunInfo;
import org.mlflow.tracking.MlflowClient;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.microsoft.aml.auth.AMLAuthentication;
import com.microsoft.ml.mlflow.MLFLowRunner;

public class MlflowExperiment {

	private static final Logger log = LoggerFactory.getLogger(MlflowExperiment.class);

	private int batchSize = 150; // Test batch size
	private int nEpochs = 1000; // Number of training epochs
	private final int seed = 123; //
	private final int datasetSize = 150;
	private final int nInputs = 4;
	private final int nOutputs = 3;

	private String experimentName;

	private MLFLowRunner mlflowRunner;

	public MlflowExperiment(MLFLowRunner mlflowRunner, String experimentName) {

		this.mlflowRunner = mlflowRunner;
		this.experimentName = experimentName;

	}

	public DataSetIterator generateDataset() {

		DataSetIterator iris_dataset = new IrisDataSetIterator(batchSize, datasetSize);
		return iris_dataset;

	}

	public DataSetIterator generateDataset(String filePath) throws IOException, InterruptedException {

		int skipLines = 0;
		char delimiter = ',';

		RecordReader recordReader = new CSVRecordReader(skipLines, delimiter);
		recordReader.initialize(new FileSplit(new File(filePath)));

		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, this.getnInputs(),
				this.getnOutputs());

		return iterator;

	}

	public DataSet[] prepDataset(DataSetIterator irisDataset) {

		DataSet transformedDataset[] = new DataSet[2];

		DataSet allData = irisDataset.next();

		allData.shuffle();

		SplitTestAndTrain combinedDataset = allData.splitTestAndTrain(0.65);

		DataSet trainingData = combinedDataset.getTrain();

		// log.info(""+trainingData.getFeatures());

		DataSet testData = combinedDataset.getTest();

		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainingData);

		normalizer.transform(trainingData); // Apply normalization to the training data
		normalizer.transform(testData); //

		transformedDataset[0] = trainingData;
		transformedDataset[1] = testData;

		return transformedDataset;

	}

	public MultiLayerNetwork initilizeDLmodel() {

		MultiLayerConfiguration mlConf = new NeuralNetConfiguration.Builder().seed(this.getSeed())
				.weightInit(WeightInit.XAVIER).activation(Activation.RELU).updater(new Adam()).l2(1e-3).list()
				.layer(0, new DenseLayer.Builder().nIn(getnInputs()).nOut(10).build())
				.layer(1, new DenseLayer.Builder().nIn(10).nOut(20).build())
				.layer(2,
						new OutputLayer.Builder().nIn(20).nOut(getnOutputs())
								.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(mlConf);
		model.setInputMiniBatchSize(this.getBatchSize());
		model.init();

		return model;

	}

	public void trainDLModel(MultiLayerNetwork model, DataSet[] experimentDataset) throws Exception {

		MlflowClient mlflowClient = this.getMlflowRunner().getMlflowAuthClient();

		Optional<Experiment> experiment = mlflowClient.getExperimentByName(this.getExperimentName());
		String experimentId = experiment.isPresent() ? experiment.get().getExperimentId()
				: mlflowClient.createExperiment(this.getExperimentName());

		// Create a new tracked run in the above experiment
		RunInfo runInfo = mlflowClient.createRun(experimentId);
		String runId = runInfo.getRunUuid();

		runExperiment(mlflowClient, runId, model, experimentDataset);

	}

	private void runExperiment(MlflowClient mlflowClient, String runId, MultiLayerNetwork model,
			DataSet[] experimentDataset) {

		int frequency = 1;
		int printIteration = 10;

		model.setListeners(new ScoreIterationListener(printIteration),
				new EvaluativeListener(experimentDataset[1], frequency, InvocationType.EPOCH_END));

		log.info("" + experimentDataset[0].getFeatures());

		for (int i = 0; i < nEpochs; i++) {
			model.fit(experimentDataset[0]);

		}

		Evaluation evaluateModel = new Evaluation(this.getnOutputs());

		INDArray modelOutput = model.output(experimentDataset[1].getFeatures());

		evaluateModel.eval(experimentDataset[1].getLabels(), modelOutput);

		// log.info(eval.stats());

		log.info("accuracy: " + evaluateModel.accuracy());
		log.info("precision: " + evaluateModel.precision());
		log.info("recall: " + evaluateModel.recall());

		mlflowClient.logMetric(runId, "accuracy", evaluateModel.accuracy());
		mlflowClient.logMetric(runId, "precision", evaluateModel.precision());
		mlflowClient.logMetric(runId, "recall", evaluateModel.recall());

		String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "model.zip");

		log.info("Saving model to tmp folder: " + path);
		// model.save(new File(path), true);
		// mlflowClient.logArtifact(runId, new File(path));

		log.info("****************Experiment finished********************");

		mlflowClient.setTerminated(runId);

	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public int getnEpochs() {
		return nEpochs;
	}

	public void setnEpochs(int nEpochs) {
		this.nEpochs = nEpochs;
	}

	public int getSeed() {
		return seed;
	}

	public int getDatasetSize() {
		return datasetSize;
	}

	public int getnInputs() {
		return nInputs;
	}

	public int getnOutputs() {
		return nOutputs;
	}

	public MLFLowRunner getMlflowRunner() {
		return mlflowRunner;
	}

	public void setMlflowRunner(MLFLowRunner mlflowRunner) {
		this.mlflowRunner = mlflowRunner;
	}

	public String getExperimentName() {
		return experimentName;
	}

	public void setExperimentName(String experimentName) {
		this.experimentName = experimentName;
	}

}
