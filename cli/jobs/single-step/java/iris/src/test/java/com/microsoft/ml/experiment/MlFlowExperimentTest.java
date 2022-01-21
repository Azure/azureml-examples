package com.microsoft.ml.experiment;

import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.Assert;

import com.microsoft.aml.auth.AMLAuthentication;
import com.microsoft.ml.mlflow.MLFLowRunner;

/**
 * This is a test for MLFlow experiment
 * 
 * @author Mufy, Abe
 * @Date 7/1/2022
 */

public class MlflowExperimentTest {

	@Test
	public void testInputDataset() throws Exception {

		String experimentName = "aml-mlflow-javaexp";
		AMLAuthentication amlAuth = AMLAuthentication.getInstance();

		Assert.notNull(amlAuth);

		MLFLowRunner mlFLowRunner = new MLFLowRunner(amlAuth);
		MlflowExperiment mlflowExperiment = new MlflowExperiment(mlFLowRunner, experimentName);
		DataSetIterator itr = mlflowExperiment.generateDataset();

		Assert.isTrue(itr.numExamples() != 0);

		DataSet dt[] = mlflowExperiment.prepDataset(itr);
		mlflowExperiment.trainDLModel(mlflowExperiment.initilizeDLmodel(), dt);

	}

	@Test(expected = Exception.class)
	public void testInvalidExperimentDataException() throws Exception {

		String experimentName = "aml-mlflow-javaexp";
		AMLAuthentication amlAuth = AMLAuthentication.getInstance();

		Assert.notNull(amlAuth);

		MLFLowRunner mlFLowRunner = new MLFLowRunner(amlAuth);
		MlflowExperiment mlflowExperiment = new MlflowExperiment(mlFLowRunner, experimentName);
		DataSetIterator itr = mlflowExperiment.generateDataset();

		Assert.isTrue(itr.numExamples() != 0);

		DataSet dt[] = null;
		mlflowExperiment.trainDLModel(mlflowExperiment.initilizeDLmodel(), dt);

	}
}
