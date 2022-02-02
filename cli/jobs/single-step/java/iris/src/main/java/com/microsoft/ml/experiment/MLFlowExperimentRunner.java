package com.microsoft.ml.experiment;

import com.microsoft.aml.auth.AMLAuthentication;
import com.microsoft.ml.mlflow.MLFLowRunner;

/**
 * This is the main class that runs the MlFlow experiment
 * 
 * @author mufy, Abe
 * @Date 7/1/2022
 */
public class MLflowExperimentRunner {

	public static void main(String args[]) throws Exception {

		String experimentName = "aml-mlflow-java-example";

		AMLAuthentication amlAuth = AMLAuthentication.getInstance();

		MLFLowRunner mlFLowRunner = new MLFLowRunner(amlAuth);

		MlflowExperiment mlflowExperiment = new MlflowExperiment(mlFLowRunner, experimentName);

		mlflowExperiment.trainDLModel(mlflowExperiment.initilizeDLmodel(),
				mlflowExperiment.prepDataset(mlflowExperiment.generateDataset()));

	}
}
