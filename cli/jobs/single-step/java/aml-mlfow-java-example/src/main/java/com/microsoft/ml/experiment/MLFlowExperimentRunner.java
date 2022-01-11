package com.microsoft.ml.experiment;

import com.microsoft.aml.auth.AMLAuthentication;
import com.microsoft.ml.mlflow.MLFLowRunner;

/**
 * This is the main class that runs the MlFlow experiment 
 * @author mufy, Abe
 * @Date 7/1/2022
 */
public class MLFlowExperimentRunner {
	
	
	public static void main(String args[]) throws Exception {
				
		String experimentName = "aml-mlflow-java-example";
		
		AMLAuthentication amlAuth = AMLAuthentication.getInstnce();
		
		MLFLowRunner mlFLowRunner = new MLFLowRunner(amlAuth);
		
		MlFlowExperiment mlflowExperiment = new MlFlowExperiment(mlFLowRunner,experimentName);
		
		mlflowExperiment.trainDLModel(mlflowExperiment.initilizeDLmodel(), mlflowExperiment.prepDataset(mlflowExperiment.generateDataset()));
		
	}
}
