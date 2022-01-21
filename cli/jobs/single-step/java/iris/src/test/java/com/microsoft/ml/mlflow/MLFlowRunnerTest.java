package com.microsoft.ml.mlflow;

import org.junit.Before;
import org.junit.Test;
import org.mlflow.tracking.MlflowClient;
import org.nd4j.linalg.io.Assert;

import com.microsoft.aml.auth.AMLAuthentication;

/**
 * This is a test class for Authentication
 * 
 * @author Mufy, Abe
 * @Date 7/1/2022
 */
public class MLflowRunnerTest {

	AMLAuthentication amlAuth;

	@Before
	public void getAuthentication() {
		amlAuth = AMLAuthentication.getInstance();
	}

	@Test
	public void testMLFlowRunner() throws Exception {

		MLFLowRunner mlFLowRunner = new MLFLowRunner(amlAuth);
		MlflowClient mlflowClient = mlFLowRunner.getMlflowAuthClient();

		Assert.notNull(mlflowClient);
	}

	@Test(expected = Exception.class)
	public void testInvalidAuthException() throws Exception {

		MLFLowRunner mlFLowRunner = new MLFLowRunner(null);
		MlflowClient mlflowClient = mlFLowRunner.getMlflowAuthClient();

		Assert.notNull(mlflowClient);
	}
}
