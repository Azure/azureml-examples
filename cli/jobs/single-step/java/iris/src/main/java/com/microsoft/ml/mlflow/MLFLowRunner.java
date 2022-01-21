package com.microsoft.ml.mlflow;

import org.mlflow.tracking.MlflowClient;
import org.mlflow.tracking.creds.BasicMlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCredsProvider;

import com.microsoft.aml.auth.AMLAuthentication;

/**
 * This class creates the MlFlow client to be used for tracking telemetry
 * 
 * @author Mufy, Abe
 * @Date 7/1/2022
 */
public class MLFLowRunner {

	private AMLAuthentication amlAuth;

	public MLFLowRunner(AMLAuthentication amlAuth) {

		this.amlAuth = amlAuth;
	}

	public MlflowClient getMlflowAuthClient() throws Exception {

		MlflowHostCredsProvider credsProvider = new BasicMlflowHostCreds(amlAuth.getTRACKING_URI(),
				amlAuth.getAccessTokenFromUserCredentials());
		MlflowClient client = new MlflowClient(credsProvider);

		return client;
	}
}
