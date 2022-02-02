package com.microsoft.aml.auth;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.microsoft.aad.adal4j.AuthenticationContext;
import com.microsoft.aad.adal4j.AuthenticationResult;
import com.microsoft.aad.adal4j.ClientCredential;
import com.microsoft.ml.experiment.MlflowExperiment;

/**
 * This class is used to authenticate with AD and retrieve access token to be
 * used with MlFLow
 * 
 * @author Mufy, Abe
 * @Date 7/1/2022
 */
public class AMLAuthentication {

	private static AMLAuthentication amlAuth;

	private static final Logger log = LoggerFactory.getLogger(AMLAuthentication.class);

	private AMLAuthentication() {
	}

	private final String TARGET_RESOURCE = "https://management.core.windows.net/";

	private final String _tenantId = System.getenv("TENANT_ID");
	private final String AUTHORITY = String.format("https://login.microsoftonline.com/%s/oauth2/v2.0/token", _tenantId);
	private final String CLIENT_ID = System.getenv("CLIENT_ID");;
	private final String SECRET_UNUSED_LEFT_FOR_CLARITY = System.getenv("AZUREMLFLOW_SP_CLIENT_SECRET");

	private final String _region = System.getenv("REGION");
	private final String _subscriptionId = System.getenv("SUB_ID");
	private final String _resourceGroup = System.getenv("RESOURCE_GROUP");
	private final String _workspaceName = System.getenv("WORKSPACE_NAME");
	private final String TRACKING_URI = String.format(
			"https://%s.api.azureml.ms/mlflow/v1.0/subscriptions/%s/resourceGroups/%s/providers/Microsoft.MachineLearningServices/workspaces/%s",
			_region, _subscriptionId, _resourceGroup, _workspaceName);

	public static AMLAuthentication getInstance() {

		if (amlAuth == null) {

			amlAuth = new AMLAuthentication();
		}

		return amlAuth;
	}

	/**
	 * This method returns the access token. Note, you should consider using the
	 * Azure Identity package for auth since this is the Azure standard package for
	 * auth
	 * https://docs.microsoft.com/en-us/java/api/overview/azure/identity-readme?view=azure-java-stable
	 * 
	 * @return String
	 * @throws Exception
	 */
	public String getAccessTokenFromUserCredentials() throws Exception {

		log.info("details: " + AUTHORITY + TARGET_RESOURCE + CLIENT_ID);

		AuthenticationResult result = getAccessTokenResultFromUserCredentials(AUTHORITY, TARGET_RESOURCE, CLIENT_ID);

		return result.getAccessToken();

	}

	public String getAccessTokenFromUserCredentials(String authority, String targetResource, String clientId)
			throws Exception {

		log.info("details: " + AUTHORITY + TARGET_RESOURCE + CLIENT_ID);

		AuthenticationResult result = getAccessTokenResultFromUserCredentials(authority, targetResource, clientId);

		return result.getAccessToken();

	}

	private AuthenticationResult getAccessTokenResultFromUserCredentials(String loginAuthorityUrl,
			String targetResourceForToken, String clientId) throws Exception {
		AuthenticationResult result;
		ExecutorService service = null;
		try {
			service = Executors.newFixedThreadPool(1);
			AuthenticationContext context = new AuthenticationContext(loginAuthorityUrl, false, service);
			Future<AuthenticationResult> future = context.acquireToken(targetResourceForToken,
					new ClientCredential(clientId, System.getenv("AZUREMLFLOW_SP_CLIENT_SECRET")), null);
			result = future.get();
		} finally {
			if (service != null) {
				service.shutdown();
			}
		}

		if (result == null) {
			throw new Exception("Authentication result was null");
		}
		return result;
	}

	public String getTRACKING_URI() {
		return TRACKING_URI;
	}

	public String getCLIENT_ID() {
		return CLIENT_ID;
	}
}
