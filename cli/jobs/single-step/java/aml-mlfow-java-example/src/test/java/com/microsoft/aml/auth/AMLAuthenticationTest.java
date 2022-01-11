package com.microsoft.aml.auth;

import org.junit.Test;
import org.nd4j.linalg.io.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is a test class for Authentication
 * @author Mufy, Abe
 * @Date 7/1/2022
 */
public class AMLAuthenticationTest {

	private static final Logger log = LoggerFactory.getLogger(AMLAuthenticationTest.class);

	@Test
	public void testAMLAuthentication() throws Exception {

		AMLAuthentication amlAuth = AMLAuthentication.getInstnce();
		String token = amlAuth.getAccessTokenFromUserCredentials();

		Assert.notNull(token);
	}
}
