package com.microsoft.aml.auth;

import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.nd4j.linalg.io.Assert;

/**
 * This is a test class for Authentication
 * 
 * @author Mufy, Abe
 * @Date 7/1/2022
 */

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class AMLAuthenticationTest {

	AMLAuthentication amlAuth = AMLAuthentication.getInstance();

	@Test
	public void testAMLAuthentication() throws Exception {

		String token = amlAuth.getAccessTokenFromUserCredentials();

		Assert.notNull(token);
	}

	@Test(expected = Exception.class)
	public void testAMLAuthenticationInvalid() throws Exception {

		AMLAuthentication.getInstance().getAccessTokenFromUserCredentials("", "", "");

	}
}
